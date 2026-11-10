"""명령줄 진입점: `lora-demo` — 6단계로 진행 상황을 보여 주며 전체 파이프라인을 실행한다."""

import argparse
import csv
import gc
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from .config import MODEL_ID, MODEL_REVISION, RunConfig
from .data import DataError, validate_jsonl


def _at_least_one(value: str) -> int:
    """--keep-runs 검증: 1 이상의 정수만 받는다 (0 이하면 방금 만든 결과까지 지워짐)."""
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"정수(1 이상)여야 합니다: {value!r}") from None
    if n < 1:
        raise argparse.ArgumentTypeError("1 이상이어야 합니다 (최근 실행 최소 1개는 남깁니다)")
    return n


def _positive_int(value: str) -> int:
    """--rank·--epochs 검증: 1 이상의 정수만 받는다."""
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"정수(1 이상)여야 합니다: {value!r}") from None
    if n < 1:
        raise argparse.ArgumentTypeError(f"1 이상이어야 합니다 (받은 값: {n})")
    return n


def _positive_float(value: str) -> float:
    """--lr 검증: 0보다 큰 유한한 실수만 받는다."""
    try:
        x = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"실수(0보다 큼)여야 합니다: {value!r}") from None
    if not (x > 0 and x != float("inf")):
        raise argparse.ArgumentTypeError(f"0보다 큰 유한한 값이어야 합니다 (받은 값: {value})")
    return x


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="lora-demo", description="작은 언어 모델에 LoRA 로 냥체 말투를 입히고 전/후를 보고서로 확인한다.")
    p.add_argument("--mock", action="store_true", help="무작위 초소형 모델로 네트워크 없이 전체 흐름만 확인(수 초)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"], help="기본 auto: CUDA > MPS > CPU")
    p.add_argument("--rank", type=_positive_int, default=8, help="LoRA 순위 r (기본 8)")
    p.add_argument("--targets", default="all-linear", choices=["all-linear", "qv"], help="LoRA 를 붙일 층 (기본 all-linear)")
    p.add_argument("--quick", action="store_true", help="1에폭만 학습(느린 PC용)")
    p.add_argument("--epochs", type=_positive_int, default=None, help="에폭 수(기본 3)")
    p.add_argument("--lr", type=_positive_float, default=None, help="학습률(기본 3e-4)")
    p.add_argument("--data", type=Path, default=None, help="내 학습 데이터 JSONL (messages 형식). 지정하면 말투 적중률은 계산하지 않음")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--runs-dir", type=Path, default=Path("runs"), help="결과 폴더 (기본 ./runs)")
    p.add_argument("--keep-runs", type=_at_least_one, default=5, help="runs/ 에 남길 최근 실행 수 (1 이상, 기본 5)")
    p.add_argument("--max-steps", type=int, default=0, help="이 스텝에서 학습을 멈춤(0이면 에폭 기준)")
    return p.parse_args(argv)


def config_from_args(a: argparse.Namespace) -> RunConfig:
    cfg = RunConfig(mock=a.mock, device=a.device, rank=a.rank, alpha=2 * a.rank, targets=a.targets, seed=a.seed,
                    runs_dir=a.runs_dir, keep_runs=a.keep_runs, max_steps=a.max_steps)
    if a.quick:
        cfg.epochs = 1
    if a.epochs is not None:
        cfg.epochs = a.epochs
    if a.lr is not None:
        cfg.lr = a.lr
    if a.data:
        cfg.data = a.data
    if cfg.mock:
        cfg.apply_mock_defaults()
        if a.lr is not None:
            cfg.lr = a.lr
    return cfg


def new_run_dir(runs_dir: Path) -> Path:
    """runs/<날짜-시각>/ . 같은 초에 또 만들면 -2, -3 을 붙인다."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    base = datetime.now().strftime("%Y%m%d-%H%M%S")
    d, n = runs_dir / base, 1
    while d.exists():
        n += 1
        d = runs_dir / f"{base}-{n}"
    d.mkdir()
    return d


def prune_runs(runs_dir: Path, keep: int) -> None:
    """실행 결과가 무한히 쌓이지 않도록 최근 keep 개만 남긴다 (어댑터가 17MB씩이라)."""
    dirs = sorted(d for d in runs_dir.iterdir() if d.is_dir() and re.fullmatch(r"\d{8}-\d{6}(-\d+)?", d.name))
    keep = max(1, keep)  # 방금 만든 최신 실행은 항상 남긴다
    for old in dirs[: max(0, len(dirs) - keep)]:
        shutil.rmtree(old)


def fmt_sec(s: float) -> str:
    return f"{int(s // 60)}분 {int(s % 60)}초" if s >= 60 else f"{s:.1f}초"


def run_pipeline(cfg: RunConfig, say=print) -> Path:
    """전체 파이프라인. 만든 실행 폴더 경로를 돌려준다."""
    import torch

    from .data import encode, load_jsonl
    from .evaluate import adapter_off_check, generate, prompt_baseline, style_hit
    from .lora import attach_lora, count_params, rank_table
    from .model import build_mock, load_real, pick_device
    from .report import render
    from .train import train

    import peft
    import transformers

    T: dict[str, float] = {}
    t_all = time.time()
    torch.manual_seed(cfg.seed)
    if cfg.mock:
        os.environ["HF_HUB_OFFLINE"] = "1"                # 모의 모드는 네트워크를 쓰지 않는다
        torch.use_deterministic_algorithms(True)          # 같은 입력 → 같은 결과
    model = None
    try:
        # ── 1. 장치 확인 ──────────────────────────────────────────────
        device, dtype = pick_device(cfg.device)
        say(f"[1/6] 장치 확인: {device.type} ({str(dtype).replace('torch.', '')})" + ("  — 모의 모드" if cfg.mock else ""))

        # ── 2. 모델 준비 ──────────────────────────────────────────────
        t = time.time()
        if cfg.mock:
            say("[2/6] 모의 모델 만들기 (무작위 초소형 모델, 네트워크 없음)")
            tok, model = build_mock(cfg)
            model = model.to(device)
        else:
            say(f"[2/6] 모델 준비: {MODEL_ID} (처음 한 번 약 1GB 내려받기)")
            tok, model = load_real(device, dtype)
        T["load_s"] = time.time() - t
        model_total = sum(p.numel() for p in model.parameters())
        table = rank_table(model)  # 학습 없이 공식으로 계산하는 표 (붙이기 전 원본 기준)

        # 데이터 읽기
        rows = load_jsonl(cfg.data)
        eval_prompts = json.loads(cfg.eval_prompts.read_text(encoding="utf-8"))
        custom_data = cfg.data.name != "nyang_100.jsonl"
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        enc = [encode(tok, r["messages"]) for r in rows]
        say(f"       학습 데이터 {len(rows)}문항, 평가 질문 {len(eval_prompts)}개")

        # ── 3. 학습 전 답변 ───────────────────────────────────────────
        t = time.time()
        before = generate(model, tok, eval_prompts, device, cfg.max_new_tokens)
        T["gen_before_s"] = time.time() - t
        t = time.time()
        baseline = prompt_baseline(model, tok, eval_prompts, device, cfg.max_new_tokens)
        T["baseline_s"] = time.time() - t
        say(f"[3/6] 학습 전 답변 {len(before)}개 (예: {eval_prompts[0]!r} → {before[0][:40]!r})")

        # ── 4. LoRA 붙이기 ────────────────────────────────────────────
        model = attach_lora(model, cfg.rank, cfg.alpha, cfg.dropout, cfg.targets)
        trainable, _ = count_params(model)
        pct = 100 * trainable / model_total
        say(f"[4/6] LoRA 붙이기 (r={cfg.rank}, {cfg.targets}): 학습 파라미터 {trainable:,} / {model_total:,} ({pct:.2f}%)")

        # ── 5. 학습 ───────────────────────────────────────────────────
        micro_per_epoch = -(-len(enc) // cfg.micro_batch)          # 올림 나눗셈
        steps_per_epoch = -(-micro_per_epoch // cfg.grad_accum)
        total_steps = steps_per_epoch * cfg.epochs
        if cfg.max_steps:
            total_steps = min(total_steps, cfg.max_steps)
        snap_steps = sorted({s for s in (10, 25, total_steps) if s <= total_steps})
        snap_q = [eval_prompts[i] for i in cfg.snapshot_questions if i < len(eval_prompts)]
        snapshots = [{"step": 0, "questions": snap_q, "answers": [before[i] for i in cfg.snapshot_questions if i < len(eval_prompts)]}]
        t_train = time.time()

        def on_step(step, losses):
            if step == 1 or step % 5 == 0 or step == total_steps:
                say(f"[5/6] 학습 step {step}/{total_steps}  loss {losses[-1]:.2f}  (경과 {fmt_sec(time.time() - t_train)})")
            if step in snap_steps:  # 같은 질문으로 학습 중간 답변을 엿본다
                snapshots.append({"step": step, "questions": snap_q, "answers": generate(model, tok, snap_q, device, cfg.max_new_tokens)})
                model.train()

        losses, train_s = train(model, enc, pad_id, device, cfg, on_step)
        T["train_s"] = train_s

        # ── 6. 학습 후 답변·저장·보고서 ───────────────────────────────
        run_dir = new_run_dir(cfg.runs_dir)
        model.save_pretrained(run_dir / "adapter")
        adapter_bytes = (run_dir / "adapter" / "adapter_model.safetensors").stat().st_size
        t = time.time()
        after = generate(model, tok, eval_prompts, device, cfg.max_new_tokens)
        T["gen_after_s"] = time.time() - t
        off_same = adapter_off_check(model, tok, eval_prompts, device, cfg.max_new_tokens, before)
        T["total_s"] = time.time() - t_all

        hits = (lambda xs: None if custom_data else sum(style_hit(x) for x in xs))
        result = {
            "mode": "mock" if cfg.mock else "real",
            "device": device.type,
            "dtype": str(dtype).replace("torch.", ""),
            "model": {"id": MODEL_ID if not cfg.mock else "mock-tiny-qwen2", "revision": MODEL_REVISION if not cfg.mock else None,
                      "total_params": model_total, "size_bytes": model_total * 2},  # 원본은 bfloat16(2바이트)
            "lora": {"r": cfg.rank, "alpha": cfg.alpha, "targets": cfg.targets, "trainable_params": trainable,
                     "trainable_pct": round(pct, 4), "adapter_bytes": adapter_bytes},
            "train": {"steps": len(losses), "losses": losses, "seconds": round(train_s, 1), "lr": cfg.lr,
                      "epochs": cfg.epochs, "micro_batch": cfg.micro_batch, "grad_accum": cfg.grad_accum, "seed": cfg.seed,
                      "examples": len(rows)},
            "eval": {"prompts": eval_prompts, "before": before, "after": after, "prompt_baseline": baseline,
                     "style_check": not custom_data, "style_hit_before": hits(before), "style_hit_after": hits(after),
                     "style_hit_prompt": hits(baseline)},
            "snapshots": snapshots,
            "adapter_off_equals_before": off_same,
            "rank_table": table,
            "timing": {k: round(v, 1) for k, v in T.items()},
            "versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                         "transformers": transformers.__version__, "peft": peft.__version__},
            "run_name": run_dir.name,
        }
        (run_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        with open(run_dir / "train_log.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["step", "loss"])
            w.writerows((i + 1, v) for i, v in enumerate(losses))
        (run_dir / "report.html").write_text(render(result), encoding="utf-8")
        prune_runs(cfg.runs_dir, cfg.keep_runs)

        say(f"[6/6] 저장 완료: {cfg.runs_dir.name}/{run_dir.name}/report.html")
        if hits(after) is not None:
            say(f"       말투 적중 {hits(before)}/{len(before)} → {hits(after)}/{len(after)}  (프롬프트만: {hits(baseline)}/{len(before)})")
        say(f"       어댑터 {adapter_bytes / 1e6:.1f}MB (원본 모델 {model_total * 2 / 1e6:.1f}MB), 학습 {fmt_sec(train_s)}, 전체 {fmt_sec(T['total_s'])}")
        say(f"       어댑터 끄면 학습 전과 같은 답: {off_same}")
        return run_dir
    finally:
        # 메모리 반환: 모델·옵티마이저 참조를 놓고 가속기 캐시를 비운다
        model = None
        gc.collect()
        if cfg.mock:
            torch.use_deterministic_algorithms(False)
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows 콘솔에서도 한글·이모지가 깨지지 않게
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    cfg = config_from_args(parse_args(argv))
    try:
        validate_jsonl(cfg.data)  # 모델 준비 전에 데이터 형식부터 확인 (오류는 줄 번호와 함께)
    except DataError as e:
        print(f"오류: {e}", file=sys.stderr)
        return 2
    try:
        run_pipeline(cfg)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("메모리가 부족합니다. `--device cpu` 로 다시 실행해 보세요.", file=sys.stderr)
            return 1
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
