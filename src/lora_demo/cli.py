"""명령줄 진입점: `lora-demo` — 6단계로 진행 상황을 보여 주며 전체 파이프라인을 실행한다."""

import argparse
import csv
import gc
import json
import os
import re
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

from .config import MODEL_ID, MODEL_REVISION, RunConfig
from .data import DataError, validate_jsonl
from .i18n import LANGS, resolve_lang
from .i18n import t as msg


def _prescan_lang(argv) -> str | None:
    """Find `--lang X` / `--lang=X` in argv before full parsing, so --help and type errors use that language."""
    args = list(sys.argv[1:] if argv is None else argv)
    for i, a in enumerate(args):
        if a == "--lang" and i + 1 < len(args):
            return args[i + 1]
        if a.startswith("--lang="):
            return a.split("=", 1)[1]
    return None


def _validators(lang: str):
    """Build the argparse type checkers for one language."""

    def at_least_one(value: str) -> int:
        """--keep-runs: integer >= 1 (0 or less would delete the run just created)."""
        try:
            n = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(msg("arg_int", lang, value=value)) from None
        if n < 1:
            raise argparse.ArgumentTypeError(msg("arg_keep", lang))
        return n

    def positive_int(value: str) -> int:
        """--rank / --epochs: integer >= 1."""
        try:
            n = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(msg("arg_int", lang, value=value)) from None
        if n < 1:
            raise argparse.ArgumentTypeError(msg("arg_min1", lang, n=n))
        return n

    def positive_float(value: str) -> float:
        """--lr: finite real number > 0."""
        try:
            x = float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(msg("arg_float", lang, value=value)) from None
        if not (x > 0 and x != float("inf")):
            raise argparse.ArgumentTypeError(msg("arg_pos_float", lang, value=value))
        return x

    return at_least_one, positive_int, positive_float


def parse_args(argv=None) -> argparse.Namespace:
    """Parse arguments. The returned namespace's ``lang`` is the resolved language ('en' or 'ko')."""
    lang = resolve_lang(_prescan_lang(argv))
    at_least_one, positive_int, positive_float = _validators(lang)
    p = argparse.ArgumentParser(prog="lora-demo", description=msg("cli_description", lang), epilog=msg("cli_epilog", lang))
    p.add_argument("--lang", choices=list(LANGS), default=None, help=msg("h_lang", lang))
    p.add_argument("--mock", action="store_true", help=msg("h_mock", lang))
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"], help=msg("h_device", lang))
    p.add_argument("--rank", type=positive_int, default=8, help=msg("h_rank", lang))
    p.add_argument("--targets", default="all-linear", choices=["all-linear", "qv"], help=msg("h_targets", lang))
    p.add_argument("--quick", action="store_true", help=msg("h_quick", lang))
    p.add_argument("--epochs", type=positive_int, default=None, help=msg("h_epochs", lang))
    p.add_argument("--lr", type=positive_float, default=None, help=msg("h_lr", lang))
    p.add_argument("--data", type=Path, default=None, help=msg("h_data", lang))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--runs-dir", type=Path, default=Path("runs"), help=msg("h_runs_dir", lang))
    p.add_argument("--keep-runs", type=at_least_one, default=5, help=msg("h_keep_runs", lang))
    p.add_argument("--max-steps", type=int, default=0, help=msg("h_max_steps", lang))
    ns = p.parse_args(argv)
    ns.lang = resolve_lang(ns.lang)
    return ns


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


class RunsDirError(Exception):
    """The results folder cannot be created (permissions, a file in the way ...)."""

    def __init__(self, path: Path, error: OSError):
        super().__init__(f"{path}: {error.__class__.__name__}")
        self.path, self.error = path, error


def check_runs_dir(runs_dir: Path) -> None:
    """학습 전에 결과 폴더를 만들고 실제로 파일을 쓸 수 있는지 확인한다.

    실제 모드는 모델 다운로드와 몇 분의 학습이 걸리므로, 저장 단계에서야 실패해 결과를 버리지 않게 미리 막는다.
    권한 비트만 보는 os.access 대신 임시 파일을 실제로 만들어 본다(읽기 전용 볼륨·root 등도 정확히 잡힌다).
    """
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=runs_dir):  # 만들자마자 닫히며 지워진다
            pass
    except OSError as err:
        raise RunsDirError(runs_dir, err) from err


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


def fmt_sec(s: float, lang: str = "en") -> str:
    return msg("min_sec_unit", lang, m=int(s // 60), s=int(s % 60)) if s >= 60 else msg("sec_unit", lang, s=s)


def run_pipeline(cfg: RunConfig, say=print, lang: str | None = None) -> Path:
    """전체 파이프라인. 만든 실행 폴더 경로를 돌려준다. lang 이 없으면 환경에서 정한다."""
    lang = lang or resolve_lang()
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
        say(msg("p_device", lang, device=device.type, dtype=str(dtype).replace("torch.", "")) + (msg("p_mock_suffix", lang) if cfg.mock else ""))

        # ── 2. 모델 준비 ──────────────────────────────────────────────
        t = time.time()
        if cfg.mock:
            say(msg("p_mock_model", lang))
            tok, model = build_mock(cfg)
            model = model.to(device)
        else:
            say(msg("p_real_model", lang, model=MODEL_ID))
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
        say(msg("p_data", lang, rows=len(rows), prompts=len(eval_prompts)))

        # ── 3. 학습 전 답변 ───────────────────────────────────────────
        t = time.time()
        before = generate(model, tok, eval_prompts, device, cfg.max_new_tokens)
        T["gen_before_s"] = time.time() - t
        t = time.time()
        baseline = prompt_baseline(model, tok, eval_prompts, device, cfg.max_new_tokens)
        T["baseline_s"] = time.time() - t
        say(msg("p_before", lang, n=len(before), q=eval_prompts[0], a=before[0][:40]))

        # ── 4. LoRA 붙이기 ────────────────────────────────────────────
        model = attach_lora(model, cfg.rank, cfg.alpha, cfg.dropout, cfg.targets)
        trainable, _ = count_params(model)
        pct = 100 * trainable / model_total
        say(msg("p_lora", lang, r=cfg.rank, targets=cfg.targets, trainable=trainable, total=model_total, pct=pct))

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
                say(msg("p_step", lang, step=step, total=total_steps, loss=losses[-1], elapsed=fmt_sec(time.time() - t_train, lang)))
            if step in snap_steps:  # 같은 질문으로 학습 중간 답변을 엿본다
                snapshots.append({"step": step, "questions": snap_q, "answers": generate(model, tok, snap_q, device, cfg.max_new_tokens)})
                model.train()

        losses, train_s = train(model, enc, pad_id, device, cfg, on_step)
        T["train_s"] = train_s

        # ── 6. 학습 후 답변·저장·보고서 ───────────────────────────────
        try:
            run_dir = new_run_dir(cfg.runs_dir)
        except OSError as err:
            raise RunsDirError(cfg.runs_dir, err) from err
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
        (run_dir / "report.html").write_text(render(result, lang), encoding="utf-8")
        prune_runs(cfg.runs_dir, cfg.keep_runs)

        say(msg("p_saved", lang, path=f"{cfg.runs_dir.name}/{run_dir.name}/report.html"))
        if hits(after) is not None:
            say(msg("p_hits", lang, b=hits(before), a=hits(after), p=hits(baseline), n=len(before)))
        say(msg("p_sizes", lang, adapter=adapter_bytes / 1e6, base=model_total * 2 / 1e6, train=fmt_sec(train_s, lang), total=fmt_sec(T["total_s"], lang)))
        say(msg("p_off", lang, same=off_same))
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
    args = parse_args(argv)
    lang = args.lang
    cfg = config_from_args(args)
    prefix = msg("error_prefix", lang)
    try:
        validate_jsonl(cfg.data)  # 모델 준비 전에 데이터 형식부터 확인 (오류는 줄 번호와 함께)
        check_runs_dir(cfg.runs_dir)  # 결과를 저장할 수 있는지도 학습 전에 확인 (날짜-시각 하위 폴더는 저장 시점에 만든다)
        run_pipeline(cfg, lang=lang)
    except DataError as e:
        print(f"{prefix}{e.render(lang)}", file=sys.stderr)
        return 2
    except RunsDirError as e:
        print(f"{prefix}{msg('runs_dir_unwritable', lang, path=e.path, error=e.error.__class__.__name__)}", file=sys.stderr)
        return 2
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(msg("oom", lang), file=sys.stderr)
            return 1
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
