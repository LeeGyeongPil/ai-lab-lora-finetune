"""Message tables (English / Korean) and language resolution for the CLI and the HTML report.

Language order: ``--lang`` > ``LORA_DEMO_LANG`` > ``LC_ALL`` > ``LC_MESSAGES`` > ``LANG``.
A value starting with ``ko`` selects Korean; anything else (unset, ``C``, ``en_US`` ...) selects English.
"""

from collections.abc import Mapping

LANGS = ("en", "ko")
ENV_VAR = "LORA_DEMO_LANG"
_LOCALE_VARS = ("LC_ALL", "LC_MESSAGES", "LANG")


def _norm(value: str | None) -> str | None:
    """Return 'ko'/'en' for a raw value, or None when it is empty (so the next source is consulted)."""
    if not value or not value.strip():
        return None
    return "ko" if value.strip().lower().startswith("ko") else "en"


def resolve_lang(cli_value: str | None = None, environ: Mapping[str, str] | None = None) -> str:
    """Pick the UI language. See the module docstring for the order."""
    import os

    env = os.environ if environ is None else environ
    for raw in (cli_value, env.get(ENV_VAR), *(env.get(k) for k in _LOCALE_VARS)):
        picked = _norm(raw)
        if picked:
            return picked
    return "en"


def t(key: str, lang: str, **params) -> str:
    """Render message ``key`` in ``lang`` (falls back to English for an unknown language)."""
    table = MESSAGES.get(lang, MESSAGES["en"])
    return table[key].format(**params)


MESSAGES: dict[str, dict[str, str]] = {
    "en": {
        # ---- CLI: argparse ----
        "cli_description": "Give a small language model a cat-speak style with LoRA and compare before/after in a report.",
        "cli_epilog": "Language: --lang > LORA_DEMO_LANG > LC_ALL > LC_MESSAGES > LANG (anything not Korean -> English).",
        "h_lang": "Output language en|ko (default: LORA_DEMO_LANG, then LC_ALL/LC_MESSAGES/LANG, else en)",
        "h_mock": "Run a random tiny model with no network, just to check the whole flow (a few seconds)",
        "h_device": "Default auto: CUDA > MPS > CPU",
        "h_rank": "LoRA rank r (default 8)",
        "h_targets": "Layers to attach LoRA to (default all-linear)",
        "h_quick": "Train for 1 epoch only (for slow PCs)",
        "h_epochs": "Number of epochs (default 3)",
        "h_lr": "Learning rate (default 3e-4)",
        "h_data": "Your own training data JSONL (messages format). When set, the style hit rate is not computed",
        "h_runs_dir": "Results folder (default ./runs)",
        "h_keep_runs": "Number of recent runs to keep in runs/ (1 or more, default 5)",
        "h_max_steps": "Stop training at this step (0 = by epochs)",
        "arg_int": "must be an integer (1 or more): {value!r}",
        "arg_keep": "must be 1 or more (at least the latest run is kept)",
        "arg_min1": "must be 1 or more (got: {n})",
        "arg_float": "must be a real number greater than 0: {value!r}",
        "arg_pos_float": "must be a finite value greater than 0 (got: {value})",
        # ---- CLI: runtime ----
        "error_prefix": "error: ",
        "oom": "Out of memory. Try again with `--device cpu`.",
        "runs_dir_unwritable": "{path}: cannot create the results folder ({error}) - check --runs-dir",
        "report_usage": "usage: python -m lora_demo.report <result.json> <out.html>",
        "report_input_error": "{path}: cannot read result.json ({error}) - check the path",
        "report_not_result": "{path}: not a result.json from lora-demo ({error}) - pass the result.json inside a runs/<time>/ folder",
        "report_output_error": "{path}: cannot write the report ({error}) - check the output path",
        "sec_unit": "{s:.1f}s",
        "min_sec_unit": "{m}m {s}s",
        "p_device": "[1/6] Device check: {device} ({dtype})",
        "p_mock_suffix": "  - mock mode",
        "p_mock_model": "[2/6] Building mock model (random tiny model, no network)",
        "p_real_model": "[2/6] Preparing model: {model} (about 1GB downloaded the first time)",
        "p_data": "       {rows} training examples, {prompts} evaluation questions",
        "p_before": "[3/6] {n} answers before training (e.g. {q!r} -> {a!r})",
        "p_lora": "[4/6] Attaching LoRA (r={r}, {targets}): trainable params {trainable:,} / {total:,} ({pct:.2f}%)",
        "p_step": "[5/6] Training step {step}/{total}  loss {loss:.2f}  (elapsed {elapsed})",
        "p_saved": "[6/6] Saved: {path}",
        "p_hits": "       Style hits {b}/{n} -> {a}/{n}  (prompt only: {p}/{n})",
        "p_sizes": "       Adapter {adapter:.1f}MB (base model {base:.1f}MB), training {train}, total {total}",
        "p_off": "       Adapter off gives the same answers as before training: {same}",
        # ---- data errors (placeholders: where = 'file' or 'file:N') ----
        "where_line": "{path}:{line}",
        "file_missing": "{where}: file not found. Check the --data path",
        "read_failed": "{where}: cannot read the file ({error}) - make sure it is a UTF-8 text file",
        "not_json": '{where}: not JSON. Write one {{"messages":[...]}} per line',
        "no_messages": '{where}: "messages" list is missing or empty',
        "bad_item": '{where}: each item of messages must be {{"role":...,"content":...}}',
        "missing_field": '{where}: message {index} needs both string "role" and "content"',
        "last_role": '{where}: the last message must have role "assistant" (an answer is required)',
        "no_user": '{where}: at least one "user" message must come before the assistant answer (cannot train without a question)',
        "empty_file": "{where}: the training data is empty",
        "template_mismatch": "The prompt tokens are not a prefix of the full tokens (chat template mismatch)",
        # ---- HTML report ----
        "r_title": "LoRA fine-tuning report",
        "r_sub": "{model} · device {device} ({dtype}) · run {run}",
        "r_mock": "<b>Mock mode</b> - a random tiny model, so answer quality is meaningless. This run only checks that the pipeline works.",
        "r_h1": "1. How light is it?",
        "r_params": "Parameters",
        "r_filesize": "File size",
        "r_pct": "Only <b>{pct:.2f}%</b> of the model was trained (r={r}, targets {targets})",
        "r_adapter_note": "Saving only the trained result (the adapter) takes a fraction of the base model's size. The base model file is untouched.",
        "r_h2": "2. Loss curve",
        "r_loss_detail": "{first:.2f} -> {last:.2f} ({steps} steps, trained {seconds}s, micro-batch {mb} × accumulation {ga}, learning rate {lr})",
        "r_axis_steps": "{n} steps",
        "r_aria_loss": "Loss curve",
        "r_h3": "3. Answers before/after training",
        "r_th_question": "Question (not in the training data)",
        "r_th_before": "Before",
        "r_th_prompt": "Prompt only",
        "r_th_after": "After",
        "r_th_hit": "Style hit",
        "r_empty": "(empty answer)",
        "r_off_q": "With the adapter off, do the answers equal the pre-training ones: ",
        "r_yes": "yes",
        "r_no": "no",
        "r_off_note": " (evidence that the original weights are untouched and the adapter can be switched on and off)",
        "r_h4": "4. Training timeline",
        "r_h5": "5. Rank table (computed by formula, no training)",
        "r_th_qv": "q/v layers only: params",
        "r_th_file": "File",
        "r_th_all": "All linear layers: params",
        "r_formula": "Formula: for each layer, r × (input dim + output dim) parameters, summed. File size is an estimate assuming float32 (4 bytes).",
        "r_h6": "6. Limits - it learns style, not knowledge",
        "r_l1": "<li>LoRA learns <b>how to speak</b> (tone, format) quickly. A few dozen steps are enough.</li>",
        "r_l2": "<li>But it is bad at memorising <b>new facts</b>. Check the answers to the questions about the fictional setting (Monggeuli, Cloud Forest Library) in the table above.</li>",
        "r_l3": "<li>Even when the style is right, the content can be a plausible wrong answer. A high style hit rate does not mean the content is correct.</li>",
        "r_l4": "<li>With a small model and 100 examples, answers can vary between runs and settings.</li>",
        "r_h7": "7. Run info",
        "r_time": "<li>Time: model {load}s · answers before {gb}s · training {train}s · answers after {ga}s · total {total}s</li>",
        "r_data": "<li>Data {examples} examples · epochs {epochs} · seed {seed}</li>",
    },
    "ko": {
        "cli_description": "작은 언어 모델에 LoRA 로 냥체 말투를 입히고 전/후를 보고서로 확인한다.",
        "cli_epilog": "언어: --lang > LORA_DEMO_LANG > LC_ALL > LC_MESSAGES > LANG (한국어가 아니면 영어).",
        "h_lang": "출력 언어 en|ko (기본: LORA_DEMO_LANG, 다음 LC_ALL/LC_MESSAGES/LANG, 그 밖은 en)",
        "h_mock": "무작위 초소형 모델로 네트워크 없이 전체 흐름만 확인(수 초)",
        "h_device": "기본 auto: CUDA > MPS > CPU",
        "h_rank": "LoRA 순위 r (기본 8)",
        "h_targets": "LoRA 를 붙일 층 (기본 all-linear)",
        "h_quick": "1에폭만 학습(느린 PC용)",
        "h_epochs": "에폭 수(기본 3)",
        "h_lr": "학습률(기본 3e-4)",
        "h_data": "내 학습 데이터 JSONL (messages 형식). 지정하면 말투 적중률은 계산하지 않음",
        "h_runs_dir": "결과 폴더 (기본 ./runs)",
        "h_keep_runs": "runs/ 에 남길 최근 실행 수 (1 이상, 기본 5)",
        "h_max_steps": "이 스텝에서 학습을 멈춤(0이면 에폭 기준)",
        "arg_int": "정수(1 이상)여야 합니다: {value!r}",
        "arg_keep": "1 이상이어야 합니다 (최근 실행 최소 1개는 남깁니다)",
        "arg_min1": "1 이상이어야 합니다 (받은 값: {n})",
        "arg_float": "실수(0보다 큼)여야 합니다: {value!r}",
        "arg_pos_float": "0보다 큰 유한한 값이어야 합니다 (받은 값: {value})",
        "error_prefix": "오류: ",
        "oom": "메모리가 부족합니다. `--device cpu` 로 다시 실행해 보세요.",
        "runs_dir_unwritable": "{path}: 결과 폴더를 만들 수 없습니다 ({error}) — --runs-dir 를 확인하세요",
        "report_usage": "사용법: python -m lora_demo.report <result.json> <출력.html>",
        "report_input_error": "{path}: result.json 을 읽을 수 없습니다 ({error}) — 경로를 확인하세요",
        "report_not_result": "{path}: lora-demo 의 result.json 이 아닙니다 ({error}) — runs/<시각>/ 폴더 안의 result.json 을 넘기세요",
        "report_output_error": "{path}: 보고서를 쓸 수 없습니다 ({error}) — 출력 경로를 확인하세요",
        "sec_unit": "{s:.1f}초",
        "min_sec_unit": "{m}분 {s}초",
        "p_device": "[1/6] 장치 확인: {device} ({dtype})",
        "p_mock_suffix": "  — 모의 모드",
        "p_mock_model": "[2/6] 모의 모델 만들기 (무작위 초소형 모델, 네트워크 없음)",
        "p_real_model": "[2/6] 모델 준비: {model} (처음 한 번 약 1GB 내려받기)",
        "p_data": "       학습 데이터 {rows}문항, 평가 질문 {prompts}개",
        "p_before": "[3/6] 학습 전 답변 {n}개 (예: {q!r} → {a!r})",
        "p_lora": "[4/6] LoRA 붙이기 (r={r}, {targets}): 학습 파라미터 {trainable:,} / {total:,} ({pct:.2f}%)",
        "p_step": "[5/6] 학습 step {step}/{total}  loss {loss:.2f}  (경과 {elapsed})",
        "p_saved": "[6/6] 저장 완료: {path}",
        "p_hits": "       말투 적중 {b}/{n} → {a}/{n}  (프롬프트만: {p}/{n})",
        "p_sizes": "       어댑터 {adapter:.1f}MB (원본 모델 {base:.1f}MB), 학습 {train}, 전체 {total}",
        "p_off": "       어댑터 끄면 학습 전과 같은 답: {same}",
        "where_line": "{path}:{line}번째 줄",
        "file_missing": "{where}: 파일이 없습니다. --data 경로를 확인하세요",
        "read_failed": "{where}: 파일을 읽을 수 없습니다 ({error}) — UTF-8 텍스트 파일인지 확인하세요",
        "not_json": '{where}: JSON 이 아닙니다. 한 줄에 {{"messages":[...]}} 하나씩 쓰세요',
        "no_messages": '{where}: "messages" 목록이 없거나 비어 있습니다',
        "bad_item": '{where}: messages 의 각 항목은 {{"role":...,"content":...}} 여야 합니다',
        "missing_field": '{where}: {index}번째 메시지에 문자열 "role" 과 "content" 가 모두 있어야 합니다',
        "last_role": '{where}: 마지막 메시지의 role 이 "assistant" 여야 합니다 (정답 답변이 필요합니다)',
        "no_user": '{where}: assistant 답변 앞에 "user" 메시지가 하나 이상 있어야 합니다 (질문 없이는 학습할 수 없습니다)',
        "empty_file": "{where}: 학습 데이터가 비어 있습니다",
        "template_mismatch": "프롬프트 토큰열이 전체 토큰열의 접두가 아닙니다 (채팅 템플릿 불일치)",
        "r_title": "LoRA 파인튜닝 보고서",
        "r_sub": "{model} · 장치 {device} ({dtype}) · 실행 {run}",
        "r_mock": "<b>모의 모드</b> — 무작위 초소형 모델이므로 답변 품질은 의미 없습니다. 파이프라인이 도는지만 확인하는 실행입니다.",
        "r_h1": "1. 얼마나 가벼운가",
        "r_params": "파라미터",
        "r_filesize": "파일 크기",
        "r_pct": "전체의 <b>{pct:.2f}%</b>만 학습했습니다 (r={r}, 대상 {targets})",
        "r_adapter_note": "학습 결과(어댑터)만 저장하면 원본 모델의 일부 크기로 충분합니다. 원본 모델 파일은 건드리지 않습니다.",
        "r_h2": "2. 손실 곡선",
        "r_loss_detail": "{first:.2f} → {last:.2f} ({steps} 스텝, 학습 {seconds}초, 마이크로배치 {mb} × 누적 {ga}, 학습률 {lr})",
        "r_axis_steps": "{n} 스텝",
        "r_aria_loss": "손실 곡선",
        "r_h3": "3. 학습 전/후 답변",
        "r_th_question": "질문 (학습에 없던 것)",
        "r_th_before": "학습 전",
        "r_th_prompt": "프롬프트만",
        "r_th_after": "학습 후",
        "r_th_hit": "말투 적중",
        "r_empty": "(빈 답)",
        "r_off_q": "어댑터를 끄면 학습 전과 같은 답이 나오는가: ",
        "r_yes": "예",
        "r_no": "아니오",
        "r_off_note": " (원래 가중치는 그대로이고 어댑터만 켜고 끌 수 있다는 증거)",
        "r_h4": "4. 학습 진행 타임라인",
        "r_h5": "5. rank 계산표 (학습 없이 공식으로 계산)",
        "r_th_qv": "q·v 층만: 파라미터",
        "r_th_file": "파일",
        "r_th_all": "모든 선형층: 파라미터",
        "r_formula": "공식: 층마다 r × (입력 차원 + 출력 차원) 개를 모두 더한다. 파일 크기는 float32(4바이트)로 저장된다고 보고 어림한 값.",
        "r_h6": "6. 한계 — 말투는 배우지만 지식은 늘지 않는다",
        "r_l1": "<li>LoRA 는 <b>말하는 방식</b>(말투·형식)을 빨리 배웁니다. 몇십 스텝이면 충분합니다.</li>",
        "r_l2": "<li>하지만 <b>새로운 사실</b>은 잘 못 외웁니다. 가상 설정(몽글이·구름숲 도서관)을 묻는 질문의 답을 위 표에서 직접 확인해 보세요.</li>",
        "r_l3": "<li>말투가 맞아도 내용이 그럴듯한 오답일 수 있습니다. 말투 적중률은 내용이 맞다는 뜻이 아닙니다.</li>",
        "r_l4": "<li>작은 모델과 100문항이라 답변이 실행마다·설정마다 흔들릴 수 있습니다.</li>",
        "r_h7": "7. 실행 정보",
        "r_time": "<li>시간: 모델 준비 {load}초 · 학습 전 답변 {gb}초 · 학습 {train}초 · 학습 후 답변 {ga}초 · 전체 {total}초</li>",
        "r_data": "<li>데이터 {examples}문항 · 에폭 {epochs} · seed {seed}</li>",
    },
}
