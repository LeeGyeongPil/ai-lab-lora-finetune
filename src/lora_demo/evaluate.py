"""생성, 말투 적중 판정, '프롬프트만' 기준선, 어댑터 on/off 확인."""

import re

import torch

from .config import PROMPT_SHOTS, PROMPT_SYSTEM

# 냥체 규칙: "냥! "으로 시작해서 "…냥. 🐾"으로 끝난다
STYLE = re.compile(r"^냥!.*냥\.?\s*🐾\s*$", re.S)


def style_hit(text: str) -> bool:
    return bool(STYLE.match(text.strip()))


def generate(model, tok, prompts: list[str], device, max_new_tokens: int, prefix: list[dict] | None = None) -> list[str]:
    """질문마다 greedy(항상 같은 답) 생성. prefix 가 있으면 앞에 붙인다(시스템 프롬프트 등)."""
    model.eval()
    answers = []
    for q in prompts:
        msgs = (prefix or []) + [{"role": "user", "content": q}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        answers.append(tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip())
    return answers


def prompt_baseline(model, tok, prompts, device, max_new_tokens) -> list[str]:
    """학습 없이 시스템 프롬프트 + 예시 2개만으로 말투를 낼 수 있는지 (원래 모델 기준)."""
    prefix = [{"role": "system", "content": PROMPT_SYSTEM}] + PROMPT_SHOTS
    return generate(model, tok, prompts, device, max_new_tokens, prefix=prefix)


def adapter_off_check(peft_model, tok, prompts, device, max_new_tokens, before: list[str]) -> bool:
    """어댑터를 끈 상태의 답변이 학습 전 답변과 같은가 = 원래 모델은 그대로라는 증거."""
    with peft_model.disable_adapter():
        off = generate(peft_model, tok, prompts, device, max_new_tokens)
    return off == before
