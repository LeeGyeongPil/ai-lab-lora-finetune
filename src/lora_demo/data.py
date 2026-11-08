"""학습 데이터 읽기, 채팅 템플릿 적용, 프롬프트 가리기(마스킹), 배치 만들기."""

import json
from pathlib import Path

import torch


def load_jsonl(path: Path) -> list[dict]:
    """한 줄에 {"messages":[{role,content},...]} 형식."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def encode(tok, messages: list[dict]) -> tuple[list[int], list[int]]:
    """대화 하나를 (입력 토큰, 정답 라벨)로 바꾼다.

    질문 부분(시스템·사용자·'<|im_start|>assistant\\n')은 라벨을 -100 으로 가려서
    손실이 '답변 토큰'에만 걸리게 한다. 모델은 질문을 외우는 게 아니라 답하는 법을 배운다.
    """
    prompt = tok.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=True, return_dict=True)["input_ids"]
    full = tok.apply_chat_template(messages, tokenize=True, return_dict=True)["input_ids"]
    # 질문 토큰열이 전체의 앞부분과 다르면 템플릿이 어긋난 것 — 조용히 틀리느니 바로 멈춘다
    if full[: len(prompt)] != prompt:
        raise ValueError("프롬프트 토큰열이 전체 토큰열의 접두가 아닙니다 (채팅 템플릿 불일치)")
    labels = [-100] * len(prompt) + full[len(prompt):]
    return full, labels


def collate(batch: list[tuple[list[int], list[int]]], pad_id: int, device) -> dict:
    """길이가 다른 문항을 가장 긴 것에 맞춰 채워(패딩) 텐서로 만든다."""
    longest = max(len(ids) for ids, _ in batch)
    input_ids = [ids + [pad_id] * (longest - len(ids)) for ids, _ in batch]
    labels = [lab + [-100] * (longest - len(lab)) for _, lab in batch]
    attn = [[1] * len(ids) + [0] * (longest - len(ids)) for ids, _ in batch]
    return {
        "input_ids": torch.tensor(input_ids, device=device),
        "labels": torch.tensor(labels, device=device),
        "attention_mask": torch.tensor(attn, device=device),
    }
