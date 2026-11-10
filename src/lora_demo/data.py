"""학습 데이터 읽기, 채팅 템플릿 적용, 프롬프트 가리기(마스킹), 배치 만들기."""

import json
from pathlib import Path

import torch


def load_jsonl(path: Path) -> list[dict]:
    """한 줄에 {"messages":[{role,content},...]} 형식."""
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
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


class DataError(ValueError):
    """학습 데이터 형식 오류. 메시지는 '파일:N번째 줄: …' 한 줄이라 그대로 보여 주면 된다."""


def validate_jsonl(path: Path) -> int:
    """모델을 준비하기 전에 JSONL 을 훑어 형식 오류를 줄 번호와 함께 알린다. 정상이면 문항 수를 돌려준다."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8-sig")  # BOM 붙은 UTF-8 도 허용
    except FileNotFoundError:
        raise DataError(f"{path}: 파일이 없습니다. --data 경로를 확인하세요") from None
    except (OSError, UnicodeDecodeError) as e:
        raise DataError(f"{path}: 파일을 읽을 수 없습니다 ({type(e).__name__}) — UTF-8 텍스트 파일인지 확인하세요") from None
    count = 0
    for n, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{path}:{n}번째 줄"
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            raise DataError(f'{where}: JSON 이 아닙니다. 한 줄에 {{"messages":[...]}} 하나씩 쓰세요') from None
        msgs = row.get("messages") if isinstance(row, dict) else None
        if not isinstance(msgs, list) or not msgs:
            raise DataError(f'{where}: "messages" 목록이 없거나 비어 있습니다')
        if not all(isinstance(m, dict) for m in msgs):
            raise DataError(f'{where}: messages 의 각 항목은 {{"role":...,"content":...}} 여야 합니다')
        for i, m in enumerate(msgs, start=1):
            if not isinstance(m.get("role"), str) or not isinstance(m.get("content"), str):
                raise DataError(f'{where}: {i}번째 메시지에 문자열 "role" 과 "content" 가 모두 있어야 합니다')
        if msgs[-1].get("role") != "assistant":
            raise DataError(f'{where}: 마지막 메시지의 role 이 "assistant" 여야 합니다 (정답 답변이 필요합니다)')
        if not any(m["role"] == "user" for m in msgs[:-1]):
            raise DataError(f'{where}: assistant 답변 앞에 "user" 메시지가 하나 이상 있어야 합니다 (질문 없이는 학습할 수 없습니다)')
        count += 1
    if count == 0:
        raise DataError(f"{path}: 학습 데이터가 비어 있습니다")
    return count
