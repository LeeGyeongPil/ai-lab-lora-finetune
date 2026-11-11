"""학습 데이터 읽기, 채팅 템플릿 적용, 프롬프트 가리기(마스킹), 배치 만들기."""

import json
from pathlib import Path

import torch

from .i18n import t


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
        raise DataError("template_mismatch")
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
    """Training-data error carrying a message ``code`` and ``params`` so the CLI can render it in any language.

    ``str(e)`` is the English rendering (for logs and debugging); ``e.render(lang)`` gives the localized one.
    """

    def __init__(self, code: str, **params):
        self.code = code
        self.params = params
        super().__init__(self.render("en"))

    def render(self, lang: str) -> str:
        p = dict(self.params)
        if "path" in p:  # 'file' or 'file:N' (the wording of the line marker is localized)
            p["where"] = t("where_line", lang, path=p["path"], line=p["line"]) if "line" in p else str(p["path"])
        return t(self.code, lang, **p)


def validate_jsonl(path: Path) -> int:
    """Scan a JSONL file before the model is prepared; raise DataError with the line number on a format error.

    Returns the number of examples when the file is valid.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8-sig")  # BOM 붙은 UTF-8 도 허용
    except FileNotFoundError:
        raise DataError("file_missing", path=str(path)) from None
    except (OSError, UnicodeDecodeError) as e:
        raise DataError("read_failed", path=str(path), error=type(e).__name__) from None
    count = 0
    for n, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        loc = {"path": str(path), "line": n}
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            raise DataError("not_json", **loc) from None
        msgs = row.get("messages") if isinstance(row, dict) else None
        if not isinstance(msgs, list) or not msgs:
            raise DataError("no_messages", **loc)
        if not all(isinstance(m, dict) for m in msgs):
            raise DataError("bad_item", **loc)
        for i, m in enumerate(msgs, start=1):
            if not isinstance(m.get("role"), str) or not isinstance(m.get("content"), str):
                raise DataError("missing_field", index=i, **loc)
        if msgs[-1].get("role") != "assistant":
            raise DataError("last_role", **loc)
        if not any(m["role"] == "user" for m in msgs[:-1]):
            raise DataError("no_user", **loc)
        count += 1
    if count == 0:
        raise DataError("empty_file", path=str(path))
    return count
