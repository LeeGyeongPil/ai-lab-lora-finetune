"""AC-5·6 — 프롬프트 마스킹과 데이터 규칙."""

import json

import pytest

from conftest import ROOT
from lora_demo.cli import config_from_args, parse_args
from lora_demo.data import encode, load_jsonl
from lora_demo.evaluate import style_hit
from lora_demo.model import build_mock


def test_prompt_tokens_masked():
    tok, _ = build_mock(config_from_args(parse_args(["--mock"])))
    msgs = [{"role": "system", "content": "시스템"}, {"role": "user", "content": "질문이다"}, {"role": "assistant", "content": "냥! 답이다냥. 🐾"}]
    ids, labels = encode(tok, msgs)
    prompt = tok.apply_chat_template(msgs[:-1], add_generation_prompt=True, tokenize=True, return_dict=True)["input_ids"]
    n = len(prompt)
    assert tok.decode(prompt).endswith("<|im_start|>assistant\n")
    assert all(x == -100 for x in labels[:n])                     # 시스템·사용자·assistant 머리말은 가림
    assert labels[n:] == ids[n:] and all(x != -100 for x in labels[n:])  # 답변은 학습 대상
    assert tok.convert_tokens_to_ids("<|im_end|>") in labels[n:]  # 답변 끝 토큰도 학습 대상


def test_prefix_mismatch_raises():
    class Broken:
        """질문 템플릿과 전체 템플릿이 어긋나는 가짜 토크나이저."""
        def apply_chat_template(self, msgs, add_generation_prompt=False, tokenize=True, return_dict=True):
            return {"input_ids": [1, 2, 3] if add_generation_prompt else [9, 9, 9, 4]}

    with pytest.raises(ValueError):
        encode(Broken(), [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}])


def test_dataset_rules():
    rows = load_jsonl(ROOT / "data" / "nyang_100.jsonl")
    assert len(rows) == 100
    assert all(style_hit(r["messages"][-1]["content"]) for r in rows)
    questions = {r["messages"][0]["content"] for r in rows}
    evals = json.loads((ROOT / "data" / "eval_prompts.json").read_text(encoding="utf-8"))
    assert len(evals) == 6 and not questions & set(evals)
    lic = (ROOT / "data" / "LICENSE").read_text(encoding="utf-8")
    assert "CC0 1.0" in lic and "CC0-1.0" in lic
