"""AC-4 — LoRA 파라미터 계산이 PEFT 와 같고, 실제 Qwen2.5-0.5B 구성에서 기대값과 일치한다."""

import json

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from lora_demo.cli import config_from_args, parse_args
from lora_demo.lora import attach_lora, lora_param_formula, rank_table
from lora_demo.model import build_mock


def qwen_05b_meta():
    """다운로드 없이 meta 장치(메모리를 쓰지 않는 빈 껍데기)에 0.5B 구조만 만든다."""
    conf = Qwen2Config(vocab_size=151936, hidden_size=896, intermediate_size=4864, num_hidden_layers=24,
                       num_attention_heads=14, num_key_value_heads=2, tie_word_embeddings=True)
    with torch.device("meta"):
        return Qwen2ForCausalLM(conf)


def test_formula_matches_peft_on_mock_model():
    cfg = config_from_args(parse_args(["--mock"]))
    for targets in ("all-linear", "qv"):
        _, model = build_mock(cfg)
        expected = lora_param_formula(model, 8, targets)
        peft_model = attach_lora(model, 8, 16, 0.0, targets)
        assert peft_model.get_nb_trainable_parameters()[0] == expected


def test_qwen_05b_known_counts():
    base = qwen_05b_meta()
    assert sum(p.numel() for p in base.parameters()) == 494_032_768
    assert lora_param_formula(base, 8, "all-linear") == 4_399_104
    assert lora_param_formula(base, 4, "qv") == 270_336


def test_rank_table_is_proportional_to_r():
    rows = rank_table(qwen_05b_meta())
    assert [r["r"] for r in rows] == [1, 2, 4, 8, 16, 32, 64]
    for r in rows:
        assert r["qv_params"] == r["r"] * rows[0]["qv_params"]
        assert r["all_params"] == r["r"] * rows[0]["all_params"]
