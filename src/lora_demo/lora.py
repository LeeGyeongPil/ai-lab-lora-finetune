"""LoRA 붙이기와 파라미터 수 계산.

LoRA 의 핵심: 큰 가중치 W (d_out × d_in) 는 얼리고, 작은 두 행렬 B (d_out × r), A (r × d_in) 만 학습한다.
층 하나가 추가로 배우는 숫자는 r·(d_in + d_out) 개 — 원래 W 의 d_in·d_out 개보다 훨씬 적다.
"""

import torch.nn as nn
from peft import LoraConfig, get_peft_model

TARGETS = {"qv": ["q_proj", "v_proj"], "all-linear": "all-linear"}


def attach_lora(model, rank: int, alpha: int, dropout: float, targets: str):
    """원래 가중치를 얼리고 LoRA 어댑터를 붙인다."""
    conf = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=TARGETS[targets], task_type="CAUSAL_LM")
    return get_peft_model(model, conf)


def count_params(model) -> tuple[int, int]:
    """(학습 가능 파라미터, 전체 파라미터). 직접 세므로 라이브러리 버전과 무관하다."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def target_layers(base_model, targets: str) -> list[tuple[str, int, int]]:
    """LoRA 가 붙을 선형층 목록 [(이름, d_in, d_out)]. all-linear 는 출력층(lm_head) 제외."""
    names = ("q_proj", "v_proj") if targets == "qv" else None
    out = []
    for name, mod in base_model.named_modules():
        if not isinstance(mod, nn.Linear) or name.endswith("lm_head"):
            continue
        if names is None or name.split(".")[-1] in names:
            out.append((name, mod.in_features, mod.out_features))
    return out


def lora_param_formula(base_model, rank: int, targets: str) -> int:
    """공식 Σ r·(d_in + d_out). 학습 없이 계산만으로 어댑터 크기를 안다."""
    return sum(rank * (d_in + d_out) for _, d_in, d_out in target_layers(base_model, targets))


def rank_table(base_model, ranks=(1, 2, 4, 8, 16, 32, 64)) -> list[dict]:
    """rank × 대상층 계산표. 어댑터 파일은 float32(4바이트)로 저장되므로 파라미터 × 4 로 어림한다."""
    rows = []
    for r in ranks:
        qv = lora_param_formula(base_model, r, "qv")
        al = lora_param_formula(base_model, r, "all-linear")
        rows.append({"r": r, "qv_params": qv, "qv_bytes": qv * 4, "all_params": al, "all_bytes": al * 4})
    return rows
