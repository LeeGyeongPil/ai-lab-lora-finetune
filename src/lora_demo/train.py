"""손으로 쓴 학습 루프 — Trainer 없이 한눈에 읽히도록."""

import random
import time

import torch

from .data import collate


def train(model, enc, pad_id, device, cfg, on_step=None):
    """enc: [(입력, 라벨)] 목록. 마이크로배치 grad_accum 번을 모아 옵티마이저 1스텝.

    on_step(step, losses): 스텝이 끝날 때마다 불리는 콜백(중간 스냅샷용).
    반환: (스텝별 손실 목록, 걸린 초)
    """
    rng = random.Random(cfg.seed)  # 문항 섞는 순서도 seed 로 고정
    params = [p for p in model.parameters() if p.requires_grad]  # LoRA 의 A·B 만
    opt = torch.optim.AdamW(params, lr=cfg.lr)
    model.train()

    losses: list[float] = []
    micro_per_step = cfg.grad_accum
    t0 = time.time()
    done = False
    for _epoch in range(cfg.epochs):
        order = list(range(len(enc)))
        rng.shuffle(order)
        micro = [order[i:i + cfg.micro_batch] for i in range(0, len(order), cfg.micro_batch)]
        for s in range(0, len(micro), micro_per_step):
            group = micro[s:s + micro_per_step]
            step_loss = 0.0
            for idx in group:
                batch = collate([enc[i] for i in idx], pad_id, device)
                loss = model(**batch).loss / len(group)  # 평균을 맞추려고 나눈다
                loss.backward()                           # 기울기를 쌓는다(누적)
                step_loss += loss.item()
            opt.step()                                    # 모아 둔 기울기로 한 번 갱신
            opt.zero_grad()
            losses.append(round(step_loss, 4))
            if on_step:
                on_step(len(losses), losses)
            if cfg.max_steps and len(losses) >= cfg.max_steps:
                done = True
                break
        if done:
            break
    if device.type == "mps":
        torch.mps.synchronize()
    # 옵티마이저 상태(모델 크기만큼의 추가 메모리)를 바로 놓는다
    del opt
    return losses, time.time() - t0
