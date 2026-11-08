"""테스트 공통 설정: 네트워크를 막아 모의 모드가 정말 오프라인으로 도는지 보증한다."""

import json
import os
import socket
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # 실제 모드 테스트(LORA_DEMO_REAL=1)에서는 풀어 준다


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    if os.environ.get("LORA_DEMO_REAL") == "1":
        return

    def blocked(*args, **kwargs):
        raise RuntimeError("테스트 중 네트워크 접근 시도")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture(scope="session")
def sample_result() -> dict:
    return json.loads((ROOT / "examples" / "sample-run-m3" / "result.json").read_text(encoding="utf-8"))
