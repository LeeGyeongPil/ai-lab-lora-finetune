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


@pytest.fixture(autouse=True)
def fixed_language(monkeypatch):
    """언어 환경을 고정한다(기본 ko — 기존 한국어 기대 문구 유지). 환경변수는 하위 프로세스에도 전달된다.

    언어 테스트는 monkeypatch 로 이 값을 덮어쓴다. 실행하는 로캘(LANG 등)과 무관하게 결과가 같아야 한다.
    """
    for var in ("LC_ALL", "LC_MESSAGES", "LANG"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LORA_DEMO_LANG", "ko")


@pytest.fixture(scope="session")
def sample_result() -> dict:
    return json.loads((ROOT / "examples" / "sample-run-m3" / "result.json").read_text(encoding="utf-8"))
