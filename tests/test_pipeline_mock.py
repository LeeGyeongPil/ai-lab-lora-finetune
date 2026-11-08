"""AC-1·2·3 — 모의 모드 전체 파이프라인 (네트워크 없음, CPU)."""

import json
import time
from pathlib import Path

import pytest

from lora_demo.cli import config_from_args, parse_args, run_pipeline


def run_mock(tmp_path, *extra):
    cfg = config_from_args(parse_args(["--mock", "--runs-dir", str(tmp_path / "runs"), *extra]))
    return run_pipeline(cfg, say=lambda *_: None)


def load(run_dir):
    return json.loads((run_dir / "result.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory):
    d = tmp_path_factory.mktemp("det")
    return run_mock(d), run_mock(d)


def test_mock_run_creates_artifacts(tmp_path):
    t0 = time.time()
    run_dir = run_mock(tmp_path)
    assert time.time() - t0 < 120  # AC-1: 테스트 1건 120초 이내
    for name in ["result.json", "report.html", "train_log.csv", "adapter/adapter_model.safetensors", "adapter/adapter_config.json"]:
        assert (run_dir / name).is_file(), name
    res = load(run_dir)
    dumped = json.dumps(res)
    # 절대 경로(홈 폴더·임시 폴더)가 결과 파일에 새지 않는다 — OS와 무관하게 확인
    assert res["mode"] == "mock"
    assert str(Path.home()) not in dumped and str(tmp_path) not in dumped


def test_runs_retention_keeps_5(tmp_path):
    for _ in range(7):
        run_mock(tmp_path, "--max-steps", "2")
    assert len(list((tmp_path / "runs").iterdir())) == 5


def test_mock_is_deterministic(two_runs):
    a, b = (load(d) for d in two_runs)
    assert a["train"]["losses"] == b["train"]["losses"]
    assert a["eval"]["before"] == b["eval"]["before"]
    assert a["eval"]["after"] == b["eval"]["after"]


def test_adapter_effect_and_off_switch(two_runs):
    res = load(two_runs[0])
    losses = res["train"]["losses"]
    assert losses[-1] < losses[0]
    assert any(x != y for x, y in zip(res["eval"]["before"], res["eval"]["after"]))
    assert res["adapter_off_equals_before"] is True
