"""AC-8·9 — 실제 모델로 돌리는 확인. 내려받기가 필요해 기본은 건너뛰고, LORA_DEMO_REAL=1 일 때만 실행한다."""

import json
import os
import subprocess
import sys
import time

import pytest

pytestmark = [pytest.mark.real, pytest.mark.skipif(os.environ.get("LORA_DEMO_REAL") != "1", reason="LORA_DEMO_REAL=1 일 때만")]


def run_cli(tmp_path, *args):
    env = {k: v for k, v in os.environ.items() if k != "HF_HUB_OFFLINE"}
    t0 = time.time()
    p = subprocess.run([sys.executable, "-m", "lora_demo.cli", "--runs-dir", str(tmp_path), *args], env=env, capture_output=True, text=True)
    elapsed = time.time() - t0
    assert p.returncode == 0, p.stderr[-2000:]
    run = sorted(tmp_path.iterdir())[-1]
    return json.loads((run / "result.json").read_text(encoding="utf-8")), elapsed


def test_real_default_run(tmp_path):  # AC-8
    res, sec = run_cli(tmp_path)
    ev = res["eval"]
    assert sec <= 600 and ev["style_hit_before"] <= 1 and ev["style_hit_after"] >= 5
    assert res["lora"]["adapter_bytes"] <= 20_000_000 and res["lora"]["trainable_pct"] < 1


def test_real_cpu_quick(tmp_path):  # AC-9
    res, sec = run_cli(tmp_path, "--device", "cpu", "--quick")
    assert sec <= 300 and res["eval"]["style_hit_after"] >= 4
