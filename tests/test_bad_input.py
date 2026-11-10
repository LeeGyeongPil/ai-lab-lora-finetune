"""잘못된 입력 — 종료 코드 2 와 한 줄 안내 (모의 모드, 네트워크 없음)."""

import json

import pytest

from lora_demo.cli import config_from_args, main, parse_args
from lora_demo.data import DataError, validate_jsonl

GOOD = json.dumps({"messages": [{"role": "user", "content": "안녕?"}, {"role": "assistant", "content": "안녕냥"}]}, ensure_ascii=False)


@pytest.mark.parametrize("flag,value", [("--rank", "0"), ("--epochs", "0"), ("--lr", "0"), ("--lr", "-1"), ("--rank", "x")])
def test_bad_numbers_exit_2(capsys, flag, value):
    with pytest.raises(SystemExit) as e:
        parse_args(["--mock", flag, value])
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert flag in err and "Traceback" not in err


def test_epochs_and_lr_are_not_silently_replaced():
    cfg = config_from_args(parse_args(["--mock", "--epochs", "2", "--lr", "0.01"]))
    assert cfg.epochs == 2 and cfg.lr == 0.01


def _run(tmp_path, capsys, content):
    p = tmp_path / "d.jsonl"
    if content is not None:
        p.write_text(content, encoding="utf-8")
    code = main(["--mock", "--runs-dir", str(tmp_path / "runs"), "--max-steps", "1", "--data", str(p)])
    return code, capsys.readouterr().err


@pytest.mark.parametrize("content,needle", [
    (None, "파일이 없습니다"),
    (GOOD + "\nnot json\n", "d.jsonl:2번째 줄"),
    ('{"messages":[{"role":"user","content":"안녕?"}]}\n', "d.jsonl:1번째 줄"),
    (GOOD + "\n" + '{"messages":[]}\n', "d.jsonl:2번째 줄"),
    (GOOD + "\n\n" + '{"x":1}\n', "d.jsonl:3번째 줄"),
    (GOOD + "\n" + '{"messages":[{"role":"user"},{"role":"assistant","content":"a"}]}\n', "d.jsonl:2번째 줄"),
    (GOOD + "\n" + '{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":5}]}\n', "d.jsonl:2번째 줄"),
    (GOOD + "\n" + '{"messages":[{"content":"q"},{"role":"assistant","content":"a"}]}\n', "d.jsonl:2번째 줄"),
    ('{"messages":[{"role":"assistant","content":"a"}]}\n', "d.jsonl:1번째 줄"),
    ('{"messages":[{"role":"system","content":"s"},{"role":"assistant","content":"a"}]}\n', "d.jsonl:1번째 줄"),
])
def test_bad_data_exit_2_with_line_number(tmp_path, capsys, content, needle):
    code, err = _run(tmp_path, capsys, content)
    assert code == 2
    assert needle in err and "Traceback" not in err and len(err.strip().splitlines()) == 1
    assert not (tmp_path / "runs").exists()  # 모델 준비·학습 전에 멈춤


def test_empty_file_rejected(tmp_path):
    p = tmp_path / "e.jsonl"
    p.write_text("\n", encoding="utf-8")
    with pytest.raises(DataError):
        validate_jsonl(p)


def test_good_data_passes_and_default_runs(tmp_path, capsys):
    p = tmp_path / "ok.jsonl"
    p.write_text(GOOD + "\n", encoding="utf-8")
    assert validate_jsonl(p) == 1
    assert main(["--mock", "--runs-dir", str(tmp_path / "runs"), "--max-steps", "1"]) == 0


def test_bom_utf8_accepted(tmp_path):
    p = tmp_path / "bom.jsonl"
    p.write_bytes(b"\xef\xbb\xbf" + GOOD.encode("utf-8") + b"\n")
    assert validate_jsonl(p) == 1
    # 모의 모드 전체 실행(build_mock 이 데이터를 다시 읽는 경로)도 BOM 파일로 끝까지 돌아야 한다
    code = main(["--mock", "--runs-dir", str(tmp_path / "runs"), "--max-steps", "1", "--data", str(p)])
    assert code == 0
