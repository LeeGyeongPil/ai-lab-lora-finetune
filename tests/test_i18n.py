"""English / Korean output: message tables, language order, every error path, help, progress log, report."""

import csv
import json
import os
import re
import string
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from lora_demo import __version__
from lora_demo.cli import config_from_args, main, parse_args, run_pipeline
from lora_demo.data import DataError
from lora_demo.i18n import MESSAGES, resolve_lang, t
from lora_demo.report import render

LANGS = ["en", "ko"]
ROOT = Path(__file__).resolve().parents[1]
GOOD = json.dumps({"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "nya"}]})


def _fields(template: str) -> set[str]:
    return {name for _, name, _, _ in string.Formatter().parse(template) if name}


def test_message_keys_match():
    assert set(MESSAGES["en"]) == set(MESSAGES["ko"])
    for key in MESSAGES["en"]:
        assert _fields(MESSAGES["en"][key]) == _fields(MESSAGES["ko"][key]), key


def test_version_is_consistent():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert __version__ == pyproject["project"]["version"] == "0.2.2"


@pytest.mark.parametrize("cli,env,expected", [
    (None, {}, "en"),
    (None, {"LANG": "C.UTF-8"}, "en"),
    (None, {"LANG": "en_US.UTF-8"}, "en"),
    (None, {"LANG": "ko_KR.UTF-8"}, "ko"),
    (None, {"LC_MESSAGES": "ko_KR.UTF-8", "LANG": "en_US.UTF-8"}, "ko"),
    (None, {"LC_ALL": "en_US.UTF-8", "LC_MESSAGES": "ko_KR.UTF-8", "LANG": "ko_KR.UTF-8"}, "en"),
    (None, {"LORA_DEMO_LANG": "en", "LC_ALL": "ko_KR.UTF-8"}, "en"),
    (None, {"LORA_DEMO_LANG": "ko", "LANG": "en_US.UTF-8"}, "ko"),
    ("en", {"LORA_DEMO_LANG": "ko", "LANG": "ko_KR.UTF-8"}, "en"),
    ("ko", {"LORA_DEMO_LANG": "en"}, "ko"),
    (None, {"LORA_DEMO_LANG": "", "LANG": "ko_KR.UTF-8"}, "ko"),
    (None, {"LANG": "fr_FR.UTF-8"}, "en"),
])
def test_resolve_lang_order(cli, env, expected):
    assert resolve_lang(cli, env) == expected


@pytest.fixture
def clean_env(monkeypatch):
    """Language-neutral environment for tests that set their own language."""
    for var in ("LORA_DEMO_LANG", "LC_ALL", "LC_MESSAGES", "LANG"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def _child_env(**extra) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ("LORA_DEMO_LANG", "LC_ALL", "LC_MESSAGES", "LANG")}
    env.update(extra)
    return env


# ---- argument errors -------------------------------------------------------

ARG_CASES = [
    (["--rank", "0"], {"en": "must be 1 or more (got: 0)", "ko": "1 이상이어야 합니다 (받은 값: 0)"}),
    (["--rank", "x"], {"en": "must be an integer (1 or more)", "ko": "정수(1 이상)여야 합니다"}),
    (["--epochs", "0"], {"en": "must be 1 or more (got: 0)", "ko": "1 이상이어야 합니다 (받은 값: 0)"}),
    (["--lr", "x"], {"en": "must be a real number greater than 0", "ko": "실수(0보다 큼)여야 합니다"}),
    (["--lr", "0"], {"en": "must be a finite value greater than 0", "ko": "0보다 큰 유한한 값이어야 합니다"}),
    (["--keep-runs", "0"], {"en": "at least the latest run is kept", "ko": "최근 실행 최소 1개는 남깁니다"}),
    (["--keep-runs", "x"], {"en": "must be an integer (1 or more)", "ko": "정수(1 이상)여야 합니다"}),
]


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("extra,needles", ARG_CASES)
def test_argument_errors_localized(capsys, clean_env, lang, extra, needles):
    with pytest.raises(SystemExit) as exc:
        parse_args(["--mock", "--lang", lang, *extra])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert needles[lang] in err and "Traceback" not in err
    other = "ko" if lang == "en" else "en"
    assert needles[other] not in err


def test_invalid_lang_value_exit_2(capsys, clean_env):
    with pytest.raises(SystemExit) as exc:
        parse_args(["--lang", "fr"])
    assert exc.value.code == 2 and "Traceback" not in capsys.readouterr().err


# ---- data errors -----------------------------------------------------------

DATA_CASES = {
    "not_json": (GOOD + "\nnot json\n", "d.jsonl:2: not JSON", "d.jsonl:2번째 줄: JSON 이 아닙니다"),
    "no_messages_missing": (GOOD + "\n" + '{"x":1}\n', "d.jsonl:2: ", "d.jsonl:2번째 줄: "),
    "no_messages_empty": (GOOD + "\n" + '{"messages":[]}\n', "list is missing or empty", "목록이 없거나 비어 있습니다"),
    "bad_item": ('{"messages":[1,2]}\n', "each item of messages", "messages 의 각 항목은"),
    "missing_role": ('{"messages":[{"content":"q"},{"role":"assistant","content":"a"}]}\n', "message 1 needs both", "1번째 메시지에"),
    "bad_content": ('{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":5}]}\n', "message 2 needs both", "2번째 메시지에"),
    "last_role": ('{"messages":[{"role":"user","content":"q"}]}\n', "the last message must have role", "마지막 메시지의 role"),
    "no_user": ('{"messages":[{"role":"system","content":"s"},{"role":"assistant","content":"a"}]}\n', 'at least one "user"', '"user" 메시지가 하나 이상'),
    "empty": ("\n", "the training data is empty", "학습 데이터가 비어 있습니다"),
}


def _main_data(tmp_path, capsys, lang, data_path):
    code = main(["--lang", lang, "--mock", "--runs-dir", str(tmp_path / "runs"), "--max-steps", "1", "--data", str(data_path)])
    return code, capsys.readouterr().err


# torch 같은 서드파티가 처음 import 될 때 stderr 에 남기는 glog 형식 로그 줄
# (예: "W1112 04:49:34.836000 99102 torch/.../redirects.py:35] NOTE: ...").
# 어떤 테스트가 프로세스에서 torch 를 처음 부르느냐에 따라 나오거나 안 나오므로, 이 형식의 줄만 세지 않는다.
_THIRD_PARTY_LOG = re.compile(r"^[WIEF]\d{4} \d\d:\d\d:\d\d\.\d+ +\d+ \S+:\d+\] ")


def _our_lines(err: str) -> list[str]:
    """stderr 에서 서드파티 로그 줄을 뺀, 우리 프로그램이 쓴 줄들."""
    return [line for line in err.strip().splitlines() if line.strip() and not _THIRD_PARTY_LOG.match(line)]


def _assert_clean_error(code, err, expected_code=2):
    assert code == expected_code
    assert "Traceback" not in err  # 걸러내기 전 전체 출력에서 검사 — 실제 오류는 숨기지 않는다
    assert len(_our_lines(err)) == 1, err


def test_clean_error_check_ignores_only_third_party_log_lines():
    torch_note = ("W1112 04:49:34.836000 99102 torch/distributed/elastic/multiprocessing/redirects.py:35] "
                  "NOTE: Redirects are currently not supported in MacOs.")
    _assert_clean_error(2, f"{torch_note}\nerror: x: cannot create the results folder (PermissionError)\n")
    assert _our_lines("error: a\nWarning: b\n") == ["error: a", "Warning: b"]  # 다른 줄은 그대로 센다
    with pytest.raises(AssertionError):
        _assert_clean_error(2, f"{torch_note}\nTraceback (most recent call last):\nerror: x\n")
    with pytest.raises(AssertionError):
        _assert_clean_error(2, f"{torch_note}\n")  # 우리 오류 줄이 없으면 실패


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("name", list(DATA_CASES))
def test_data_errors_localized(tmp_path, capsys, clean_env, lang, name):
    content, en_needle, ko_needle = DATA_CASES[name]
    p = tmp_path / "d.jsonl"
    p.write_text(content, encoding="utf-8")
    code, err = _main_data(tmp_path, capsys, lang, p)
    _assert_clean_error(code, err)
    assert err.startswith("error: " if lang == "en" else "오류: ")
    assert (en_needle if lang == "en" else ko_needle) in err
    assert not (tmp_path / "runs").exists()  # stops before the model is prepared


@pytest.mark.parametrize("lang,needle", [("en", "file not found"), ("ko", "파일이 없습니다")])
def test_missing_file_localized(tmp_path, capsys, clean_env, lang, needle):
    code, err = _main_data(tmp_path, capsys, lang, tmp_path / "missing.jsonl")
    _assert_clean_error(code, err)
    assert needle in err


@pytest.mark.parametrize("lang,needle", [("en", "cannot read the file (IsADirectoryError)"), ("ko", "파일을 읽을 수 없습니다 (IsADirectoryError)")])
def test_oserror_directory_localized(tmp_path, capsys, clean_env, lang, needle):
    code, err = _main_data(tmp_path, capsys, lang, tmp_path)
    _assert_clean_error(code, err)
    assert needle in err


@pytest.mark.parametrize("lang,needle", [("en", "cannot read the file (UnicodeDecodeError)"), ("ko", "파일을 읽을 수 없습니다 (UnicodeDecodeError)")])
def test_unicode_error_localized(tmp_path, capsys, clean_env, lang, needle):
    p = tmp_path / "bin.jsonl"
    p.write_bytes(b"\xff\xfe\x00bad\x80")
    code, err = _main_data(tmp_path, capsys, lang, p)
    _assert_clean_error(code, err)
    assert needle in err


@pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0), reason="chmod 000 is not enforced here")
@pytest.mark.parametrize("lang,needle", [("en", "cannot read the file (PermissionError)"), ("ko", "파일을 읽을 수 없습니다 (PermissionError)")])
def test_permission_error_localized(tmp_path, capsys, clean_env, lang, needle):
    p = tmp_path / "locked.jsonl"
    p.write_text(GOOD + "\n", encoding="utf-8")
    p.chmod(0)
    try:
        code, err = _main_data(tmp_path, capsys, lang, p)
    finally:
        p.chmod(0o600)
    _assert_clean_error(code, err)
    assert needle in err


@pytest.mark.parametrize("lang,needle", [("en", "Out of memory"), ("ko", "메모리가 부족합니다")])
def test_out_of_memory_localized(tmp_path, capsys, clean_env, monkeypatch, lang, needle):
    def boom(*_a, **_k):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr("lora_demo.cli.run_pipeline", boom)
    code = main(["--lang", lang, "--mock", "--runs-dir", str(tmp_path / "runs")])
    err = capsys.readouterr().err
    _assert_clean_error(code, err, 1)
    assert needle in err and "--device cpu" in err


def test_data_error_str_is_english_and_carries_code():
    e = DataError("not_json", path="x.jsonl", line=3)
    assert e.code == "not_json" and e.params == {"path": "x.jsonl", "line": 3}
    assert str(e) == 'x.jsonl:3: not JSON. Write one {"messages":[...]} per line'
    assert "3번째 줄" in e.render("ko")


def test_language_from_environment(tmp_path, capsys, clean_env):
    missing = tmp_path / "nope.jsonl"
    clean_env.setenv("LORA_DEMO_LANG", "en")
    clean_env.setenv("LANG", "ko_KR.UTF-8")
    assert main(["--mock", "--data", str(missing)]) == 2
    assert "file not found" in capsys.readouterr().err
    clean_env.delenv("LORA_DEMO_LANG")
    assert main(["--mock", "--data", str(missing)]) == 2
    assert "파일이 없습니다" in capsys.readouterr().err
    clean_env.setenv("LANG", "en_US.UTF-8")
    assert main(["--mock", "--data", str(missing)]) == 2
    assert "file not found" in capsys.readouterr().err


# ---- help ------------------------------------------------------------------

@pytest.mark.parametrize("lang,needles,absent", [
    ("en", ["Give a small language model", "--lang", "LORA_DEMO_LANG", "Learning rate"], "학습률"),
    ("ko", ["작은 언어 모델에", "--lang", "LORA_DEMO_LANG", "학습률"], "Learning rate"),
])
def test_help_language(capsys, clean_env, lang, needles, absent):
    with pytest.raises(SystemExit) as exc:
        parse_args(["--help", "--lang", lang])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert all(n in out for n in needles) and absent not in out
    assert "--lang > LORA_DEMO_LANG > LC_ALL > LC_MESSAGES > LANG" in out


def test_help_follows_env_in_subprocess():
    cmd = [sys.executable, "-m", "lora_demo.cli", "--help"]
    out = subprocess.run(cmd, capture_output=True, text=True, env=_child_env(LORA_DEMO_LANG="en"), timeout=120).stdout
    assert "Learning rate" in out
    out = subprocess.run([*cmd, "--lang", "ko"], capture_output=True, text=True, env=_child_env(LORA_DEMO_LANG="en"), timeout=120).stdout
    assert "학습률" in out


# ---- progress log and artifacts ---------------------------------------------

@pytest.fixture(scope="module")
def logged(tmp_path_factory):
    d = tmp_path_factory.mktemp("i18n-runs")
    out = {}
    for lang in LANGS:
        cfg = config_from_args(parse_args(["--mock", "--lang", lang, "--runs-dir", str(d / lang), "--max-steps", "2"]))
        lines: list[str] = []
        out[lang] = (run_pipeline(cfg, say=lines.append, lang=lang), lines)
    return out


def test_progress_log_localized(logged):
    en = "\n".join(logged["en"][1])
    ko = "\n".join(logged["ko"][1])
    for i in range(1, 7):
        assert f"[{i}/6]" in en and f"[{i}/6]" in ko
    assert "Device check" in en and "mock mode" in en and "Saved:" in en and "Adapter off gives" in en
    assert "장치 확인" in ko and "모의 모드" in ko and "저장 완료" in ko and "어댑터 끄면" in ko
    assert "장치 확인" not in en and "Device check" not in ko


def test_artifacts_do_not_depend_on_language(logged):
    keys, headers, rep = {}, {}, {}
    for lang in LANGS:
        run_dir = logged[lang][0]
        res = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        keys[lang] = set(res)
        with open(run_dir / "train_log.csv", encoding="utf-8") as f:
            headers[lang] = next(csv.reader(f))
        rep[lang] = (run_dir / "report.html").read_text(encoding="utf-8")
    assert keys["en"] == keys["ko"] and "lang" not in keys["en"]
    assert headers["en"] == headers["ko"] == ["step", "loss"]
    assert '<html lang="en">' in rep["en"] and '<html lang="ko">' in rep["ko"]
    assert "LoRA fine-tuning report" in rep["en"] and "LoRA 파인튜닝 보고서" in rep["ko"]


# ---- report ----------------------------------------------------------------

def test_report_english_keeps_korean_data(sample_result):
    html = render(sample_result, "en")
    assert '<html lang="en">' in html
    assert "1. How light is it?" in html and "Loss curve" in html
    assert "<script" not in html and "얼마나 가벼운가" not in html
    assert sample_result["eval"]["prompts"][0] in html  # question text is data, not translated


def test_report_korean(sample_result):
    html = render(sample_result, "ko")
    assert '<html lang="ko">' in html and "1. 얼마나 가벼운가" in html and "손실 곡선" in html
    assert "How light" not in html


@pytest.mark.parametrize("lang,banner", [("en", "Mock mode"), ("ko", "모의 모드")])
def test_mock_banner_by_language(sample_result, lang, banner):
    assert banner not in render(sample_result, lang)
    assert banner in render(dict(sample_result, mode="mock"), lang)


def test_report_default_language_from_env(sample_result, clean_env):
    clean_env.setenv("LANG", "ko_KR.UTF-8")
    assert '<html lang="ko">' in render(sample_result)
    clean_env.setenv("LANG", "en_US.UTF-8")
    assert '<html lang="en">' in render(sample_result)


def test_report_module_cli_uses_env(tmp_path, sample_result):
    src = tmp_path / "result.json"
    src.write_text(json.dumps(sample_result), encoding="utf-8")
    out = tmp_path / "out.html"
    subprocess.run([sys.executable, "-m", "lora_demo.report", str(src), str(out)], check=True, env=_child_env(LORA_DEMO_LANG="en"), timeout=120)
    assert '<html lang="en">' in out.read_text(encoding="utf-8")


def test_t_formats_placeholders():
    assert t("arg_min1", "en", n=3) == "must be 1 or more (got: 3)"
    assert t("arg_min1", "xx", n=3) == "must be 1 or more (got: 3)"  # unknown language -> English


# ---- unwritable results folder and `python -m lora_demo.report` bad input ---

_NO_ROOT = pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0), reason="permission bits are not enforced here")


@pytest.fixture
def no_training(monkeypatch):
    """Fail the test if the pipeline (model download / training) starts: a bad --runs-dir must stop before it."""
    def must_not_run(*_a, **_k):
        raise AssertionError("training started although --runs-dir is unusable")

    monkeypatch.setattr("lora_demo.cli.run_pipeline", must_not_run)


@_NO_ROOT
@pytest.mark.parametrize("lang,needle", [("en", "cannot create the results folder"), ("ko", "결과 폴더를 만들 수 없습니다")])
def test_unwritable_runs_dir_localized(tmp_path, capsys, clean_env, no_training, lang, needle):
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o555)
    try:
        code = main(["--lang", lang, "--mock", "--runs-dir", str(ro / "x")])
    finally:
        ro.chmod(0o700)
    out, err = capsys.readouterr()
    _assert_clean_error(code, err)
    assert needle in err and "PermissionError" in err
    assert "[1/6]" not in out  # stopped before training


@_NO_ROOT
@pytest.mark.parametrize("lang,needle", [("en", "cannot create the results folder"), ("ko", "결과 폴더를 만들 수 없습니다")])
def test_existing_readonly_runs_dir_stops_before_training(tmp_path, capsys, clean_env, no_training, lang, needle):
    ro = tmp_path / "runs"
    ro.mkdir()
    ro.chmod(0o555)  # the folder exists, so mkdir succeeds; only the write probe catches it
    try:
        code = main(["--lang", lang, "--mock", "--runs-dir", str(ro)])
    finally:
        ro.chmod(0o700)
    err = capsys.readouterr().err
    _assert_clean_error(code, err)
    assert needle in err and "PermissionError" in err


@pytest.mark.parametrize("lang,needle", [("en", "cannot create the results folder"), ("ko", "결과 폴더를 만들 수 없습니다")])
def test_runs_dir_is_a_file_localized(tmp_path, capsys, clean_env, no_training, lang, needle):
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    code = main(["--lang", lang, "--mock", "--runs-dir", str(blocker)])
    err = capsys.readouterr().err
    _assert_clean_error(code, err)
    assert needle in err


@pytest.mark.parametrize("lang,needle", [("en", "cannot create the results folder"), ("ko", "결과 폴더를 만들 수 없습니다")])
def test_runs_dir_broken_during_training_still_one_line(tmp_path, capsys, clean_env, monkeypatch, lang, needle):
    """If the folder breaks after the pre-check (e.g. removed while training), saving still fails cleanly."""
    monkeypatch.setattr("lora_demo.cli.check_runs_dir", lambda _p: None)
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    code = main(["--lang", lang, "--mock", "--max-steps", "1", "--runs-dir", str(blocker)])
    err = capsys.readouterr().err
    _assert_clean_error(code, err)  # torch's import-time log line may appear here; only our line is counted
    assert needle in err


def test_runs_dir_check_leaves_no_files(tmp_path):
    from lora_demo.cli import check_runs_dir

    runs = tmp_path / "a" / "runs"
    check_runs_dir(runs)
    assert runs.is_dir() and list(runs.iterdir()) == []  # the write probe is removed again


def _run_report(args, lang):
    return subprocess.run([sys.executable, "-m", "lora_demo.report", *args], capture_output=True, text=True,
                          env=_child_env(LORA_DEMO_LANG=lang), timeout=120)


@pytest.mark.parametrize("lang,needle", [("en", "usage: python -m lora_demo.report"), ("ko", "사용법: python -m lora_demo.report")])
def test_report_module_usage_localized(lang, needle):
    r = _run_report([], lang)
    _assert_clean_error(r.returncode, r.stderr)
    assert needle in r.stderr


@pytest.mark.parametrize("lang,needle", [("en", "cannot read result.json"), ("ko", "result.json 을 읽을 수 없습니다")])
def test_report_module_missing_input_localized(tmp_path, lang, needle):
    r = _run_report([str(tmp_path / "nope.json"), str(tmp_path / "o.html")], lang)
    _assert_clean_error(r.returncode, r.stderr)
    assert needle in r.stderr and not (tmp_path / "o.html").exists()


@pytest.mark.parametrize("lang,needle", [("en", "cannot read result.json"), ("ko", "result.json 을 읽을 수 없습니다")])
def test_report_module_bad_json_localized(tmp_path, lang, needle):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    r = _run_report([str(bad), str(tmp_path / "o.html")], lang)
    _assert_clean_error(r.returncode, r.stderr)
    assert needle in r.stderr


@pytest.mark.parametrize("content", ["{}", "[1]"])
@pytest.mark.parametrize("lang,needle,absent", [
    ("en", "not a result.json from lora-demo", "result.json 이 아닙니다"),
    ("ko", "lora-demo 의 result.json 이 아닙니다", "not a result.json"),
])
def test_report_module_not_a_result_localized(tmp_path, content, lang, needle, absent):
    other = tmp_path / "other.json"
    other.write_text(content, encoding="utf-8")
    r = _run_report([str(other), str(tmp_path / "o.html")], lang)
    _assert_clean_error(r.returncode, r.stderr)
    assert needle in r.stderr and absent not in r.stderr
    assert "check the path" not in r.stderr and "경로를 확인하세요" not in r.stderr  # the file was read; the path is fine
    assert not (tmp_path / "o.html").exists()


@pytest.mark.parametrize("lang,needle", [("en", "cannot write the report"), ("ko", "보고서를 쓸 수 없습니다")])
def test_report_module_unwritable_output_localized(tmp_path, sample_result, lang, needle):
    src = tmp_path / "result.json"
    src.write_text(json.dumps(sample_result), encoding="utf-8")
    r = _run_report([str(src), str(tmp_path / "no-such-dir" / "o.html")], lang)
    _assert_clean_error(r.returncode, r.stderr)
    assert needle in r.stderr
