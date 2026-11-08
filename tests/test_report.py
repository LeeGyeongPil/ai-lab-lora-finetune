"""AC-7 — 실측 기록(sample-run-m3/result.json)으로 보고서를 렌더링해 구성요소를 확인한다."""

import re

from lora_demo.report import render


def test_report_has_all_sections(sample_result):
    html = render(sample_result)
    n_q = len(sample_result["eval"]["prompts"])
    assert "<svg" in html and "<polyline" in html                       # 손실 곡선
    assert html.count('class="qrow"') == n_q                            # 질문별 전/후 행
    assert "4,399,104" in html and "0.89%" in html                      # 학습 파라미터와 비율
    assert "MB" in html and f'{sample_result["model"]["total_params"]:,}' in html  # 어댑터·모델 크기
    assert html.count('class="rrow"') == 7                              # rank 계산표
    assert html.count('<div class="step">') == len(sample_result["snapshots"]) >= 3  # 타임라인


def test_report_is_self_contained(sample_result):
    html = render(sample_result)
    assert not re.search(r'(src|href)\s*=\s*["\']?(https?:)?//', html)
    assert "<script" not in html and "@import" not in html and "url(" not in html


def test_report_escapes_text(sample_result):
    r = dict(sample_result, eval=dict(sample_result["eval"], after=["<script>x</script>"] * len(sample_result["eval"]["after"])))
    assert "<script>x" not in render(r)


def test_mock_banner(sample_result):
    assert "모의 모드" not in render(sample_result)
    assert "모의 모드" in render(dict(sample_result, mode="mock"))
