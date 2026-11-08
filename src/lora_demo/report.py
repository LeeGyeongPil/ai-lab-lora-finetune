"""result.json → 자체 완결 HTML 보고서 (외부 리소스 없음, 인라인 SVG, 라이트/다크).

사용: python -m lora_demo.report result.json out.html
"""

import json
import sys
from html import escape as e

CSS = """
:root{--bg:#fafaf7;--fg:#1e1e1c;--mut:#6b6b66;--card:#fff;--line:#e3e2dc;--acc:#2f6fed;--ok:#1f8a4c;--bad:#c0392b;--bar:#d8d7d0;--warn:#fff4d6}
@media (prefers-color-scheme:dark){:root{--bg:#161614;--fg:#ecebe6;--mut:#9a9992;--card:#202020;--line:#34332f;--acc:#6ea0ff;--ok:#4cc17f;--bad:#ef6c5d;--bar:#4a4944;--warn:#3a3320}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 -apple-system,"Segoe UI","Apple SD Gothic Neo","Malgun Gothic",sans-serif}
main{max-width:1000px;padding:24px 20px 60px}h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:34px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--line)}
.sub{color:var(--mut);margin:0 0 14px}.band{background:var(--warn);border:1px solid var(--line);border-radius:8px;padding:8px 12px;margin:12px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:10px 0}
.row{display:flex;align-items:center;gap:10px;margin:6px 0}.row .lab{width:120px;color:var(--mut);flex:none}.row .num{width:190px;flex:none;text-align:right;font-variant-numeric:tabular-nums}
.track{flex:1;background:var(--bar);height:18px;border-radius:4px;overflow:hidden}.fill{height:100%;background:var(--acc);min-width:3px}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);font-size:14px}th,td{padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top;text-align:left}th{color:var(--mut);font-weight:600}
td.hit{border-left:3px solid var(--ok)}td.miss{border-left:3px solid var(--bad)}td.n{text-align:right;font-variant-numeric:tabular-nums}
.big{font-size:20px;font-weight:700}.ok{color:var(--ok)}.bad{color:var(--bad)}.mut{color:var(--mut)}
svg text{fill:var(--mut);font-size:11px}svg .ln{stroke:var(--acc);fill:none;stroke-width:2}svg .gr{stroke:var(--line)}
.tl{display:grid;gap:10px}.tl .step{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 14px}.tl b{color:var(--acc)}
ul{margin:6px 0;padding-left:20px}code{background:var(--bar);padding:1px 5px;border-radius:4px}
"""


def mb(n: int) -> str:
    return f"{n / 1e6:.1f}MB"


def bar_row(label: str, value: int, total: int, text: str) -> str:
    w = max(0.4, 100 * value / total)
    return (f'<div class="row"><span class="lab">{e(label)}</span><div class="track"><div class="fill" style="width:{w:.2f}%"></div></div>'
            f'<span class="num">{e(text)}</span></div>')


def loss_svg(losses: list[float]) -> str:
    """손실 곡선: 가로 = 학습 스텝, 세로 = 손실. 폭 900 × 높이 260 의 인라인 SVG."""
    W, H, L, R, T, B = 900, 260, 46, 12, 12, 30
    lo, hi = min(losses), max(losses)
    span = (hi - lo) or 1.0
    n = max(len(losses) - 1, 1)
    pts = " ".join(f"{L + (W - L - R) * i / n:.1f},{T + (H - T - B) * (1 - (v - lo) / span):.1f}" for i, v in enumerate(losses))
    grid = ""
    for k in range(5):  # 세로축 눈금 5개
        v = lo + span * k / 4
        y = T + (H - T - B) * (1 - k / 4)
        grid += f'<line class="gr" x1="{L}" x2="{W - R}" y1="{y:.1f}" y2="{y:.1f}"/><text x="{L - 6}" y="{y + 4:.1f}" text-anchor="end">{v:.2f}</text>'
    xl = f'<text x="{L}" y="{H - 8}">1</text><text x="{W - R}" y="{H - 8}" text-anchor="end">{len(losses)} 스텝</text>'
    return f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="손실 곡선">{grid}{xl}<polyline class="ln" points="{pts}"/></svg>'


def cell(text: str, hit: bool | None) -> str:
    cls = "" if hit is None else ("hit" if hit else "miss")
    return f'<td class="{cls}">{e(text) or "(빈 답)"}</td>'


def render(r: dict) -> str:
    from .evaluate import style_hit

    m, lo, tr, ev = r["model"], r["lora"], r["train"], r["eval"]
    check = ev.get("style_check", True)
    qn = len(ev["prompts"])
    out = [f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
           f"<title>LoRA 파인튜닝 보고서</title><style>{CSS}</style></head><body><main>"]
    out.append("<h1>LoRA 파인튜닝 보고서</h1>")
    out.append(f'<p class="sub">{e(m["id"])} · 장치 {e(r["device"])} ({e(r["dtype"])}) · 실행 {e(r.get("run_name", ""))}</p>')
    if r["mode"] == "mock":
        out.append('<div class="band"><b>모의 모드</b> — 무작위 초소형 모델이므로 답변 품질은 의미 없습니다. 파이프라인이 도는지만 확인하는 실행입니다.</div>')

    # 1) 얼마나 가벼운가
    out.append('<section id="light"><h2>1. 얼마나 가벼운가</h2><div class="card">')
    out.append(bar_row("파라미터", lo["trainable_params"], m["total_params"], f'{lo["trainable_params"]:,} / {m["total_params"]:,}'))
    out.append(f'<p class="mut" style="margin:0 0 10px">전체의 <b>{lo["trainable_pct"]:.2f}%</b>만 학습했습니다 (r={lo["r"]}, 대상 {e(lo["targets"])})</p>')
    out.append(bar_row("파일 크기", lo["adapter_bytes"], m["size_bytes"], f'{mb(lo["adapter_bytes"])} / {mb(m["size_bytes"])}'))
    out.append('<p class="mut" style="margin:0">학습 결과(어댑터)만 저장하면 원본 모델의 일부 크기로 충분합니다. 원본 모델 파일은 건드리지 않습니다.</p></div></section>')

    # 2) 손실 곡선
    ls = tr["losses"]
    out.append(f'<section id="loss"><h2>2. 손실 곡선</h2><div class="card">{loss_svg(ls)}'
               f'<p class="mut" style="margin:6px 0 0">{ls[0]:.2f} → {ls[-1]:.2f} ({tr["steps"]} 스텝, 학습 {tr["seconds"]}초, '
               f'마이크로배치 {tr["micro_batch"]} × 누적 {tr["grad_accum"]}, 학습률 {tr["lr"]})</p></div></section>')

    # 3) 전/후 답변
    out.append('<section id="compare"><h2>3. 학습 전/후 답변</h2><table><tr><th>질문 (학습에 없던 것)</th><th>학습 전</th><th>프롬프트만</th><th>학습 후</th></tr>')
    for i, q in enumerate(ev["prompts"]):
        h = (lambda k: style_hit(ev[k][i]) if check else None)
        out.append(f'<tr class="qrow"><td>{e(q)}</td>{cell(ev["before"][i], h("before"))}{cell(ev["prompt_baseline"][i], h("prompt_baseline"))}{cell(ev["after"][i], h("after"))}</tr>')
    if check:
        out.append(f'<tr><th>말투 적중</th><td class="big">{ev["style_hit_before"]}/{qn}</td><td class="big">{ev["style_hit_prompt"]}/{qn}</td><td class="big ok">{ev["style_hit_after"]}/{qn}</td></tr>')
    out.append("</table>")
    off = r["adapter_off_equals_before"]
    out.append(f'<p class="mut">어댑터를 끄면 학습 전과 같은 답이 나오는가: <b class="{"ok" if off else "bad"}">{"예" if off else "아니오"}</b> '
               "(원래 가중치는 그대로이고 어댑터만 켜고 끌 수 있다는 증거)</p></section>")

    # 4) 타임라인
    out.append('<section id="timeline"><h2>4. 학습 진행 타임라인</h2><div class="tl">')
    for s in r["snapshots"]:
        out.append(f'<div class="step"><b>step {s["step"]}</b>')
        for q, a in zip(s["questions"], s["answers"]):
            out.append(f'<div><span class="mut">{e(q)}</span><br>{e(a) or "(빈 답)"}</div>')
        out.append("</div>")
    out.append("</div></section>")

    # 5) rank 표
    out.append('<section id="ranks"><h2>5. rank 계산표 (학습 없이 공식으로 계산)</h2><table><tr><th>r</th><th>q·v 층만: 파라미터</th><th>파일</th><th>모든 선형층: 파라미터</th><th>파일</th></tr>')
    for row in r["rank_table"]:
        cur = ' style="font-weight:700"' if row["r"] == lo["r"] else ""
        out.append(f'<tr class="rrow"{cur}><td>{row["r"]}</td><td class="n">{row["qv_params"]:,}</td><td class="n">{mb(row["qv_bytes"])}</td><td class="n">{row["all_params"]:,}</td><td class="n">{mb(row["all_bytes"])}</td></tr>')
    out.append('</table><p class="mut">공식: 층마다 r × (입력 차원 + 출력 차원) 개를 모두 더한다. 파일 크기는 float32(4바이트)로 저장된다고 보고 어림한 값.</p></section>')

    # 6) 한계
    out.append('<section id="limits"><h2>6. 한계 — 말투는 배우지만 지식은 늘지 않는다</h2><div class="card"><ul>'
               "<li>LoRA 는 <b>말하는 방식</b>(말투·형식)을 빨리 배웁니다. 몇십 스텝이면 충분합니다.</li>"
               "<li>하지만 <b>새로운 사실</b>은 잘 못 외웁니다. 가상 설정(몽글이·구름숲 도서관)을 묻는 질문의 답을 위 표에서 직접 확인해 보세요.</li>"
               "<li>말투가 맞아도 내용이 그럴듯한 오답일 수 있습니다. 말투 적중률은 내용이 맞다는 뜻이 아닙니다.</li>"
               "<li>작은 모델과 100문항이라 답변이 실행마다·설정마다 흔들릴 수 있습니다.</li></ul></div></section>")

    # 7) 실행 정보
    t, v = r["timing"], r["versions"]
    out.append('<section id="info"><h2>7. 실행 정보</h2><div class="card"><ul>'
               f'<li>시간: 모델 준비 {t.get("load_s", 0)}초 · 학습 전 답변 {t.get("gen_before_s", 0)}초 · 학습 {t.get("train_s", 0)}초 · 학습 후 답변 {t.get("gen_after_s", 0)}초 · 전체 {t.get("total_s", 0)}초</li>'
               f'<li>데이터 {tr["examples"]}문항 · 에폭 {tr["epochs"]} · seed {tr["seed"]}</li>'
               f'<li>Python {e(v["python"])} · torch {e(v["torch"])} · transformers {e(v["transformers"])} · peft {e(v["peft"])}</li>'
               "</ul></div></section></main></body></html>")
    return "".join(out)


if __name__ == "__main__":
    data = json.loads(open(sys.argv[1], encoding="utf-8").read())
    open(sys.argv[2], "w", encoding="utf-8").write(render(data))
