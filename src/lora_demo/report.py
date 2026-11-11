"""result.json → 자체 완결 HTML 보고서 (외부 리소스 없음, 인라인 SVG, 라이트/다크).

사용: python -m lora_demo.report result.json out.html
"""

import json
import sys
from html import escape as e

from .i18n import resolve_lang, t

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


def loss_svg(losses: list[float], lang: str = "en") -> str:
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
    xl = f'<text x="{L}" y="{H - 8}">1</text><text x="{W - R}" y="{H - 8}" text-anchor="end">{t("r_axis_steps", lang, n=len(losses))}</text>'
    return f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="{t("r_aria_loss", lang)}">{grid}{xl}<polyline class="ln" points="{pts}"/></svg>'


def cell(text: str, hit: bool | None, lang: str = "en") -> str:
    cls = "" if hit is None else ("hit" if hit else "miss")
    return f'<td class="{cls}">{e(text) or t("r_empty", lang)}</td>'


def render(r: dict, lang: str | None = None) -> str:
    """Render result.json as HTML in ``lang`` ('en'/'ko'; default: LORA_DEMO_LANG, then the locale)."""
    from .evaluate import style_hit

    lang = lang or resolve_lang()
    L = lambda key, **kw: t(key, lang, **kw)  # noqa: E731  (짧은 별칭)
    m, lo, tr, ev = r["model"], r["lora"], r["train"], r["eval"]
    check = ev.get("style_check", True)
    qn = len(ev["prompts"])
    out = [f'<!doctype html><html lang="{lang}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
           f"<title>{L('r_title')}</title><style>{CSS}</style></head><body><main>"]
    out.append(f"<h1>{L('r_title')}</h1>")
    out.append(f'<p class="sub">{L("r_sub", model=e(m["id"]), device=e(r["device"]), dtype=e(r["dtype"]), run=e(r.get("run_name", "")))}</p>')
    if r["mode"] == "mock":
        out.append(f'<div class="band">{L("r_mock")}</div>')

    # 1) 얼마나 가벼운가
    out.append(f'<section id="light"><h2>{L("r_h1")}</h2><div class="card">')
    out.append(bar_row(L("r_params"), lo["trainable_params"], m["total_params"], f'{lo["trainable_params"]:,} / {m["total_params"]:,}'))
    out.append(f'<p class="mut" style="margin:0 0 10px">{L("r_pct", pct=lo["trainable_pct"], r=lo["r"], targets=e(lo["targets"]))}</p>')
    out.append(bar_row(L("r_filesize"), lo["adapter_bytes"], m["size_bytes"], f'{mb(lo["adapter_bytes"])} / {mb(m["size_bytes"])}'))
    out.append(f'<p class="mut" style="margin:0">{L("r_adapter_note")}</p></div></section>')

    # 2) 손실 곡선
    ls = tr["losses"]
    out.append(f'<section id="loss"><h2>{L("r_h2")}</h2><div class="card">{loss_svg(ls, lang)}'
               f'<p class="mut" style="margin:6px 0 0">{L("r_loss_detail", first=ls[0], last=ls[-1], steps=tr["steps"], seconds=tr["seconds"], mb=tr["micro_batch"], ga=tr["grad_accum"], lr=tr["lr"])}</p></div></section>')

    # 3) 전/후 답변
    out.append(f'<section id="compare"><h2>{L("r_h3")}</h2><table><tr><th>{L("r_th_question")}</th><th>{L("r_th_before")}</th><th>{L("r_th_prompt")}</th><th>{L("r_th_after")}</th></tr>')
    for i, q in enumerate(ev["prompts"]):
        h = (lambda k: style_hit(ev[k][i]) if check else None)
        out.append(f'<tr class="qrow"><td>{e(q)}</td>{cell(ev["before"][i], h("before"), lang)}{cell(ev["prompt_baseline"][i], h("prompt_baseline"), lang)}{cell(ev["after"][i], h("after"), lang)}</tr>')
    if check:
        out.append(f'<tr><th>{L("r_th_hit")}</th><td class="big">{ev["style_hit_before"]}/{qn}</td><td class="big">{ev["style_hit_prompt"]}/{qn}</td><td class="big ok">{ev["style_hit_after"]}/{qn}</td></tr>')
    out.append("</table>")
    off = r["adapter_off_equals_before"]
    out.append(f'<p class="mut">{L("r_off_q")}<b class="{"ok" if off else "bad"}">{L("r_yes") if off else L("r_no")}</b>{L("r_off_note")}</p></section>')

    # 4) 타임라인
    out.append(f'<section id="timeline"><h2>{L("r_h4")}</h2><div class="tl">')
    for s in r["snapshots"]:
        out.append(f'<div class="step"><b>step {s["step"]}</b>')
        for q, a in zip(s["questions"], s["answers"]):
            out.append(f'<div><span class="mut">{e(q)}</span><br>{e(a) or L("r_empty")}</div>')
        out.append("</div>")
    out.append("</div></section>")

    # 5) rank 표
    out.append(f'<section id="ranks"><h2>{L("r_h5")}</h2><table><tr><th>r</th><th>{L("r_th_qv")}</th><th>{L("r_th_file")}</th><th>{L("r_th_all")}</th><th>{L("r_th_file")}</th></tr>')
    for row in r["rank_table"]:
        cur = ' style="font-weight:700"' if row["r"] == lo["r"] else ""
        out.append(f'<tr class="rrow"{cur}><td>{row["r"]}</td><td class="n">{row["qv_params"]:,}</td><td class="n">{mb(row["qv_bytes"])}</td><td class="n">{row["all_params"]:,}</td><td class="n">{mb(row["all_bytes"])}</td></tr>')
    out.append(f'</table><p class="mut">{L("r_formula")}</p></section>')

    # 6) 한계
    out.append(f'<section id="limits"><h2>{L("r_h6")}</h2><div class="card"><ul>'
               f'{L("r_l1")}{L("r_l2")}{L("r_l3")}{L("r_l4")}</ul></div></section>')

    # 7) 실행 정보
    tm, v = r["timing"], r["versions"]
    out.append(f'<section id="info"><h2>{L("r_h7")}</h2><div class="card"><ul>'
               + L("r_time", load=tm.get("load_s", 0), gb=tm.get("gen_before_s", 0), train=tm.get("train_s", 0), ga=tm.get("gen_after_s", 0), total=tm.get("total_s", 0))
               + L("r_data", examples=tr["examples"], epochs=tr["epochs"], seed=tr["seed"])
               + f'<li>Python {e(v["python"])} · torch {e(v["torch"])} · transformers {e(v["transformers"])} · peft {e(v["peft"])}</li>'
               "</ul></div></section></main></body></html>")
    return "".join(out)


def main(argv: list[str] | None = None) -> int:
    """CLI entry: one localized line and exit code 2 on a bad argument count or an unreadable input."""
    argv = sys.argv[1:] if argv is None else argv
    lang = resolve_lang()
    if len(argv) != 2:
        print(t("report_usage", lang), file=sys.stderr)
        return 2
    try:
        with open(argv[0], encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, UnicodeDecodeError, ValueError) as err:
        print(t("report_input_error", lang, path=argv[0], error=err.__class__.__name__), file=sys.stderr)
        return 2
    try:
        html = render(data, lang)
    except (KeyError, TypeError, IndexError, AttributeError, ValueError) as err:
        # 파일은 읽혔지만 result.json 모양이 아니다 → 경로가 아니라 파일 종류를 안내한다
        print(t("report_not_result", lang, path=argv[0], error=err.__class__.__name__), file=sys.stderr)
        return 2
    try:
        with open(argv[1], "w", encoding="utf-8") as f:
            f.write(html)
    except OSError as err:
        print(t("report_output_error", lang, path=argv[1], error=err.__class__.__name__), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    # 언어: LORA_DEMO_LANG > LC_ALL > LC_MESSAGES > LANG (인자 형식은 그대로)
    sys.exit(main())
