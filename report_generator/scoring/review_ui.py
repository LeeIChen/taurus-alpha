"""Human review pages: score runs, correct assumptions, record outcomes, compare two runs.

Plain server-rendered HTML; forms post JSON with fetch. Local tool, no authentication.
"""

from __future__ import annotations

import html
import json
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field

from report_generator.run_log import RUNS_DIR
from report_generator.scoring.reviews import SECTIONS, add_preference, add_review, export_dpo, export_editor_pairs, load_preferences, load_reviews
from report_generator.scoring.reference import reference_runs
from report_generator.scoring.score import score_all

router = APIRouter()
e = lambda v: html.escape(str(v if v is not None else "—"))

CSS = """
:root { --ink:#16202a; --muted:#5b6670; --rule:#d8dde2; --panel:#f1f4f6; --accent:#0f5c6e; --good:#1f7a3f; --bad:#b3261e; }
* { box-sizing: border-box; }
body { margin:0; padding:24px 20px 64px; font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; color:var(--ink); background:#fafbfc; }
main { max-width:1100px; margin:0 auto; }
h1 { font-size:1.5rem; margin:0 0 4px; } h2 { font-size:1.1rem; margin:28px 0 8px; border-top:1px solid var(--rule); padding-top:14px; }
a { color:var(--accent); } .muted { color:var(--muted); }
table { border-collapse:collapse; width:100%; font-size:.88rem; } th,td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--rule); vertical-align:top; }
th { font-size:.75rem; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); }
td.n { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
.good { color:var(--good); font-weight:600; } .bad { color:var(--bad); font-weight:600; }
.due { background:#fff5e6; }
input,select,textarea { font:inherit; padding:4px 6px; border:1px solid var(--rule); border-radius:4px; background:#fff; }
textarea { width:100%; min-height:60px; } input.small { width:80px; }
.row { display:flex; gap:16px; flex-wrap:wrap; align-items:center; }
button { font:inherit; padding:8px 16px; background:var(--accent); color:#fff; border:0; border-radius:4px; cursor:pointer; }
details { margin-top:12px; } pre.memo { white-space:pre-wrap; background:var(--panel); padding:14px; max-height:600px; overflow:auto; font-size:.85rem; }
.cols { display:grid; grid-template-columns:1fr 1fr; gap:16px; } @media (max-width:800px) { .cols { grid-template-columns:1fr; } }
#status { margin-left:12px; }
"""


def page(title: str, body: str) -> HTMLResponse:
    return HTMLResponse(f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' "
                        f"content='width=device-width,initial-scale=1'><title>{e(title)}</title><style>{CSS}</style>"
                        f"</head><body><main>{body}</main></body></html>")


def _horizon_cells(r: Dict[str, Any]) -> str:
    cells = []
    for v in r["horizons"].values():
        if v.get("status") == "done":
            mark = {True: "<span class='good'>✓</span>", False: "<span class='bad'>✗</span>"}.get(v["direction_correct"], "")
            cells.append(f"<td class='n'>{v['excess_pct']:+.1f}% {mark}</td>")
        else:
            cells.append(f"<td class='n muted'>{e(v.get('status'))}</td>")
    return "".join(cells)


@router.get("/scores")
def scores() -> Dict[str, Any]:
    return score_all()


@router.get("/review", response_class=HTMLResponse)
def review_index() -> HTMLResponse:
    result = score_all()
    s = result["summary"]
    rows = "".join(
        f"<tr><td><a href='/review/{e(r['run_id'])}'>{e(r['run_id'])}</a></td><td>{e(r['company'])}</td>"
        f"<td>{e(r['rating'])}</td><td class='n'>{e(r['target'] and round(r['target'], 2))}</td>"
        f"<td class='n'>{e(r['price_at_run'] and round(r['price_at_run'], 2))}</td>"
        f"<td class='n'>{e(r.get('latest', {}).get('excess_pct'))}</td>{_horizon_cells(r)}"
        f"<td class='n'>{len(r['catalysts_due'])}</td><td class='n'>{r['reviews']}</td>"
        f"<td class='n'>{e(r['human_overall'])}</td></tr>"
        for r in result["runs"]
    )
    horizon = " · ".join(f"{h}: {v['completed']} done, hit rate {e(v['hit_rate'])}" for h, v in s["by_horizon"].items())
    body = (f"<h1>Run scores</h1><p class='muted'>As of {e(s['as_of'])}. Excess return is vs SPY. "
            f"✓/✗ = rating direction right or wrong. {horizon}. PoS Brier score: {e(s['pos_brier'])} "
            f"({s['pos_outcomes']} outcomes). <a href='/compare'>Compare two runs</a> · "
            f"<a href='/export/dpo'>Export DPO pairs</a></p>"
            f"<table><thead><tr><th>Run</th><th>Company</th><th>Rating</th><th>Target</th><th>Price at run</th>"
            f"<th>Excess now %</th><th>30d</th><th>90d</th><th>180d</th><th>365d</th><th>Catalysts due</th>"
            f"<th>Reviews</th><th>Avg score</th></tr></thead><tbody>{rows}</tbody></table>")
    return page("Run scores", body)


@router.get("/review/{run_id}", response_class=HTMLResponse)
def review_page(run_id: str) -> HTMLResponse:
    path = RUNS_DIR / f"{run_id}.json"
    if not path.exists():
        raise HTTPException(404, "run not found")
    record = json.loads(path.read_text())
    result = record["result"] or {}
    scored = next(r for r in score_all()["runs"] if r["run_id"] == run_id)
    thesis = result.get("thesis") or {}
    due = {c["event"] for c in scored["catalysts_due"]}

    programs = "".join(
        f"<tr data-program='{e(p['name'])}'><td>{e(p['name'])}</td><td>{e(p['stage'])}</td>"
        f"<td class='n'>{p['probability_of_success']:.0%}</td>"
        f"<td><input class='small pos' type='number' step='0.01' min='0' max='1' placeholder='0-1'></td>"
        f"<td><input class='small peak' type='number' step='1' placeholder='$M'></td>"
        f"<td><select class='outcome'><option value=''>—</option><option>pending</option><option>success</option>"
        f"<option>failure</option></select></td><td><input class='pcomment' placeholder='why'></td></tr>"
        for p in (result.get("pipeline_valuation") or {}).get("programs", [])
    ) or "<tr><td colspan='7' class='muted'>No program-level valuation in this run.</td></tr>"
    catalysts = "".join(
        f"<tr class='{'due' if c['event'] in due else ''}' data-event='{e(c['event'])}'><td>{e(c['timing'])}</td>"
        f"<td>{e(c['event'])}</td><td>{e(c['direction'])}</td><td><select class='coutcome'><option value=''>—</option>"
        f"<option>as expected</option><option>better</option><option>worse</option><option>delayed</option>"
        f"<option>did not happen</option></select></td></tr>"
        for c in thesis.get("catalysts", [])
    ) or "<tr><td colspan='4' class='muted'>No catalysts in this run.</td></tr>"
    sections = "".join(
        f"<tr data-section='{s}'><td>{s.replace('_', ' ')}</td><td><select class='sscore'><option value=''>—</option>"
        + "".join(f"<option>{i}</option>" for i in range(1, 6)) + "</select></td>"
        f"<td><input class='scomment' style='width:100%' placeholder='comment'></td></tr>"
        for s in SECTIONS
    )
    previous = "".join(f"<li>{e(r.get('created_at'))} · {e(r.get('reviewer'))} · overall {e(r.get('overall'))} · "
                       f"{e(r.get('comment', ''))}</li>" for r in load_reviews().get(run_id, [])) or "<li>None yet</li>"
    lt = scored.get("latest", {})
    body = f"""
<p><a href='/review'>← All runs</a></p>
<h1>{e(record['request'].get('company_name'))} · {e(run_id)}</h1>
<p class='muted'>{e(record['request'].get('query'))}</p>
<table><tr><th>Rating</th><th>Target</th><th>Price at run</th><th>Latest</th><th>Return</th><th>vs SPY</th>
<th>30d</th><th>90d</th><th>180d</th><th>365d</th></tr>
<tr><td>{e(scored['rating'])}</td><td class='n'>{e(scored['target'])}</td><td class='n'>{e(scored['price_at_run'])}</td>
<td class='n'>{e(lt.get('price'))} <span class='muted'>{e(lt.get('date'))}</span></td><td class='n'>{e(lt.get('return_pct'))}%</td>
<td class='n'>{e(lt.get('excess_pct'))}%</td>{_horizon_cells(scored)}</tr></table>
<p>{e(thesis.get('summary', ''))}</p>

<h2>Section scores</h2>
<table><thead><tr><th>Section</th><th>Score 1-5</th><th>Comment</th></tr></thead><tbody>{sections}</tbody></table>

<h2>Programs: correct assumptions or record outcomes</h2>
<table><thead><tr><th>Program</th><th>Stage</th><th>Model PoS</th><th>Your PoS</th><th>Your peak sales</th>
<th>Outcome</th><th>Comment</th></tr></thead><tbody>{programs}</tbody></table>

<h2>Catalysts <span class='muted'>(highlighted = date has passed)</span></h2>
<table><thead><tr><th>Timing</th><th>Event</th><th>Expected</th><th>What happened</th></tr></thead><tbody>{catalysts}</tbody></table>

<h2>Errors and overall</h2>
<p>Factual errors you found, one per line:</p><textarea id='errors'></textarea>
<p>Overall comment:</p><textarea id='comment'></textarea>
<div class='row' style='margin-top:12px'>
<label>Overall score <select id='overall'><option value=''>—</option>{"".join(f"<option>{i}</option>" for i in range(1, 6))}</select></label>
<label>Reviewer <input id='reviewer' placeholder='your name'></label>
<button id='save'>Save review</button><span id='status' class='muted'></span></div>

<h2>Previous reviews</h2><ul>{previous}</ul>
<details><summary>Memo</summary><pre class='memo'>{e(result.get('report', ''))}</pre></details>

<script>
const val = (el, sel) => el.querySelector(sel).value.trim();
document.getElementById('save').onclick = async () => {{
  const review = {{ reviewer: document.getElementById('reviewer').value.trim(),
    overall: document.getElementById('overall').value ? Number(document.getElementById('overall').value) : null,
    comment: document.getElementById('comment').value.trim(),
    factual_errors: document.getElementById('errors').value.split('\\n').map(s => s.trim()).filter(Boolean),
    sections: {{}}, program_corrections: {{}}, program_outcomes: {{}}, catalyst_outcomes: {{}} }};
  document.querySelectorAll('[data-section]').forEach(tr => {{
    const score = val(tr, '.sscore'), comment = val(tr, '.scomment');
    if (score || comment) review.sections[tr.dataset.section] = {{ score: score ? Number(score) : null, comment }};
  }});
  document.querySelectorAll('[data-program]').forEach(tr => {{
    const pos = val(tr, '.pos'), peak = val(tr, '.peak'), comment = val(tr, '.pcomment'), outcome = val(tr, '.outcome');
    if (pos || peak || comment) review.program_corrections[tr.dataset.program] =
      {{ probability_of_success: pos ? Number(pos) : null, peak_sales: peak ? Number(peak) : null, comment }};
    if (outcome) review.program_outcomes[tr.dataset.program] = outcome;
  }});
  document.querySelectorAll('[data-event]').forEach(tr => {{
    const o = val(tr, '.coutcome'); if (o) review.catalyst_outcomes[tr.dataset.event] = o;
  }});
  const status = document.getElementById('status');
  const res = await fetch(location.pathname, {{ method: 'POST', headers: {{ 'Content-Type': 'application/json' }}, body: JSON.stringify(review) }});
  status.textContent = res.ok ? 'Saved. Reload to see it under previous reviews.' : 'Not saved: ' + (await res.json()).detail;
}};
</script>"""
    return page(f"Review {run_id}", body)


class ReviewIn(BaseModel):
    reviewer: str = Field(min_length=1)
    overall: Optional[int] = Field(default=None, ge=1, le=5)
    comment: str = ""
    factual_errors: list = []
    sections: dict = {}
    program_corrections: dict = {}
    program_outcomes: dict = {}
    catalyst_outcomes: dict = {}


@router.post("/review/{run_id}")
def save_review(run_id: str, review: ReviewIn) -> Dict[str, Any]:
    try:
        return add_review(run_id, review.model_dump())
    except FileNotFoundError:
        raise HTTPException(404, "run not found")


@router.get("/compare", response_class=HTMLResponse)
def compare_page(a: Optional[str] = None, b: Optional[str] = None) -> HTMLResponse:
    run_ids = [r["run_id"] for r in score_all()["runs"]] + reference_runs()  # references: outside reports
    options = lambda sel: "".join(f"<option {'selected' if r == sel else ''}>{e(r)}</option>" for r in run_ids)
    pick = (f"<form class='row' method='get'><label>A <select name='a'>{options(a)}</select></label>"
            f"<label>B <select name='b'>{options(b)}</select></label><button>Show</button></form>")
    body = f"<p><a href='/review'>← All runs</a></p><h1>Compare two runs</h1>{pick}"
    if a and b and a != b:
        memo = lambda r: e((json.loads((RUNS_DIR / f'{r}.json').read_text()).get('result') or {}).get('report', ''))
        body += (f"<div class='cols' style='margin-top:16px'><div><h2>A · {e(a)}</h2><pre class='memo'>{memo(a)}</pre></div>"
                 f"<div><h2>B · {e(b)}</h2><pre class='memo'>{memo(b)}</pre></div></div>"
                 f"<h2>Which is better?</h2><div class='row'><label>Winner <select id='winner'><option value='a'>A</option>"
                 f"<option value='b'>B</option><option value='tie'>Tie</option></select></label>"
                 f"<label>Reviewer <input id='reviewer'></label></div><p>Why:</p><textarea id='reason'></textarea>"
                 f"<p><button id='save'>Save preference</button><span id='status' class='muted'></span></p>"
                 f"<script>document.getElementById('save').onclick = async () => {{"
                 f"const res = await fetch('/preference', {{method:'POST', headers:{{'Content-Type':'application/json'}},"
                 f"body: JSON.stringify({{run_a:{json.dumps(a)}, run_b:{json.dumps(b)}, winner: document.getElementById('winner').value,"
                 f"reviewer: document.getElementById('reviewer').value.trim(), reason: document.getElementById('reason').value.trim()}})}});"
                 f"document.getElementById('status').textContent = res.ok ? ' Saved.' : ' Not saved: ' + (await res.json()).detail; }};</script>")
    body += f"<h2>Recorded preferences</h2><p class='muted'>{len(load_preferences())} so far.</p>"
    return page("Compare runs", body)


class PreferenceIn(BaseModel):
    run_a: str
    run_b: str
    winner: str
    reviewer: str = Field(min_length=1)
    reason: str = ""


@router.post("/preference")
def save_preference(p: PreferenceIn) -> Dict[str, Any]:
    try:
        return add_preference(p.run_a, p.run_b, p.winner, p.reviewer, p.reason)
    except FileNotFoundError:
        raise HTTPException(404, "run not found")
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@router.get("/export/dpo", response_class=PlainTextResponse)
def export() -> PlainTextResponse:
    path = export_dpo()
    return PlainTextResponse(path.read_text() or "(no preference pairs yet; ties are skipped)\n",
                             headers={"X-Export-Path": str(path)})


@router.get("/export/editor-pairs", response_class=PlainTextResponse)
def export_editor() -> PlainTextResponse:
    path = export_editor_pairs()
    return PlainTextResponse(path.read_text() or "(no revised memos yet)\n", headers={"X-Export-Path": str(path)})
