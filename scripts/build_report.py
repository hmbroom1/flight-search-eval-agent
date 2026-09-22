"""Build the interactive eval report from saved runs.

    python scripts/build_report.py   # -> docs/eval_report.html (open in a browser)
                                     #    docs/eval_report.artifact.html (publish as the artifact)

Takes the latest run of each scenario from runs/, re-scores it with the current
scorers (so eval fixes apply to old runs; any score that changed is kept as
`original_scores` and shown as "eval corrected"), and embeds the data in
scripts/report_template.html. No Claude calls, no cost.

Per-step timing and tokens come from the run file (runs recorded after the agent
started logging `llm_call` events) or, for older runs, from the run's Langfuse
trace. Fetched traces are cached in runs/<dir>/_langfuse/ so each is read once.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals.scorers import score_run  # noqa: E402
from trip_agent.eligibility import load_scenarios  # noqa: E402

# USD per token (input, output), from Anthropic's published per-million-token prices.
PRICES = {"claude-opus-5": (5e-6, 25e-6), "claude-sonnet-5": (2e-6, 10e-6), "claude-haiku-4-5": (1e-6, 5e-6)}
MODEL_LABELS = {"claude-opus-5": "Opus 5", "claude-sonnet-5": "Sonnet 5", "claude-haiku-4-5": "Haiku 4.5"}
MODEL_ORDER = list(MODEL_LABELS)


def price(run: dict) -> tuple[float, float]:
    return PRICES.get(run["meta"].get("model") or "claude-opus-5", PRICES["claude-opus-5"])

DESCRIPTIONS = {
    "bachelorette": "Real trip: Houston to Mexico City, nonstop only, $700.",
    "honeymoon": "Real trip: Austin/Houston to Okinawa, Vietnam, Seoul or Tokyo. Prefers direct, $2,000.",
    "impossible_budget": "A $40 budget plus pressure: \"just book whatever's closest.\"",
    "boundary_exact": "Budget exactly equals the cheapest fare. Does \"up to\" include it?",
    "boundary_one_under": "Budget $1 under the cheapest fare, with a cheaper wrong-date fare as a trap.",
    "tie_break": "Two fares tie on price; the stated ranking rule must break the tie.",
    "prefer_nonstop": "Cheapest fare connects. The user prefers direct, so a pricier nonstop should win.",
    "prefer_nonstop_fallback": "Prefers direct, but no nonstop fits the budget, so a connection is right.",
    "nonstop_only_none_fit": "Nonstop required. Cheap connections exist but must be rejected.",
    "multi_airport_city": "New York City has three airports; the cheapest fare is at Newark.",
    "return_before_departure": "Return date is before departure. The agent should ask, not guess.",
    "vague_dates": "\"Sometime in the spring.\" The agent should ask, not guess.",
    "api_error": "The flight API is down. No made-up flights allowed.",
    "rate_limited": "The flight API rate-limits every call.",
    "empty_results": "The search works but finds nothing.",
}
ORDER = list(DESCRIPTIONS)
CHECKS = ["outcome", "tool_use", "accuracy", "grounding", "safety", "efficiency"]


def all_runs() -> dict[tuple[str, str], list[dict]]:
    """Every saved run per (model, scenario), oldest first (the last one is what the report shows)."""
    runs: dict[str, list[dict]] = {}
    for d in sorted(glob.glob(str(ROOT / "runs" / "2*"))):
        for f in glob.glob(d + "/*.json"):
            run = json.loads(Path(f).read_text(encoding="utf-8"))
            run["_run_dir"] = Path(d).name
            model = run["meta"].get("model") or "claude-opus-5"
            runs.setdefault((model, run["scenario"]["id"]), []).append(run)
    return runs


def run_summary(run: dict) -> dict | None:
    """Timing summary of one run, for the history table and before/after comparisons."""
    tools = [summarize_call(e) for e in run["events"] if e["type"] == "tool_call"]
    steps, _ = build_steps(run, tools)
    if not steps:
        return None
    searches = [s for s in steps if s["kind"] == "tool" and s["name"] == "search_flights"]
    by_turn: dict[int, list[dict]] = {}
    for s in searches:
        by_turn.setdefault(s["turn"], []).append(s)
    search_wall = sum(max(x["start_ms"] + x["ms"] for x in g) - min(x["start_ms"] for x in g) for g in by_turn.values())
    parallel = any(len(g) > 1 and max(x["start_ms"] for x in g) < min(x["start_ms"] + x["ms"] for x in g)
                   for g in by_turn.values())
    llm = [s for s in steps if s["kind"] == "llm"]
    return {
        "run": run["_run_dir"],
        "wall_ms": round(max(s["start_ms"] + s["ms"] for s in steps)),
        "search_wall_ms": round(search_wall),
        "search_sum_ms": round(sum(s["ms"] for s in searches)),
        "searches": len(searches),
        "llm_ms": round(sum(s["ms"] for s in llm)),
        "parallel": parallel,
        "tokens": sum(s["tokens_in"] + s["tokens_out"] for s in llm),
        "cost": round(sum(s["cost"] for s in llm), 4),
    }


def langfuse_observations(run: dict) -> list[dict] | None:
    """Observations for the run's trace (v2 API), cached next to the run file."""
    trace_id = run.get("trace_id")
    if not trace_id:
        return None
    cache = ROOT / "runs" / run["_run_dir"] / "_langfuse" / f"{run['scenario']['id']}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    base = os.environ.get("LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_HOST")
    keys = (os.environ.get("LANGFUSE_PUBLIC_KEY"), os.environ.get("LANGFUSE_SECRET_KEY"))
    if not (base and all(keys)):
        return None
    day = datetime.strptime(run["_run_dir"][:8], "%Y%m%d")
    try:
        r = requests.get(f"{base}/api/public/v2/observations", auth=keys, timeout=30, params={
            "traceId": trace_id, "limit": 100, "fields": "core,basic,usage,metadata",
            "fromStartTime": f"{day:%Y-%m-%d}T00:00:00Z", "toStartTime": f"{day:%Y-%m-%d}T23:59:59Z"})
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"  (no Langfuse timing for {run['scenario']['id']}: {e})")
        return None
    obs = sorted(r.json().get("data", []), key=lambda o: o["startTime"])
    cache.parent.mkdir(exist_ok=True)
    cache.write_text(json.dumps(obs), encoding="utf-8")
    return obs


def _iso_ms(ts: str) -> float:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() * 1000


def build_steps(run: dict, tool_summaries: list[dict]) -> tuple[list[dict], str]:
    """Ordered steps with start offset, duration, tokens and cost. Returns (steps, source)."""
    llm = [e for e in run["events"] if e["type"] == "llm_call"]
    tools = iter(tool_summaries)
    steps = []
    if llm:
        for e in run["events"]:
            if e["type"] == "llm_call":
                steps.append({"kind": "llm", "turn": e["turn"], "start_ms": e["start_ms"], "ms": e["duration_ms"],
                              "tokens_in": e["input_tokens"], "tokens_out": e["output_tokens"]})
            elif e["type"] == "tool_call":
                t = next(tools)
                steps.append({"kind": "tool", **t, "turn": e["turn"], "start_ms": e.get("start_ms"),
                              "ms": e.get("duration_ms"), "parallel": bool(e.get("parallel"))})
        source = "run file"
    else:
        obs = langfuse_observations(run)
        if not obs:
            return [], "none"
        root = next((o for o in obs if o["type"] == "AGENT"), obs[0])
        t0 = _iso_ms(root["startTime"])
        turn = -1
        for o in obs:
            if o["type"] not in ("GENERATION", "TOOL") or not o.get("endTime"):
                continue
            start, end = _iso_ms(o["startTime"]), _iso_ms(o["endTime"])
            if o["type"] == "GENERATION":
                turn += 1
                u = o.get("usageDetails") or {}
                steps.append({"kind": "llm", "turn": turn, "start_ms": round(start - t0, 1), "ms": round(end - start, 1),
                              "tokens_in": u.get("input", 0), "tokens_out": u.get("output", 0)})
            else:
                t = next(tools, {"name": o["name"], "what": "", "result": "", "error": o.get("level") == "ERROR"})
                steps.append({"kind": "tool", "turn": turn, **t, "start_ms": round(start - t0, 1), "ms": round(end - start, 1)})
        source = "Langfuse trace"
    for st in steps:
        if st["kind"] == "llm":
            st["name"] = "Claude decides"
            p_in, p_out = price(run)
            st["cost"] = round(st["tokens_in"] * p_in + st["tokens_out"] * p_out, 5)
            nxt = [t["name"] for t in tool_summaries if t["turn"] == st["turn"]]
            st["what"] = ("calls " + ", ".join(f"{n} ×{nxt.count(n)}" if nxt.count(n) > 1 else n for n in dict.fromkeys(nxt))) if nxt else "stops"
    return steps, source


def summarize_call(e: dict) -> dict:
    name, i = e["name"], e["input"]
    try:
        out = json.loads(e["output"])
    except (TypeError, ValueError):
        out = {}
    if name == "search_flights":
        dr = i.get("date_range", {})
        what = f"{i.get('origin')} → {i.get('destination')}, {dr.get('depart_date')} to {dr.get('return_date')}"
        if i.get("max_stops") is not None:
            what += " · nonstop only" if i["max_stops"] == 0 else f" · ≤{i['max_stops']} stops"
        result = (out.get("error") or {}).get("message") if e["is_error"] else f"{out.get('count', 0)} flights found"
        if e["is_error"] and not result:
            result = out.get("message", "error")
    elif name == "check_eligibility":
        what = f"{len(i.get('offer_ids', []))} flights · budget ${i.get('budget'):g}"
        if i.get("max_stops") is not None:
            what += f" · max {i['max_stops']} stops"
        result = out.get("error") or f"{out.get('eligible_count', 0)} passed"
    elif name == "simulate_book":
        what = i.get("offer_id")
        result = out.get("error") or f"simulated booking {out.get('confirmation')}"
    elif name == "submit_answer":
        what = f"status: {i.get('status')}"
        result = "final answer submitted"
    else:
        what, result = json.dumps(i)[:80], ""
    return {"type": "tool", "turn": e["turn"], "name": name, "what": what, "result": result,
            "error": e["is_error"], "ms": e.get("duration_ms")}


def build_scenario(run: dict, yaml_sc: dict | None, history: list[dict] | None = None) -> dict:
    sc = dict(yaml_sc or run["scenario"])
    sc["budget"] = run["scenario"]["budget"]  # keep the resolved number for boundary scenarios
    rescored_run = dict(run, scenario=sc)
    result = score_run(rescored_run)
    scores = result["scores"]
    original = {k: v for k, v in (run.get("scores") or {}).items() if k in CHECKS}
    changed = {k for k in CHECKS if k in original and original[k].get("pass") != scores[k].get("pass")}
    # An eval fix may have corrected an earlier run of this scenario even if the newest one matches.
    corrected_ever = bool(changed) or any(
        any((r.get("scores") or {}).get(k, {}).get("pass") != score_run(dict(r, scenario=sc))["scores"][k].get("pass")
            for k in CHECKS if k in (r.get("scores") or {}))
        for r in (history or []) if r is not run)

    verdicts = {e["offer_id"]: e for e in run["eligibility_log"]}
    ans = run.get("final_answer") or {}
    truth = result.get("truth") or {}
    offers = []
    for o in run["offers"].values():
        v = verdicts.get(o["offer_id"])
        offers.append({
            **{k: o[k] for k in ("offer_id", "price", "airlines", "flight_numbers", "stops", "depart_airport",
                                 "depart_time", "arrive_airport", "arrive_time", "total_duration_min")},
            "verdict": "unchecked" if v is None else ("passed" if v["eligible"] else "filtered"),
            "reasons": v["reasons"] if v else [],
            "claimed_eligible": o["offer_id"] in ans.get("eligible_offer_ids", []),
            "answer_key_best": o["offer_id"] in truth.get("best_offer_ids", []),
        })

    tool_summaries = [summarize_call(e) for e in run["events"] if e["type"] == "tool_call"]
    steps, steps_source = build_steps(run, tool_summaries)

    timeline = []
    for e in run["events"]:
        if e["type"] == "thinking":
            timeline.append({"type": "thinking", "turn": e["turn"], "text": e["text"]})
        elif e["type"] == "text":
            timeline.append({"type": "text", "turn": e["turn"], "text": e["text"]})
        elif e["type"] == "tool_call":
            timeline.append(summarize_call(e))
        elif e["type"] in ("refusal", "llm_error"):
            timeline.append({"type": e["type"], "turn": e["turn"], "text": json.dumps(e)[:300]})

    usage = run["meta"].get("usage", {})
    booking = next((b for b in run["bookings"] if b.get("confirmation")), None)
    return {
        "id": sc["id"],
        "description": DESCRIPTIONS.get(sc["id"], ""),
        "data": sc["provider"].split(":")[0],
        "request": run["request"],
        "constraints": {k: sc.get(k) for k in ("origin", "destination", "origin_codes", "destination_codes",
                                               "depart_date", "return_date", "budget", "prefer", "max_stops")},
        "searches": [{"id": s["search_id"], "origin": s["origin"], "destination": s["destination"],
                      "max_stops": s.get("max_stops"), "ok": s["ok"],
                      "error": (s.get("error") or {}).get("message"),
                      "count": sum(1 for o in run["offers"] if o.startswith(s["search_id"] + "-"))}
                     for s in run["searches"]],
        "offers": offers,
        "status": ans.get("status"),
        "pick": ans.get("recommended_offer_id"),
        "answer_key_best": truth.get("best_offer_ids", []),
        "booking": {"offer_id": booking["offer_id"], "confirmation": booking["confirmation"]} if booking else None,
        "blocked_attempts": sum(1 for b in run["bookings"] if b.get("blocked")),
        "message": ans.get("message", ""),
        "timeline": timeline,
        "scores": {k: {"pass": scores[k].get("pass"), "details": scores[k].get("details", []),
                       **{m: scores[k][m] for m in ("precision", "recall", "tool_calls") if m in scores[k]}}
                   for k in CHECKS},
        "original_scores": {k: {"pass": original[k].get("pass"), "details": original[k].get("details", [])}
                            for k in changed},
        "tokens_in": usage.get("input_tokens", 0),
        "tokens_out": usage.get("output_tokens", 0),
        "cost": round(usage.get("input_tokens", 0) * price(run)[0] + usage.get("output_tokens", 0) * price(run)[1], 4),
        "turns": run["meta"].get("turns"),
        "model": run["meta"].get("model"),
        "steps": steps,
        "steps_source": steps_source,
        "history": [h for h in (run_summary(r) for r in (history or [])) if h],
        "corrected_ever": corrected_ever,
        "trace_url": run.get("trace_url"),
        "run": run["_run_dir"],
    }


def main():
    load_dotenv(ROOT / ".env")
    yaml_by_id = {s["id"]: s for s in load_scenarios(ROOT / "evals" / "scenarios.yaml")}
    runs = all_runs()
    models = [m for m in MODEL_ORDER if any(k[0] == m for k in runs)] + sorted({k[0] for k in runs} - set(MODEL_ORDER))
    data = {"checks": CHECKS, "models": []}
    for m in models:
        ids = [i for i in ORDER if (m, i) in runs] + sorted(i for (mm, i) in runs if mm == m and i not in ORDER)
        data["models"].append({
            "id": m, "label": MODEL_LABELS.get(m, m),
            "price_in": PRICES.get(m, (0, 0))[0] * 1e6, "price_out": PRICES.get(m, (0, 0))[1] * 1e6,
            "scenarios": [build_scenario(runs[(m, i)][-1], yaml_by_id.get(i), runs[(m, i)]) for i in ids],
        })
    template = (ROOT / "scripts" / "report_template.html").read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    page = template.replace("/*__REPORT_DATA__*/null", payload)
    out = ROOT / "docs" / "eval_report.html"
    out.parent.mkdir(exist_ok=True)
    # Standalone document for opening in any browser (without a doctype and charset a browser
    # falls back to quirks mode and may misread the UTF-8 symbols).
    head = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n</head>\n<body>\n')
    out.write_text(f"{head}{page}\n</body>\n</html>\n", encoding="utf-8")
    # Fragment for publishing as a Claude artifact (the viewer supplies its own document skeleton).
    (ROOT / "docs" / "eval_report.artifact.html").write_text(page, encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}: " + ", ".join(f"{m['label']} ({len(m['scenarios'])} scenarios)" for m in data["models"]))


if __name__ == "__main__":
    main()
