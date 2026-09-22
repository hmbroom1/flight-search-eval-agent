"""Run scenarios through the agent, score each trajectory, report.

    python -m evals.run_evals                      # all runnable scenarios
    python -m evals.run_evals -s tie_break,api_error
    python -m evals.run_evals --cache-mode replay  # live scenarios from cache only
    python -m evals.run_evals --judge              # add LLM-judge reasoning score
    python -m evals.run_evals --enforce-guardrail  # block unsafe bookings at runtime
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import date, datetime
from pathlib import Path

from dotenv import load_dotenv

from trip_agent.agent import run_agent
from trip_agent.eligibility import load_scenarios, validate_date_range
from trip_agent.providers import CachedProvider, FaultProvider, FixtureProvider, SearchError, SerpApiProvider
from trip_agent.tools import RunContext
from trip_agent.tracing import Tracer

from .ground_truth import cheapest_price
from .scorers import score_run

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "evals" / "fixtures"
RUBRIC = ["outcome", "tool_use", "accuracy", "grounding", "safety", "efficiency"]


def has_todo(sc: dict) -> bool:
    return "TODO" in json.dumps(sc)


def build_provider(spec: str, cache_mode: str):
    kind, _, arg = spec.partition(":")
    if kind == "live":
        inner = None if cache_mode == "replay" else SerpApiProvider()
        return CachedProvider(inner, ROOT / "data" / "cache", cache_mode)
    if kind == "fixture":
        return FixtureProvider(FIXTURES / f"{arg}.json")
    if kind == "fault":
        return FaultProvider(arg)
    raise ValueError(f"unknown provider spec {spec!r}")


def resolve_budget(sc: dict, provider) -> dict:
    """Turn {cheapest_plus: N} into a number by looking at the same data the agent will see."""
    sc = dict(sc)
    if isinstance(sc["budget"], dict):
        raw = provider.search(",".join(sc["origin_codes"]), ",".join(sc["destination_codes"]),
                              sc["depart_date"], sc["return_date"], sc.get("max_stops"))
        cheapest = cheapest_price(raw, sc)
        if cheapest is None:
            raise RuntimeError(f"{sc['id']}: no fares found to anchor a boundary budget")
        sc["budget"] = cheapest + sc["budget"]["cheapest_plus"]
    return sc


def reference_search(sc: dict, provider) -> list[dict]:
    """Harness-side search over the scenario's full airport list (the 'oracle' query)."""
    if validate_date_range(sc.get("depart_date"), sc.get("return_date")) is not None:
        return []
    try:
        raw = provider.search(",".join(sc["origin_codes"]), ",".join(sc["destination_codes"]),
                              sc["depart_date"], sc["return_date"], sc.get("max_stops"))
    except SearchError:
        return []
    return [{"search_id": "ref", "ok": True, "raw": raw, "depart_date": sc["depart_date"],
             "return_date": sc["return_date"]}]


def run_scenario(sc: dict, args, tracer: Tracer, session_id: str) -> dict:
    provider = build_provider(sc["provider"], args.cache_mode)
    sc = resolve_budget(sc, provider)
    reference = reference_search(sc, provider)
    request = " ".join(sc["request"].format(**sc).split())
    ctx = RunContext(provider=provider, today=date.today(), enforce_guardrail=args.enforce_guardrail)

    model = args.model or os.environ.get("AGENT_MODEL", "claude-opus-5")
    with tracer.trace("plan-trip", session_id=session_id, tags=[sc["id"], sc["provider"], model],
                      metadata={"scenario": sc["id"], "budget": sc["budget"]}, input=request) as (root, trace_id):
        meta = run_agent(request, ctx, tracer, model=model, max_turns=args.max_turns,
                         parallel_search=not args.sequential_tools)
        root.update(output=ctx.final_answer, metadata={"stop": meta["stop"]})

    run = {
        "scenario": sc, "request": request, "provider": provider.name, "meta": meta,
        "events": ctx.events, "searches": ctx.searches, "offers": ctx.offers,
        "eligibility_log": ctx.eligibility_log, "bookings": ctx.bookings,
        "final_answer": ctx.final_answer, "reference_searches": reference,
        "trace_id": trace_id, "trace_url": tracer.trace_url(trace_id),
    }
    result = score_run(run)
    if args.judge:
        from .judge import judge_reasoning
        result["scores"]["reasoning_judge"] = judge_reasoning(run)
    run.update(result)

    for name, s in result["scores"].items():
        if s.get("pass") is not None:
            tracer.score(trace_id, name, 1.0 if s["pass"] else 0.0, comment="; ".join(s.get("details", []))[:500] or None)
    acc = result["scores"]["accuracy"]
    if acc.get("pass") is not None:
        tracer.score(trace_id, "precision", acc["precision"])
        tracer.score(trace_id, "recall", acc["recall"])
    tracer.score(trace_id, "tool_calls", result["scores"]["efficiency"]["tool_calls"])
    return run


def mark(s: dict) -> str:
    return "n/a" if s.get("pass") is None else ("PASS" if s["pass"] else "FAIL")


def main(argv=None):
    load_dotenv(ROOT / ".env")
    p = argparse.ArgumentParser()
    p.add_argument("-s", "--scenarios", help="comma-separated scenario ids")
    p.add_argument("--file", default=str(ROOT / "evals" / "scenarios.yaml"))
    p.add_argument("--cache-mode", choices=["record", "replay", "live"], default="record")
    p.add_argument("--model", default=None)
    p.add_argument("--max-turns", type=int, default=12)
    p.add_argument("--judge", action="store_true")
    p.add_argument("--enforce-guardrail", action="store_true")
    p.add_argument("--sequential-tools", action="store_true", help="disable parallel searches (for comparison)")
    args = p.parse_args(argv)

    scenarios = load_scenarios(Path(args.file))
    if args.scenarios:
        wanted = set(args.scenarios.split(","))
        scenarios = [s for s in scenarios if s["id"] in wanted]

    model = args.model or os.environ.get("AGENT_MODEL", "claude-opus-5")
    out_dir = ROOT / "runs" / f"{datetime.now():%Y%m%d-%H%M%S}-{model.removeprefix('claude-')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    session_id = f"eval-{out_dir.name}-{uuid.uuid4().hex[:6]}"
    tracer = Tracer()
    print(f"Langfuse tracing: {'on' if tracer.enabled else 'off (no keys)'}; results -> {out_dir}")

    columns = RUBRIC + (["reasoning_judge"] if args.judge else [])
    rows = []
    try:
        for sc in scenarios:
            if has_todo(sc):
                print(f"- {sc['id']}: skipped (fill in TODO fields in scenarios.yaml)")
                continue
            print(f"- {sc['id']} ...", flush=True)
            try:
                run = run_scenario(sc, args, tracer, session_id)
            except Exception as e:  # keep the suite going; record the harness failure
                print(f"  harness error: {type(e).__name__}: {e}")
                rows.append((sc["id"], ["ERR"] * len(columns), [f"harness error: {e}"], None))
                continue
            (out_dir / f"{sc['id']}.json").write_text(json.dumps(run, indent=2, default=str), encoding="utf-8")
            scores = run["scores"]
            failures = [f"{k}: {d}" for k in columns for d in scores.get(k, {}).get("details", [])
                        if scores.get(k, {}).get("pass") is False]
            rows.append((sc["id"], [mark(scores.get(k, {})) for k in columns], failures, run["trace_url"]))
            print("  " + "  ".join(f"{k}={mark(scores.get(k, {}))}" for k in columns))
    finally:
        tracer.flush()

    lines = ["| scenario | " + " | ".join(columns) + " |", "|" + "---|" * (len(columns) + 1)]
    lines += [f"| {sid} | " + " | ".join(marks) + " |" for sid, marks, _, _ in rows]
    lines.append("")
    for sid, _, failures, url in rows:
        if failures or url:
            lines.append(f"### {sid}" + (f" ([trace]({url}))" if url else ""))
            lines += [f"- {f}" for f in failures] or ["- all checks passed"]
    summary = "\n".join(lines)
    (out_dir / "summary.md").write_text(summary + "\n", encoding="utf-8")
    print("\n" + summary)
    return 0 if all("FAIL" not in m and "ERR" not in m for _, m, _, _ in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
