"""Searches requested together run concurrently, with deterministic ids and per-call results."""

import json
import time
from datetime import date
from types import SimpleNamespace

from conftest import DR, FIXTURES

from trip_agent.agent import run_tools
from trip_agent.providers import FixtureProvider
from trip_agent.tools import RunContext
from trip_agent.tracing import Tracer


class SlowProvider(FixtureProvider):
    def search(self, *args, **kw):
        time.sleep(0.3)
        return super().search(*args, **kw)


def uses():
    return [SimpleNamespace(id=f"t{i}", name="search_flights",
                            input={"origin": "AUS", "destination": dest, "date_range": DR, "max_stops": None})
            for i, dest in enumerate(["MIA", "Miami", "MIA,FLL"])]  # middle one is invalid (not IATA)


def run(parallel):
    ctx = RunContext(provider=SlowProvider(FIXTURES / "standard.json"), today=date(2026, 9, 16))
    t0 = time.perf_counter()
    out = run_tools(uses(), ctx, Tracer(enabled=False), parallel_search=parallel)
    return ctx, out, time.perf_counter() - t0


def test_parallel_is_faster_and_deterministic():
    ctx, out, elapsed = run(parallel=True)
    assert elapsed < 0.5  # two valid searches overlap: ~0.3s, not ~0.6s
    assert [s["search_id"] for s in ctx.searches] == ["s1", "s2", "s3"]  # ids follow request order
    assert json.loads(out["t0"][0])["search_id"] == "s1"
    assert out["t1"][1] is True and "IATA" in out["t1"][0]  # invalid one fails without a fetch
    assert json.loads(out["t2"][0])["search_id"] == "s3"
    assert "s1-o0" in ctx.offers and "s3-o0" in ctx.offers


def test_sequential_mode_matches_results():
    ctx_p, out_p, _ = run(parallel=True)
    ctx_s, out_s, elapsed = run(parallel=False)
    assert elapsed >= 0.6
    assert {k: v[:2] for k, v in out_p.items()} == {k: v[:2] for k, v in out_s.items()}
