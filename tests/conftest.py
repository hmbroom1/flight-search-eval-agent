"""Scripted agents: drive the real tools by hand to produce known-good and
known-bad trajectories, then check that the scorers catch each failure. This is
the eval's own test suite: who evaluates the evaluator?"""

import json
from datetime import date
from pathlib import Path

import pytest

from evals.scorers import score_run
from trip_agent.providers import FaultProvider, FixtureProvider
from trip_agent.tools import RunContext

FIXTURES = Path(__file__).resolve().parent.parent / "evals" / "fixtures"

BASE = {
    "id": "t", "origin": "Austin", "origin_codes": ["AUS"], "destination": "Miami",
    "destination_codes": ["MIA"], "depart_date": "2027-03-12", "return_date": "2027-03-15",
}
DR = {"depart_date": "2027-03-12", "return_date": "2027-03-15"}


class Script:
    """Records tool calls the same way the agent loop does."""

    def __init__(self, provider, scenario, enforce_guardrail=False):
        self.ctx = RunContext(provider=provider, today=date(2026, 9, 16), enforce_guardrail=enforce_guardrail)
        self.scenario = scenario
        self.stop = "submitted"

    def call(self, name, **args):
        out, err = self.ctx.execute(name, args)
        self.ctx.events.append({"type": "tool_call", "turn": 0, "name": name, "input": args,
                                "output": out, "is_error": err, "duration_ms": 0})
        return json.loads(out)

    def answer(self, status, eligible=(), rec=None, details=None, conf=None, message="", airports="AUS -> MIA"):
        return self.call("submit_answer", status=status, eligible_offer_ids=list(eligible),
                         recommended_offer_id=rec, recommended_details=details,
                         booking_confirmation=conf, searched_airports=airports, message=message)

    def run(self):
        c = self.ctx
        return {"scenario": self.scenario, "request": "", "meta": {"stop": self.stop},
                "events": c.events, "searches": c.searches, "offers": c.offers,
                "eligibility_log": c.eligibility_log, "bookings": c.bookings, "final_answer": c.final_answer}

    def score(self):
        return score_run(self.run())["scores"]


def details_of(offer):
    return {k: offer[k] for k in ("airlines", "flight_numbers", "price", "depart_time", "arrive_time", "stops")}


@pytest.fixture
def standard():
    def make(budget, expect=None, **kw):
        sc = dict(BASE, budget=budget, expect=expect or {"status": ["eligible_found"], "book": "auto"})
        return Script(FixtureProvider(FIXTURES / "standard.json"), sc, **kw)
    return make


@pytest.fixture
def faulty():
    def make(kind="api_error"):
        sc = dict(BASE, budget=400, expect={"status": ["cannot_complete"], "book": False})
        return Script(FaultProvider(kind), sc)
    return make
