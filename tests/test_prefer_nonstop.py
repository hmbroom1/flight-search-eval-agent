"""The 'direct flights as much as possible' preference: agent ranking, eval ground truth, tracker tiers."""

from datetime import date, timedelta

import pytest
from conftest import BASE, DR, FIXTURES, Script, details_of

from trip_agent import tracker
from trip_agent.eligibility import pick_best
from trip_agent.providers import FixtureProvider


def tradeoff(budget, prefer="fewest_stops"):
    sc = dict(BASE, budget=budget, prefer=prefer, expect={"status": ["eligible_found"], "book": "auto"})
    s = Script(FixtureProvider(FIXTURES / "nonstop_tradeoff.json"), sc)
    s.call("search_flights", origin="AUS", destination="MIA", date_range=DR)
    chk = s.call("check_eligibility", offer_ids=list(s.ctx.offers), budget=budget, date_range=DR)
    return s, [r["offer_id"] for r in chk["results"] if r["eligible"]]


def recommend(s, eligible, oid):
    conf = s.call("simulate_book", offer_id=oid)["confirmation"]
    s.answer("eligible_found", eligible, oid, details_of(s.ctx.offers[oid]), conf)
    return s.score()


# Fixture offers: s1-o0 $210 1-stop, s1-o1 $265 nonstop, s1-o2 $280 nonstop, s1-o3 $228 1-stop

def test_ranking_rule_follows_preference():
    s, _ = tradeoff(300)
    offers = list(s.ctx.offers.values())
    assert pick_best(offers, "price")["offer_id"] == "s1-o0"
    assert pick_best(offers, "fewest_stops")["offer_id"] == "s1-o1"


def test_picking_cheapest_connection_fails_when_direct_preferred():
    s, eligible = tradeoff(300)
    acc = recommend(s, eligible, "s1-o0")["accuracy"]
    assert acc["pass"] is False and not acc["recommendation_correct"]


def test_cheapest_nonstop_passes_when_direct_preferred():
    s, eligible = tradeoff(300)
    assert all(v["pass"] in (True, None) for v in recommend(s, eligible, "s1-o1").values())


def test_falls_back_to_connection_when_no_nonstop_fits():
    s, eligible = tradeoff(250)
    assert set(eligible) == {"s1-o0", "s1-o3"}
    scores = recommend(s, eligible, "s1-o0")
    assert all(v["pass"] in (True, None) for v in scores.values()), scores


def test_price_preference_unchanged():
    s, eligible = tradeoff(300, prefer="price")
    assert recommend(s, eligible, "s1-o0")["accuracy"]["pass"] is True


# --- tracker tiers -------------------------------------------------------------

TODAY = "2026-10-01"


def seg(fn, frm, to, t):
    return {"departure_airport": {"id": frm, "time": f"2027-03-12 {t}"},
            "arrival_airport": {"id": to, "time": f"2027-03-12 {t}"},
            "airline": "Test", "flight_number": fn}


def raw(nonstop_price, connect_price=150):
    items = [{"flights": [seg("AA 1", "AUS", "IAH", "06:00"), seg("AA 2", "IAH", "MIA", "09:00")],
              "layovers": [{"id": "IAH", "duration": 60}], "price": connect_price}]
    if nonstop_price is not None:
        items.append({"flights": [seg("AA 3", "AUS", "MIA", "07:00")], "price": nonstop_price})
    return {"best_flights": items}


class Seq:
    def __init__(self, raws):
        self.raws = iter(raws)

    def search(self, *a):
        return next(self.raws)


@pytest.fixture
def trip(monkeypatch):
    t = dict(BASE, id="t", track=True, budget=400, prefer="fewest_stops")
    monkeypatch.setattr(tracker, "tracked_trips", lambda: [t])
    return t


def test_tracker_follows_nonstop_tier(tmp_path, trip, monkeypatch):
    sent = []
    monkeypatch.setattr(tracker, "send_email", lambda s, b: sent.append((s, b)))
    path = tmp_path / "h.jsonl"
    d0 = date.fromisoformat(TODAY)
    # 15 days of nonstop at $300 (connections at a cheaper $150 the whole time), then nonstop drops to $250.
    days = [(d0 - timedelta(days=i)).isoformat() for i in range(15, 0, -1)] + [TODAY]
    provider = Seq([raw(300)] * 15 + [raw(250)])
    for d in days:
        log = tracker.run(provider=provider, today=d, history_path=path)
    assert any("nonstop" in line and "emailed" in line for line in log), log
    subject, body = sent[-1]
    assert "nonstop $250" in subject and "AA 3" in body


def test_tracker_falls_back_to_one_stop_tier(tmp_path, trip):
    log = tracker.run(dry_run=True, provider=Seq([raw(None)]), today=TODAY, history_path=tmp_path / "h.jsonl")
    assert log[0].startswith("t: 1-stop ")
