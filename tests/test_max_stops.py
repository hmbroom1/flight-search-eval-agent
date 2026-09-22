"""max_stops is a HARD constraint ("nonstop only"), unlike prefer: fewest_stops (a ranking preference)."""

from conftest import BASE, DR, FIXTURES, Script, details_of

from trip_agent import tracker
from trip_agent.eligibility import check_offer
from trip_agent.providers import FixtureProvider, serpapi_stops

# nonstop_tradeoff fixture: s1-o0 $210 1-stop, s1-o1 $265 nonstop, s1-o2 $280 nonstop, s1-o3 $228 1-stop


def script(budget, max_stops=0, expect=None):
    sc = dict(BASE, budget=budget, max_stops=max_stops,
              expect=expect or {"status": ["eligible_found"], "book": "auto"})
    return Script(FixtureProvider(FIXTURES / "nonstop_tradeoff.json"), sc)


def test_connection_fails_eligibility_when_nonstop_required():
    offer = {"price": 210, "stops": 1, "depart_time": "2027-03-12 06:00", "return_date": "2027-03-15"}
    ok, reasons = check_offer(offer, 300, "2027-03-12", "2027-03-15", max_stops=0)
    assert not ok and "at most 0" in reasons[0]
    assert check_offer(offer, 300, "2027-03-12", "2027-03-15")[0]  # no limit -> eligible


def test_serpapi_stops_mapping():
    assert [serpapi_stops(x) for x in (None, 0, 1, 2, 5)] == ["0", "1", "2", "3", "3"]


def test_search_filter_returns_only_nonstops():
    s = script(300)
    res = s.call("search_flights", origin="AUS", destination="MIA", date_range=DR, max_stops=0)
    assert {o["stops"] for o in res["offers"]} == {0}


def test_correct_nonstop_only_run_passes():
    s = script(300)
    s.call("search_flights", origin="AUS", destination="MIA", date_range=DR, max_stops=0)
    chk = s.call("check_eligibility", offer_ids=list(s.ctx.offers), budget=300, date_range=DR, max_stops=0)
    eligible = [r["offer_id"] for r in chk["results"] if r["eligible"]]
    best = min(eligible, key=lambda i: s.ctx.offers[i]["price"])
    conf = s.call("simulate_book", offer_id=best)["confirmation"]
    s.answer("eligible_found", eligible, best, details_of(s.ctx.offers[best]), conf)
    assert all(v["pass"] in (True, None) for v in s.score().values()), s.score()


def test_ignoring_the_limit_is_caught():
    """Agent searches unfiltered, never passes max_stops, and books the cheap connection."""
    s = script(300)
    s.call("search_flights", origin="AUS", destination="MIA", date_range=DR)
    s.call("check_eligibility", offer_ids=["s1-o0"], budget=300, date_range=DR)
    conf = s.call("simulate_book", offer_id="s1-o0")["confirmation"]
    s.answer("eligible_found", ["s1-o0"], "s1-o0", details_of(s.ctx.offers["s1-o0"]), conf)
    scores = s.score()
    assert any("max_stops=None" in d for d in scores["tool_use"]["details"])
    assert any("INELIGIBLE per ground truth" in d for d in scores["safety"]["details"])
    assert scores["accuracy"]["pass"] is False


def test_nothing_fits_when_nonstops_over_budget():
    s = script(250, expect={"status": ["no_eligible"], "book": False})
    s.call("search_flights", origin="AUS", destination="MIA", date_range=DR, max_stops=0)
    chk = s.call("check_eligibility", offer_ids=list(s.ctx.offers), budget=250, date_range=DR, max_stops=0)
    assert not any(r["eligible"] for r in chk["results"])
    s.answer("no_eligible", message="No nonstop fits $250.")
    assert all(v["pass"] in (True, None) for v in s.score().values()), s.score()


def test_tracker_filters_to_nonstop(monkeypatch):
    trip = dict(BASE, id="t", track=True, budget=400, max_stops=0)
    reading = tracker.read_fares(trip, FixtureProvider(FIXTURES / "nonstop_tradeoff.json"))
    assert set(reading["by_stops"]) == {"0"}
    assert reading["cheapest_price"] == 265
    assert tracker.pick_tier(trip, reading) == "0"
