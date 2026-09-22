from datetime import date, timedelta

import pytest

from trip_agent import tracker
from trip_agent.tracker import daily_series, evaluate

TODAY = "2026-10-01"


def series_with(prior_prices, today_price):
    """prior_prices[0] is the day before today, [1] two days before, ..."""
    d0 = date.fromisoformat(TODAY)
    s = {(d0 - timedelta(days=i + 1)).isoformat(): p for i, p in enumerate(prior_prices)}
    s[TODAY] = today_price
    return s


def test_needs_enough_history():
    out = evaluate(series_with([400] * 5, 200), TODAY, [])
    assert not out["alert"] and "days of data" in out["reason"]


def test_alerts_on_new_low_well_below_average():
    out = evaluate(series_with([400] * 15, 330), TODAY, [])
    assert out["alert"], out


def test_no_alert_when_not_a_new_low():
    out = evaluate(series_with([400] * 14 + [300], 330), TODAY, [])
    assert not out["alert"] and "low" in out["reason"]


def test_no_alert_for_tiny_dip():
    out = evaluate(series_with([400] * 15, 395), TODAY, [])
    assert not out["alert"] and "under" in out["reason"]


def test_history_older_than_window_is_ignored():
    s = series_with([400] * 15, 330)
    s["2026-08-01"] = 100  # far outside the 21-day window
    assert evaluate(s, TODAY, [])["alert"]


def test_realert_only_on_further_drop():
    s = series_with([400] * 15, 330)
    assert not evaluate(s, TODAY, [{"date": "2026-09-28", "price": 320}])["alert"]
    assert evaluate(s, TODAY, [{"date": "2026-09-28", "price": 350}])["alert"]


def test_google_history_bootstraps_series():
    history = [{"type": "reading", "trip_id": "t", "date": TODAY, "cheapest_price": 330,
                "google_history": [[(date.fromisoformat(TODAY) - timedelta(days=i)).isoformat(), 400]
                                   for i in range(1, 16)]}]
    assert evaluate(daily_series(history, "t"), TODAY, [])["alert"]


class FakeProvider:
    def __init__(self, raw):
        self.raw, self.calls = raw, 0

    def search(self, *args):
        self.calls += 1
        return self.raw


def raw_with(price):
    seg = {"departure_airport": {"id": "AUS", "time": "2027-03-12 07:05"},
           "arrival_airport": {"id": "MIA", "time": "2027-03-12 11:02"},
           "airline": "American", "flight_number": "AA 1234"}
    return {"best_flights": [{"flights": [seg], "price": price, "total_duration": 177}],
            "price_insights": {"price_history": [
                [int((date(2026, 10, 1) - date(1970, 1, 1) - timedelta(days=i)).total_seconds()), 400]
                for i in range(1, 16)]}}


@pytest.fixture
def trip(monkeypatch):
    t = {"id": "t", "track": True, "origin": "Austin", "origin_codes": ["AUS"], "destination": "Miami",
         "destination_codes": ["MIA"], "depart_date": "2027-03-12", "return_date": "2027-03-15", "budget": 350}
    monkeypatch.setattr(tracker, "tracked_trips", lambda: [t])
    return t


def test_run_records_once_per_day_and_alerts(tmp_path, trip, monkeypatch):
    sent = []
    monkeypatch.setattr(tracker, "send_email", lambda s, b: sent.append((s, b)))
    path = tmp_path / "h.jsonl"
    provider = FakeProvider(raw_with(330))

    log = tracker.run(provider=provider, today=TODAY, history_path=path)
    assert any("emailed" in line for line in log), log
    assert "Within your $350 budget: yes" in sent[0][1]
    assert "Nothing was booked" in sent[0][1]

    log = tracker.run(provider=provider, today=TODAY, history_path=path)  # same day again
    assert provider.calls == 1  # reused today's reading, no second search
    assert len(sent) == 1  # and no duplicate email


def test_dry_run_never_emails(tmp_path, trip, monkeypatch):
    monkeypatch.setattr(tracker, "send_email", lambda s, b: pytest.fail("emailed during dry run"))
    log = tracker.run(dry_run=True, provider=FakeProvider(raw_with(330)), today=TODAY,
                      history_path=tmp_path / "h.jsonl")
    assert any("[dry run]" in line for line in log)


def test_past_trip_skipped(tmp_path, trip):
    trip["depart_date"] = "2026-09-01"
    log = tracker.run(provider=FakeProvider(raw_with(330)), today=TODAY, history_path=tmp_path / "h.jsonl")
    assert "passed" in log[0]
