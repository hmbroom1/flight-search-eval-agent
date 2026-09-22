"""Daily fare tracker with email alerts. No LLM, no booking; just search, record, compare.

    python -m trip_agent.tracker               # check all tracked trips, email on a deal
    python -m trip_agent.tracker --dry-run     # check and print, never email
    python -m trip_agent.tracker --test-email  # send a test email and exit

Trips come from evals/scenarios.yaml entries with `track: true` (TODO fields skipped).
One live search per trip per day; a second run on the same day reuses today's reading.

Trips with `prefer: fewest_stops` are tracked per stop count: each day the tracker
follows the most direct tier that has a fare (nonstop if any, else 1-stop, ...)
and compares it only with earlier fares in that same tier. Google's price history
isn't broken out by stops, so it only seeds the "any" tier (trips without that
preference); a nonstop tier needs MIN_DAYS of our own readings before alerting.

Alert rule (all must hold):
  - at least MIN_DAYS distinct days of history in the last WINDOW_DAYS (Google's own
    price history from the response counts, so alerts can start before we've
    collected two weeks ourselves)
  - today's fare is strictly lower than every earlier fare in the window
  - today's fare is at least DROP_PCT below the window average
  - no earlier alert in the window at this price or lower (re-alert only on a further drop)
"""

from __future__ import annotations

import argparse
import json
import os
import smtplib
import sys
from datetime import date, datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path

from dotenv import load_dotenv

from .eligibility import load_scenarios
from .providers import SearchError, SerpApiProvider
from .tools import normalize_offers

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = ROOT / "evals" / "scenarios.yaml"
HISTORY = ROOT / "data" / "price_history.jsonl"

WINDOW_DAYS = 21
MIN_DAYS = 14
DROP_PCT = 0.10


# --- history ------------------------------------------------------------------

def load_history(path: Path = HISTORY) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_history(record: dict, path: Path = HISTORY):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def daily_series(history: list[dict], trip_id: str, tier: str = "any") -> dict[str, float]:
    """date -> lowest fare that day. Tier "any" merges in Google's history; "0", "1", ... are stop counts."""
    series: dict[str, float] = {}
    for r in history:
        if r["trip_id"] != trip_id or r.get("type") != "reading":
            continue
        if tier == "any":
            points = [(r["date"], r["cheapest_price"])] if r.get("cheapest_price") is not None else []
            points += [(d, p) for d, p in r.get("google_history", [])]
        else:
            fare = (r.get("by_stops") or {}).get(tier)
            points = [(r["date"], fare["price"])] if fare else []
        for d, p in points:
            series[d] = min(p, series.get(d, p))
    return series


# --- the rule -------------------------------------------------------------------

def evaluate(series: dict[str, float], today: str, alerts: list[dict]) -> dict:
    """Pure decision function. Returns {'alert': bool, 'reason': str, ...stats}."""
    if today not in series:
        return {"alert": False, "reason": "no fare today"}
    price = series[today]
    start = (date.fromisoformat(today) - timedelta(days=WINDOW_DAYS)).isoformat()
    prior = {d: p for d, p in series.items() if start <= d < today}
    stats = {"price": price, "days_of_data": len(prior) + 1}
    if len(prior) + 1 < MIN_DAYS:
        return {**stats, "alert": False, "reason": f"only {len(prior) + 1} days of data (need {MIN_DAYS})"}
    window = list(prior.values()) + [price]
    avg = sum(window) / len(window)
    low = min(prior.values())
    stats.update(window_avg=round(avg, 2), prior_low=low, pct_below_avg=round(1 - price / avg, 4))
    if price >= low:
        return {**stats, "alert": False, "reason": f"${price:g} is not below the {WINDOW_DAYS}-day low of ${low:g}"}
    if price > avg * (1 - DROP_PCT):
        return {**stats, "alert": False, "reason": f"${price:g} is less than {DROP_PCT:.0%} under the ${avg:.0f} average"}
    recent = [a["price"] for a in alerts if start <= a["date"] < today]
    if recent and price >= min(recent):
        return {**stats, "alert": False, "reason": f"already alerted at ${min(recent):g} in this window"}
    return {**stats, "alert": True, "reason": "new low and well below average"}


# --- search -------------------------------------------------------------------

def tracked_trips(path: Path = SCENARIOS) -> list[dict]:
    trips = load_scenarios(path)
    return [t for t in trips if t.get("track") and "TODO" not in json.dumps(t)]


def google_history(raw: dict) -> list[tuple[str, float]]:
    """price_insights.price_history is [[unix_ts, price], ...] when Google provides it."""
    out = []
    for point in (raw.get("price_insights") or {}).get("price_history") or []:
        try:
            ts, price = point
            out.append((datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat(), float(price)))
        except (TypeError, ValueError):
            continue
    return out


def read_fares(trip: dict, provider) -> dict:
    """One live search; cheapest priced fare on the user's route and departure date."""
    max_stops = trip.get("max_stops")
    raw = provider.search(",".join(trip["origin_codes"]), ",".join(trip["destination_codes"]),
                          trip["depart_date"], trip["return_date"], max_stops)
    offers = [
        o for o in normalize_offers(raw, "t", trip["return_date"])
        if o["price"] is not None
        and (max_stops is None or o["stops"] <= max_stops)
        and o["depart_time"][:10] == trip["depart_date"]
        and o["depart_airport"] in trip["origin_codes"]
        and o["arrive_airport"] in trip["destination_codes"]
    ]
    fields = ("airlines", "flight_numbers", "depart_airport", "depart_time", "arrive_airport", "arrive_time", "stops")
    best = min(offers, key=lambda o: (o["price"], o["stops"])) if offers else None
    by_stops: dict[str, dict] = {}
    for o in sorted(offers, key=lambda o: o["price"]):
        by_stops.setdefault(str(o["stops"]), {"price": o["price"], "offer": {k: o[k] for k in fields}})
    insights = raw.get("price_insights") or {}
    return {
        "cheapest_price": best["price"] if best else None,
        "cheapest_offer": {k: best[k] for k in fields} if best else None,
        "by_stops": by_stops,
        "offer_count": len(offers),
        "google_typical_range": insights.get("typical_price_range"),
        "google_price_level": insights.get("price_level"),
        "google_history": google_history(raw),
    }


# --- email --------------------------------------------------------------------

def send_email(subject: str, body: str):
    sender = os.environ.get("GMAIL_ADDRESS")
    password = (os.environ.get("GMAIL_APP_PASSWORD") or "").replace(" ", "")  # Google shows it in groups of 4
    to = os.environ.get("ALERT_TO") or sender
    if not (sender and password):
        raise RuntimeError("GMAIL_ADDRESS and GMAIL_APP_PASSWORD must be set in .env")
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, sender, to
    msg.set_content(body)
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as smtp:
        smtp.login(sender, password)
        smtp.send_message(msg)


def tier_label(tier: str) -> str:
    return {"any": "", "0": "nonstop "}.get(tier, f"{tier}-stop ")


def pick_tier(trip: dict, reading: dict) -> str:
    """The most direct stop count with a fare today, for trips that prefer direct flights."""
    by_stops = reading.get("by_stops") or {}
    if (trip.get("prefer") == "fewest_stops" or trip.get("max_stops") is not None) and by_stops:
        return min(by_stops, key=int)
    return "any"


def format_alert(trip: dict, reading: dict, decision: dict, tier: str = "any") -> tuple[str, str]:
    o = reading["cheapest_offer"] if tier == "any" else reading["by_stops"][tier]["offer"]
    budget = trip.get("budget")
    within = "yes" if isinstance(budget, (int, float)) and decision["price"] <= budget else "no"
    label = tier_label(tier)
    subject = (f"Fare drop: {trip['id']} {label}${decision['price']:g} "
               f"({decision['pct_below_avg']:.0%} under avg)")
    body = "\n".join([
        f"Trip: {trip['id']} ({trip['origin']} -> {trip['destination']})",
        f"Dates: {trip['depart_date']} to {trip['return_date']}",
        "",
        f"Cheapest {label}round-trip fare today: ${decision['price']:g} per person",
        f"{WINDOW_DAYS}-day {label}average: ${decision['window_avg']:.0f}   "
        f"previous {WINDOW_DAYS}-day {label}low: ${decision['prior_low']:g}",
        f"Within your ${budget} budget: {within}" if budget is not None else "",
        "",
        f"Flight: {', '.join(o['airlines'])} {' / '.join(o['flight_numbers'])}, {o['stops']} stop(s)",
        f"Departs {o['depart_airport']} {o['depart_time']}, arrives {o['arrive_airport']} {o['arrive_time']}",
        f"Google's price level: {reading.get('google_price_level') or 'n/a'}; "
        f"typical range: {reading.get('google_typical_range') or 'n/a'}",
        "",
        "Nothing was booked. Prices change quickly; confirm on the airline or Google Flights.",
    ])
    return subject, body


# --- main -----------------------------------------------------------------------

def run(dry_run: bool = False, provider=None, today: str | None = None, history_path: Path = HISTORY) -> list[str]:
    today = today or date.today().isoformat()
    history = load_history(history_path)
    trips = tracked_trips()
    log = []
    if not trips:
        return ["no tracked trips (set track: true and fill in TODOs in evals/scenarios.yaml)"]
    for trip in trips:
        tid = trip["id"]
        if trip["depart_date"] <= today:
            log.append(f"{tid}: departure date has passed, skipping")
            continue
        existing = [r for r in history if r["trip_id"] == tid and r["date"] == today and r.get("type") == "reading"]
        if not existing:
            try:
                provider = provider or SerpApiProvider()
                reading = read_fares(trip, provider)
            except (SearchError, RuntimeError) as e:
                log.append(f"{tid}: search failed ({e}); nothing recorded")
                continue
            record = {"type": "reading", "trip_id": tid, "date": today, **reading}
            append_history(record, history_path)
            history.append(record)
        else:
            reading = existing[-1]

        tier = pick_tier(trip, reading)
        alerts = [r for r in history if r["trip_id"] == tid and r.get("type") == "alert"
                  and r.get("tier", "any") == tier]
        decision = evaluate(daily_series(history, tid, tier), today, alerts)
        log.append(f"{tid}: {tier_label(tier)}{decision['reason']}"
                   + (f" (${decision['price']:g})" if "price" in decision else ""))
        if decision["alert"] and not any(a["date"] == today for a in alerts):
            subject, body = format_alert(trip, reading, decision, tier)
            if dry_run:
                log.append(f"{tid}: [dry run] would email: {subject}")
                continue
            try:
                send_email(subject, body)
            except (OSError, smtplib.SMTPException, RuntimeError) as e:
                log.append(f"{tid}: EMAIL FAILED: {e}")
                continue
            alert = {"type": "alert", "trip_id": tid, "date": today, "tier": tier, "price": decision["price"]}
            append_history(alert, history_path)
            log.append(f"{tid}: emailed: {subject}")
    return log


def main(argv=None) -> int:
    load_dotenv(ROOT / ".env")
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--test-email", action="store_true")
    args = p.parse_args(argv)
    if args.test_email:
        send_email("Trip fare tracker: test email", "If you can read this, fare alerts will reach you.")
        print("test email sent")
        return 0
    log = run(dry_run=args.dry_run)
    print("\n".join(log))
    return 1 if any("FAILED" in line or "failed" in line for line in log) else 0


if __name__ == "__main__":
    sys.exit(main())
