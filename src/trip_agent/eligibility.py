"""Deterministic eligibility rules and the documented tie-break.

No LLM here on purpose: "is this price <= budget and on these dates" is
objectively checkable, so a rule is more accurate and cheaper than a judge.

Rules (all must hold):
  1. The offer has a price.
  2. price <= budget. The budget is INCLUSIVE, in USD, round-trip, per person.
  3. The first outbound segment departs on the requested departure date.
  4. The offer was searched with the requested return date.

Ranking among eligible offers (documented so "correct choice" is gradeable):
  prefer="price" (default):  lowest price -> fewest stops -> shortest duration -> earliest departure
  prefer="fewest_stops":     fewest stops -> lowest price -> shortest duration -> earliest departure
`prefer` is a ranking preference: a 1-stop fare within budget is still eligible,
it just ranks below any eligible nonstop. `max_stops` is different: a HARD
constraint (e.g. max_stops=0 = "nonstop only"); anything with more stops fails
eligibility, exactly like a fare over budget.
"""

from __future__ import annotations

from datetime import date


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def load_scenarios(path) -> list[dict]:
    """Read scenarios.yaml; unquoted YAML dates become ISO strings so comparisons work."""
    import yaml
    from pathlib import Path

    scenarios = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["scenarios"]
    for sc in scenarios:
        for key in ("depart_date", "return_date"):
            if isinstance(sc.get(key), date):
                sc[key] = sc[key].isoformat()
    return scenarios


def validate_date_range(depart_date: str, return_date: str, today: date | None = None) -> str | None:
    """Return an error message, or None if the range is usable."""
    try:
        d, r = parse_date(depart_date), parse_date(return_date)
    except (TypeError, ValueError):
        return f"Dates must be YYYY-MM-DD; got depart_date={depart_date!r}, return_date={return_date!r}"
    if r < d:
        return f"return_date {return_date} is before depart_date {depart_date}"
    if today is not None and d < today:
        return f"depart_date {depart_date} is in the past (today is {today.isoformat()})"
    return None


def check_offer(offer: dict, budget: float, depart_date: str, return_date: str,
                max_stops: int | None = None) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if max_stops is not None and offer.get("stops", 99) > max_stops:
        reasons.append(f"{offer.get('stops')} stop(s), user allows at most {max_stops}")
    price = offer.get("price")
    if price is None:
        reasons.append("no price listed")
    elif price > budget:
        reasons.append(f"price ${price} exceeds budget ${budget:g}")
    depart_time = offer.get("depart_time") or ""
    if depart_time[:10] != depart_date:
        reasons.append(f"departs {depart_time[:10] or 'unknown'}, requested {depart_date}")
    if offer.get("return_date") != return_date:
        reasons.append(f"searched with return {offer.get('return_date')}, requested {return_date}")
    return (not reasons), reasons


def rank_key(offer: dict, prefer: str = "price") -> tuple:
    price, stops = offer["price"], offer.get("stops", 99)
    first = (stops, price) if prefer == "fewest_stops" else (price, stops)
    return (*first, offer.get("total_duration_min") or 10**9, offer.get("depart_time") or "")


def pick_best(offers: list[dict], prefer: str = "price") -> dict | None:
    return min(offers, key=lambda o: rank_key(o, prefer)) if offers else None
