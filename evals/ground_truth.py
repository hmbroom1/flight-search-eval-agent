"""Ground truth, computed straight from the raw API responses.

This is written independently of src/trip_agent/eligibility.py on purpose. If
the eval reused the agent's own check_eligibility, a bug in that function
(say, `<` instead of `<=`) would be invisible: the tool and the grader would
agree with each other and both be wrong. The boundary-budget scenarios exist
to catch exactly that.

Truth encodes the *user's* intent from the scenario (their airports, dates,
budget), not whatever arguments the agent happened to pass to its tools.

Offers are identified by content (flight numbers + departure time + price), not
by offer_id, so the same flight found by two different searches counts once.

The harness's reference search exists to catch routes the agent never searched.
It is NOT a complete list of flights: Google returns a different selection of
itineraries for a different query, so a reference-only offer on a route the agent
DID search is not something the agent could have found. Those are skipped (pass
`covered`, the (origin, destination) pairs the agent's successful searches spanned).
This was found on the first real honeymoon run, where it produced false "misses".
"""

from __future__ import annotations


def flight_identity(flight_numbers, depart_time, price) -> tuple:
    return (tuple(fn.replace(" ", "").upper() for fn in flight_numbers), depart_time, price)


def _items(raw: dict) -> list[dict]:
    return list(raw.get("best_flights") or []) + list(raw.get("other_flights") or [])


def covered_routes(searches: list[dict]) -> set[tuple[str, str]]:
    """(origin, destination) airport pairs spanned by the agent's successful searches."""
    return {(o, d) for s in searches if s.get("ok")
            for o in s["origin"].split(",") for d in s["destination"].split(",")}


def compute_truth(searches: list[dict], scenario: dict, covered: set | None = None) -> dict:
    """Return {'eligible': {identity: info}, 'best': set(identity), 'cheapest_seen': price|None}."""
    budget = scenario["budget"]
    max_stops = scenario.get("max_stops")
    depart, ret = scenario["depart_date"], scenario["return_date"]
    origins = set(scenario["origin_codes"])
    dests = set(scenario["destination_codes"])

    eligible: dict[tuple, dict] = {}
    cheapest = None
    for s in searches:
        if not s.get("ok") or s.get("return_date") != ret:
            continue
        for idx, item in enumerate(_items(s["raw"])):
            segs = item.get("flights") or []
            price = item.get("price")
            if not segs or not isinstance(price, (int, float)):
                continue
            first_dep = segs[0]["departure_airport"]
            last_arr = segs[-1]["arrival_airport"]
            if s.get("search_id") == "ref" and covered is not None and (first_dep["id"], last_arr["id"]) in covered:
                continue
            on_route = first_dep["id"] in origins and last_arr["id"] in dests
            on_date = first_dep["time"].split(" ")[0] == depart
            if on_route and on_date:
                cheapest = price if cheapest is None else min(cheapest, price)
            stops_ok = max_stops is None or len(item.get("layovers") or []) <= max_stops
            if on_route and on_date and stops_ok and price <= budget:
                ident = flight_identity([x["flight_number"] for x in segs], first_dep["time"], price)
                eligible.setdefault(ident, {
                    "offer_id": f"{s['search_id']}-o{idx}",
                    "price": price,
                    "stops": len(item.get("layovers") or []),
                    "duration": item.get("total_duration") or 10**9,
                    "depart_time": first_dep["time"],
                })

    best: set[tuple] = set()
    if eligible:
        prefer = scenario.get("prefer", "price")

        def key(i):
            first = (i["stops"], i["price"]) if prefer == "fewest_stops" else (i["price"], i["stops"])
            return (*first, i["duration"], i["depart_time"])
        top = min(key(v) for v in eligible.values())
        best = {ident for ident, v in eligible.items() if key(v) == top}
    return {"eligible": eligible, "best": best, "cheapest_seen": cheapest}


def cheapest_price(raw: dict, scenario: dict) -> float | None:
    """Cheapest on-route, on-date fare in one response (used to set boundary budgets)."""
    fake = dict(scenario, budget=float("inf"))
    search = {"ok": True, "raw": raw, "search_id": "pre", "return_date": scenario["return_date"]}
    return compute_truth([search], fake)["cheapest_seen"]
