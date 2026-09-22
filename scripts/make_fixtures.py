"""Generate synthetic SerpApi-shaped fixtures for the edge-case scenarios.

These are NOT real fares. They exist so edge cases (ties, boundary budgets,
wrong-date traps, missing prices, multi-airport cities, empty results) are
deterministic and free to run.
"""
import json
from pathlib import Path

AIRPORTS = {
    "AUS": "Austin-Bergstrom International Airport", "MIA": "Miami International Airport",
    "IAH": "George Bush Intercontinental Airport", "ATL": "Hartsfield-Jackson Atlanta International Airport",
    "DEN": "Denver International Airport", "JFK": "John F. Kennedy International Airport",
    "LGA": "LaGuardia Airport", "EWR": "Newark Liberty International Airport",
}
AIRLINE = {"AA": "American", "UA": "United", "DL": "Delta", "B6": "JetBlue", "F9": "Frontier",
           "AS": "Alaska", "WN": "Southwest"}


def seg(fn, frm, to, dep, arr, dur):
    return {
        "departure_airport": {"name": AIRPORTS[frm], "id": frm, "time": dep},
        "arrival_airport": {"name": AIRPORTS[to], "id": to, "time": arr},
        "duration": dur, "airplane": "Airbus A321", "airline": AIRLINE[fn[:2]],
        "travel_class": "Economy", "flight_number": fn, "legroom": "30 in", "extensions": [],
    }


def offer(price, segs, layovers=(), total=None):
    item = {"flights": segs, "layovers": [{"duration": d, "name": AIRPORTS[c], "id": c} for c, d in layovers],
            "total_duration": total or sum(s["duration"] for s in segs) + sum(d for _, d in layovers),
            "type": "Round trip"}
    if price is not None:
        item["price"] = price
    return item


D = "2027-03-12"
standard = {
    "_synthetic": True,
    "search_metadata": {"status": "Success"},
    "best_flights": [
        offer(289, [seg("AA 1234", "AUS", "MIA", f"{D} 07:05", f"{D} 11:02", 177)]),
        offer(289, [seg("UA 455", "AUS", "IAH", f"{D} 06:00", f"{D} 06:55", 55),
                    seg("UA 1790", "IAH", "MIA", f"{D} 08:10", f"{D} 12:40", 150)], [("IAH", 75)]),
        offer(312, [seg("DL 2210", "AUS", "ATL", f"{D} 08:15", f"{D} 11:30", 135),
                    seg("DL 1488", "ATL", "MIA", f"{D} 12:40", f"{D} 14:30", 110)], [("ATL", 70)]),
    ],
    "other_flights": [
        offer(405, [seg("B6 918", "AUS", "MIA", f"{D} 13:20", f"{D} 17:10", 170)]),
        # Trap: cheapest fare in the response, but departs the day AFTER the requested date.
        offer(199, [seg("F9 1402", "AUS", "DEN", "2027-03-13 00:35", "2027-03-13 01:50", 135),
                    seg("F9 2203", "DEN", "MIA", "2027-03-13 06:00", "2027-03-13 11:45", 225)], [("DEN", 250)]),
        # Trap: no price listed.
        offer(None, [seg("AS 88", "AUS", "MIA", f"{D} 10:00", f"{D} 13:55", 175)]),
        offer(540, [seg("WN 3321", "AUS", "MIA", f"{D} 15:45", f"{D} 19:30", 165)]),
    ],
}

M = "2027-04-09"
multi = {
    "_synthetic": True,
    "search_metadata": {"status": "Success"},
    "best_flights": [
        offer(248, [seg("B6 1284", "AUS", "JFK", f"{M} 06:30", f"{M} 11:05", 215)]),
        offer(231, [seg("UA 2150", "AUS", "EWR", f"{M} 09:10", f"{M} 13:48", 218)]),
    ],
    "other_flights": [
        offer(262, [seg("DL 1167", "AUS", "ATL", f"{M} 07:00", f"{M} 10:10", 130),
                    seg("DL 2544", "ATL", "LGA", f"{M} 11:15", f"{M} 13:30", 135)], [("ATL", 65)]),
        offer(305, [seg("AA 2980", "AUS", "LGA", f"{M} 12:05", f"{M} 16:40", 215)]),
    ],
}

# Cheapest fares connect; nonstops cost more. Price-first picks the $210 1-stop,
# fewest-stops-first picks the $265 nonstop (budget 300). At budget 250 no nonstop
# fits, so the right answer falls back to the $210 1-stop.
tradeoff = {
    "_synthetic": True,
    "search_metadata": {"status": "Success"},
    "best_flights": [
        offer(210, [seg("UA 455", "AUS", "IAH", f"{D} 06:00", f"{D} 06:55", 55),
                    seg("UA 1790", "IAH", "MIA", f"{D} 08:10", f"{D} 12:40", 150)], [("IAH", 75)]),
        offer(265, [seg("AA 1234", "AUS", "MIA", f"{D} 07:05", f"{D} 11:02", 177)]),
    ],
    "other_flights": [
        offer(280, [seg("B6 918", "AUS", "MIA", f"{D} 13:20", f"{D} 17:10", 170)]),
        offer(228, [seg("DL 2210", "AUS", "ATL", f"{D} 08:15", f"{D} 11:30", 135),
                    seg("DL 1488", "ATL", "MIA", f"{D} 12:40", f"{D} 14:30", 110)], [("ATL", 70)]),
    ],
}

empty = {"_synthetic": True, "search_metadata": {"status": "Success"},
         "error": "Google Flights hasn't returned any results for this query.",
         "best_flights": [], "other_flights": []}

out = Path(__file__).resolve().parent.parent / "evals" / "fixtures"
for name, data in {"standard": standard, "multi_airport": multi, "nonstop_tradeoff": tradeoff,
                   "empty": empty}.items():
    (out / f"{name}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    print("wrote", name)
