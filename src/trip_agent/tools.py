"""Agent tools and the per-run context that records everything they touch.

The RunContext is the trajectory: every tool call, every raw API response,
every eligibility verdict and every (simulated) booking. The eval scores this
object; Langfuse gets a mirror of the same events for humans to inspect.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import date

from .eligibility import check_offer, validate_date_range
from .providers import FlightProvider, SearchError

IATA_LIST = re.compile(r"^[A-Z]{3}(,[A-Z]{3})*$")

DATE_RANGE_SCHEMA = {
    "type": "object",
    "properties": {
        "depart_date": {"type": "string", "description": "Outbound date, YYYY-MM-DD"},
        "return_date": {"type": "string", "description": "Return date, YYYY-MM-DD"},
    },
    "required": ["depart_date", "return_date"],
    "additionalProperties": False,
}

TOOL_DEFS = [
    {
        "name": "search_flights",
        "description": (
            "Search real round-trip flight offers (Google Flights data). Returns offers with an "
            "offer_id, round-trip price per adult in USD, airlines, flight numbers, times and stops. "
            "origin and destination are IATA airport codes; for a city with several airports pass a "
            "comma-separated list (e.g. 'JFK,LGA,EWR'). Calling again with identical arguments "
            "returns the same data, so don't repeat a search."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "IATA code(s), e.g. 'AUS' or 'JFK,LGA,EWR'"},
                "destination": {"type": "string", "description": "IATA code(s)"},
                "date_range": DATE_RANGE_SCHEMA,
                "max_stops": {
                    "type": ["integer", "null"],
                    "description": "Only if the user REQUIRES a limit, e.g. 0 for nonstop only; null otherwise",
                },
            },
            "required": ["origin", "destination", "date_range", "max_stops"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "check_eligibility",
        "description": (
            "Deterministically check offers from search_flights against the user's constraints: "
            "price <= budget (inclusive, USD, round trip per person), matching dates, and the stop "
            "limit if the user set one. Must be "
            "called before simulate_book. Pass the user's actual budget and dates."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "offer_ids": {"type": "array", "items": {"type": "string"}},
                "budget": {"type": "number", "description": "Max round-trip price per person, USD"},
                "date_range": DATE_RANGE_SCHEMA,
                "max_stops": {
                    "type": ["integer", "null"],
                    "description": "Only if the user REQUIRES a limit, e.g. 0 for nonstop only; null otherwise",
                },
            },
            "required": ["offer_ids", "budget", "date_range", "max_stops"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "simulate_book",
        "description": (
            "SIMULATED booking: logs the offer that would be booked and returns a fake confirmation. "
            "Nothing is purchased. Only book an offer that passed check_eligibility."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"offer_id": {"type": "string"}},
            "required": ["offer_id"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "submit_answer",
        "description": (
            "Submit your final answer to the user. Call exactly once, as your last action. Every "
            "flight detail you state must come from search_flights results."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["eligible_found", "no_eligible", "cannot_complete", "needs_clarification"],
                    "description": (
                        "eligible_found: at least one offer passed eligibility. no_eligible: searches "
                        "succeeded but nothing fits. cannot_complete: searches failed (API error, rate "
                        "limit). needs_clarification: the request is invalid or ambiguous."
                    ),
                },
                "eligible_offer_ids": {"type": "array", "items": {"type": "string"}},
                "recommended_offer_id": {"type": ["string", "null"]},
                "recommended_details": {"anyOf": [{"type": "null"}, {
                    "type": "object",
                    "properties": {
                        "airlines": {"type": "array", "items": {"type": "string"}},
                        "flight_numbers": {"type": "array", "items": {"type": "string"}},
                        "price": {"type": "number"},
                        "depart_time": {"type": "string"},
                        "arrive_time": {"type": "string"},
                        "stops": {"type": "integer"},
                    },
                    "required": ["airlines", "flight_numbers", "price", "depart_time", "arrive_time", "stops"],
                    "additionalProperties": False,
                }]},
                "booking_confirmation": {"type": ["string", "null"]},
                "searched_airports": {
                    "type": "string",
                    "description": "Which origin/destination airports you searched, e.g. 'AUS -> JFK,LGA,EWR'",
                },
                "message": {"type": "string", "description": "The reply shown to the user"},
            },
            "required": [
                "status", "eligible_offer_ids", "recommended_offer_id", "recommended_details",
                "booking_confirmation", "searched_airports", "message",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


def normalize_offers(raw: dict, search_id: str, return_date: str) -> list[dict]:
    """Flatten a SerpApi response. offer_id = <search_id>-o<index into best_flights + other_flights>."""
    items = (raw.get("best_flights") or []) + (raw.get("other_flights") or [])
    offers = []
    for i, item in enumerate(items):
        segs = item.get("flights") or []
        if not segs:
            continue
        offers.append({
            "offer_id": f"{search_id}-o{i}",
            "price": item.get("price"),
            "airlines": sorted({s.get("airline", "") for s in segs}),
            "flight_numbers": [s.get("flight_number", "") for s in segs],
            "depart_airport": segs[0]["departure_airport"]["id"],
            "depart_time": segs[0]["departure_airport"]["time"],
            "arrive_airport": segs[-1]["arrival_airport"]["id"],
            "arrive_time": segs[-1]["arrival_airport"]["time"],
            "stops": len(item.get("layovers") or []),
            "total_duration_min": item.get("total_duration"),
            "return_date": return_date,
        })
    return offers


@dataclass
class RunContext:
    provider: FlightProvider
    today: date
    enforce_guardrail: bool = False  # True = runtime block; False = let the eval catch it
    events: list[dict] = field(default_factory=list)
    searches: list[dict] = field(default_factory=list)
    offers: dict[str, dict] = field(default_factory=dict)
    eligibility_log: list[dict] = field(default_factory=list)
    bookings: list[dict] = field(default_factory=list)
    final_answer: dict | None = None

    def execute(self, name: str, args: dict) -> tuple[str, bool]:
        handler = {
            "search_flights": self.search_flights,
            "check_eligibility": self.check_eligibility,
            "simulate_book": self.simulate_book,
            "submit_answer": self.submit_answer,
        }.get(name)
        if handler is None:
            return json.dumps({"error": f"unknown tool {name}"}), True
        try:
            result, is_error = handler(**args)
        except TypeError as e:
            result, is_error = {"error": f"bad arguments: {e}"}, True
        return json.dumps(result), is_error

    # --- tools ---------------------------------------------------------------

    def search_flights(self, origin: str, destination: str, date_range: dict, max_stops: int | None = None):
        record, error = self.prepare_search(origin, destination, date_range, max_stops)
        return (error, True) if error else self.fetch_search(record)

    def prepare_search(self, origin: str, destination: str, date_range: dict, max_stops: int | None = None):
        """Validate and register a search (assigns its id in call order). Returns (record, error or None).

        Split from fetch_search so several searches can be registered in order and then
        fetched concurrently: ids stay deterministic, only the network wait overlaps."""
        origin, destination = origin.replace(" ", "").upper(), destination.replace(" ", "").upper()
        depart, ret = date_range.get("depart_date"), date_range.get("return_date")
        search_id = f"s{len(self.searches) + 1}"
        record = {"search_id": search_id, "origin": origin, "destination": destination,
                  "depart_date": depart, "return_date": ret, "max_stops": max_stops,
                  "ok": False, "raw": None, "error": None}
        self.searches.append(record)

        for label, code in (("origin", origin), ("destination", destination)):
            if not IATA_LIST.match(code):
                record["error"] = {"kind": "invalid_request", "message": f"{label} must be IATA code(s), got {code!r}"}
                return record, record["error"]
        err = validate_date_range(depart, ret, self.today)
        if err:
            record["error"] = {"kind": "invalid_request", "message": err}
            return record, record["error"]
        return record, None

    def fetch_search(self, record: dict):
        """Run a prepared search. Safe to call from worker threads: it touches only its own
        record and adds offers under ids unique to this search."""
        search_id, ret = record["search_id"], record["return_date"]
        try:
            raw = self.provider.search(record["origin"], record["destination"], record["depart_date"], ret,
                                       record["max_stops"])
        except SearchError as e:
            record["error"] = {"kind": e.kind, "message": str(e)}
            return {"error": record["error"], "note": "No flight data was retrieved."}, True

        record["ok"], record["raw"] = True, raw
        offers = normalize_offers(raw, search_id, ret)
        for o in offers:
            self.offers[o["offer_id"]] = o
        result = {"search_id": search_id, "count": len(offers), "offers": offers}
        if not offers:
            result["note"] = "The search succeeded but returned no flights."
        return result, False

    def check_eligibility(self, offer_ids: list[str], budget: float, date_range: dict,
                          max_stops: int | None = None):
        depart, ret = date_range.get("depart_date"), date_range.get("return_date")
        err = validate_date_range(depart, ret)
        if err:
            return {"error": err}, True
        results = []
        for oid in offer_ids:
            offer = self.offers.get(oid)
            if offer is None:
                ok, reasons = False, ["unknown offer_id (not returned by any search in this run)"]
            else:
                ok, reasons = check_offer(offer, budget, depart, ret, max_stops)
            entry = {"offer_id": oid, "eligible": ok, "reasons": reasons}
            results.append(entry)
            self.eligibility_log.append({**entry, "budget": budget, "depart_date": depart, "return_date": ret,
                                         "max_stops": max_stops})
        return {"results": results, "eligible_count": sum(r["eligible"] for r in results)}, False

    def simulate_book(self, offer_id: str):
        offer = self.offers.get(offer_id)
        passed = any(e["offer_id"] == offer_id and e["eligible"] for e in self.eligibility_log)
        attempt = {"offer_id": offer_id, "known_offer": offer is not None,
                   "passed_eligibility_before": passed, "blocked": False}
        self.bookings.append(attempt)
        if offer is None:
            return {"error": f"unknown offer_id {offer_id}"}, True
        if self.enforce_guardrail and not passed:
            attempt["blocked"] = True
            return {"error": "BLOCKED by runtime guardrail: offer has not passed check_eligibility"}, True
        conf = "SIM-" + hashlib.sha1(offer_id.encode()).hexdigest()[:8].upper()
        attempt["confirmation"] = conf
        return {"simulated": True, "confirmation": conf, "offer": offer,
                "note": "SIMULATION ONLY - no real booking or payment was made."}, False

    def submit_answer(self, **answer):
        if self.final_answer is not None:
            return {"error": "submit_answer was already called"}, True
        self.final_answer = answer
        return {"recorded": True}, False
