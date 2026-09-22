"""Flight data providers.

Every provider returns the *raw* SerpApi Google Flights response shape. The raw
response is kept on the run context so the eval can compute ground truth and
check grounding against the source data, not against the agent's view of it.

- SerpApiProvider: live Google Flights data via SerpApi.
- CachedProvider:  record/replay wrapper so eval runs are reproducible and don't
                   burn free-tier searches.
- FixtureProvider: synthetic SerpApi-shaped JSON for edge-case scenarios.
- FaultProvider:   injects API errors / rate limits to test graceful degradation.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Protocol

import requests

SERPAPI_URL = "https://serpapi.com/search.json"


class SearchError(Exception):
    """A failed search. `kind` is one of: rate_limit, api_error, timeout."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


class FlightProvider(Protocol):
    name: str

    def search(self, origin: str, destination: str, depart_date: str, return_date: str,
               max_stops: int | None = None) -> dict: ...


def _codes(value: str) -> set[str]:
    return {c.strip().upper() for c in value.split(",") if c.strip()}


def serpapi_stops(max_stops: int | None) -> str:
    """SerpApi `stops`: 0 any, 1 nonstop only, 2 one stop or fewer, 3 two stops or fewer."""
    return "0" if max_stops is None else str(min(max_stops, 2) + 1)


class SerpApiProvider:
    name = "serpapi"

    def __init__(self, api_key: str | None = None, timeout: float = 30.0):
        self.api_key = api_key or os.environ.get("SERPAPI_API_KEY")
        if not self.api_key:
            raise RuntimeError("SERPAPI_API_KEY is not set (see .env.example)")
        self.timeout = timeout

    def search(self, origin, destination, depart_date, return_date, max_stops=None):
        params = {
            "engine": "google_flights",
            "departure_id": origin,
            "arrival_id": destination,
            "outbound_date": depart_date,
            "return_date": return_date,
            "type": "1",  # round trip; prices are round-trip totals per adult
            "adults": "1",
            "currency": "USD",
            "hl": "en",
            "gl": "us",
            "stops": serpapi_stops(max_stops),
            "api_key": self.api_key,
        }
        try:
            resp = requests.get(SERPAPI_URL, params=params, timeout=self.timeout)
        except requests.Timeout as e:
            raise SearchError("timeout", f"SerpApi request timed out: {e}") from None
        except requests.RequestException as e:
            raise SearchError("api_error", f"SerpApi request failed: {type(e).__name__}") from None

        if resp.status_code == 429:
            raise SearchError("rate_limit", "SerpApi rate limit or monthly quota exceeded (HTTP 429)")
        try:
            raw = resp.json()
        except ValueError:
            raise SearchError("api_error", f"SerpApi returned non-JSON (HTTP {resp.status_code})") from None

        error = raw.get("error")
        if error:
            # SerpApi reports "no results" as an error string; that is an empty
            # result set, not a failure, and the agent should say "none found".
            if "hasn't returned any results" in error or "no results" in error.lower():
                raw.setdefault("best_flights", [])
                raw.setdefault("other_flights", [])
                return _scrub(raw, self.api_key)
            raise SearchError("api_error", f"SerpApi error: {error}")
        if resp.status_code >= 400:
            raise SearchError("api_error", f"SerpApi HTTP {resp.status_code}")
        return _scrub(raw, self.api_key)


def _scrub(raw: dict, secret: str) -> dict:
    """Remove anything that could echo the API key before the response is stored."""
    text = json.dumps(raw)
    if secret and secret in text:
        text = text.replace(secret, "REDACTED")
    raw = json.loads(text)
    meta = raw.get("search_metadata")
    if isinstance(meta, dict):
        for k in list(meta):
            if k.endswith("_url") or k.endswith("_file") or k == "json_endpoint":
                meta.pop(k)
    return raw


class CachedProvider:
    """Record/replay wrapper.

    mode="record": use the cached response if present, otherwise call live and save.
    mode="replay": cache only; a miss is an error (fully offline, reproducible).
    mode="live":   always call live and overwrite the cache.
    """

    def __init__(self, inner: FlightProvider | None, cache_dir: Path, mode: str = "record"):
        if mode not in {"record", "replay", "live"}:
            raise ValueError(f"unknown cache mode {mode!r}")
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.mode = mode
        self.name = f"cached({inner.name if inner else 'none'},{mode})"

    def _path(self, *args: str) -> Path:
        key = hashlib.sha256("|".join(a.upper() for a in args).encode()).hexdigest()[:16]
        return self.cache_dir / f"{key}.json"

    def search(self, origin, destination, depart_date, return_date, max_stops=None):
        key = [origin, destination, depart_date, return_date]
        if max_stops is not None:  # keeps cache keys for unfiltered searches unchanged
            key.append(f"stops<={max_stops}")
        path = self._path(*key)
        if self.mode != "live" and path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        if self.mode == "replay":
            raise SearchError("api_error", f"replay mode: no cached response for {origin}->{destination} {depart_date}/{return_date}")
        if self.inner is None:
            raise SearchError("api_error", "no live provider configured")
        raw = self.inner.search(origin, destination, depart_date, return_date, max_stops)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        return raw


class FixtureProvider:
    """Serves a synthetic SerpApi-shaped response from disk.

    Offers are filtered by the requested airport codes so that a multi-airport
    scenario behaves like the real API: search only JFK and you only get JFK.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.name = f"fixture({self.path.stem})"

    def search(self, origin, destination, depart_date, return_date, max_stops=None):
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        origins, dests = _codes(origin), _codes(destination)

        def keep(item: dict) -> bool:
            segs = item.get("flights") or []
            if not segs:
                return True
            if max_stops is not None and len(item.get("layovers") or []) > max_stops:
                return False
            return (segs[0]["departure_airport"]["id"] in origins
                    and segs[-1]["arrival_airport"]["id"] in dests)

        for key in ("best_flights", "other_flights"):
            raw[key] = [it for it in raw.get(key, []) if keep(it)]
        raw["search_parameters"] = {
            "engine": "google_flights", "departure_id": origin, "arrival_id": destination,
            "outbound_date": depart_date, "return_date": return_date, "currency": "USD",
        }
        return raw


class FaultProvider:
    """Fails the first `fail_times` calls (None = always), then delegates."""

    def __init__(self, kind: str, inner: FlightProvider | None = None, fail_times: int | None = None):
        self.kind = kind
        self.inner = inner
        self.fail_times = fail_times
        self.calls = 0
        self.name = f"fault({kind})"

    def search(self, origin, destination, depart_date, return_date, max_stops=None):
        self.calls += 1
        if self.fail_times is None or self.calls <= self.fail_times:
            messages = {
                "rate_limit": "SerpApi rate limit or monthly quota exceeded (HTTP 429)",
                "api_error": "SerpApi error: upstream service unavailable (HTTP 503)",
                "timeout": "SerpApi request timed out",
            }
            raise SearchError(self.kind, messages.get(self.kind, "search failed"))
        return self.inner.search(origin, destination, depart_date, return_date, max_stops)
