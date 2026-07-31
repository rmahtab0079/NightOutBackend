"""
Launch-gate latency suite for Night Out restaurant + activity endpoints.

Goals
-----
1. Cache HIT responses for /night_out_suggestion and /night_out_events stay
   under CACHE_HIT_BUDGET_MS (default 50ms).
2. Cold (cache MISS) paths are timed and reported — they call Google Places /
   Ticketmaster so they are NOT held to 50ms.
3. Same cuisine / interest filters produce a stable first pick (shared cache).

Run (server must already be up on :8000, or set API_BASE_URL):

    make test-latency

Or in-process (no server — exercises the same cache):

    make test-latency-local
"""

from __future__ import annotations

import os
import statistics
import time
from typing import Any

import httpx
import pytest

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
CACHE_HIT_BUDGET_MS = float(os.getenv("CACHE_HIT_BUDGET_MS", "50"))
COLD_PATH_WARN_MS = float(os.getenv("COLD_PATH_WARN_MS", "3000"))
# NYC-ish coords used throughout recent debugging
LAT = float(os.getenv("TEST_LAT", "40.83826525489168"))
LON = float(os.getenv("TEST_LON", "-74.04045383281529"))
RADIUS_METERS = float(os.getenv("TEST_RADIUS_METERS", str(5 * 1609.34)))
RADIUS_MILES = float(os.getenv("TEST_RADIUS_MILES", "25"))

# Warmed once per session so subsequent tests can assert on hits.
_SESSION_WARMED: dict[str, Any] = {}


def _ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def _restaurant_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "party_size": 1,
        "latitude": LAT,
        "longitude": LON,
        "radius_meters": RADIUS_METERS,
        "dietary_preferences": [],
        "cuisines": ["Italian"],
        "interests": [],
        "excluded_names": [],
    }
    body.update(overrides)
    return body


def _events_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "party_size": 1,
        "latitude": LAT,
        "longitude": LON,
        "radius_miles": RADIUS_MILES,
        "interests": ["Live Music"],
        "excluded_event_ids": [],
    }
    body.update(overrides)
    return body


@pytest.fixture(scope="session")
def client() -> httpx.Client:
    with httpx.Client(base_url=API_BASE_URL, timeout=60.0) as c:
        # Fail fast if the server isn't up.
        try:
            r = c.get("/")
            assert r.status_code < 500
        except httpx.ConnectError as e:
            pytest.skip(
                f"API not reachable at {API_BASE_URL}. "
                f"Start it with `make run`, then re-run. ({e})"
            )
        yield c


def _post_timed(client: httpx.Client, path: str, body: dict) -> tuple[float, dict]:
    started = time.perf_counter()
    resp = client.post(path, json=body)
    elapsed = _ms(started)
    assert resp.status_code == 200, f"{path} -> {resp.status_code}: {resp.text[:300]}"
    data = resp.json()
    assert isinstance(data, dict)
    return elapsed, data


# ---------------------------------------------------------------------------
# Warm-up helpers
# ---------------------------------------------------------------------------
def _warm_restaurant(client: httpx.Client, body: dict) -> dict:
    key = f"rest:{sorted(body.get('cuisines') or [])}:{sorted(body.get('dietary_preferences') or [])}"
    if key in _SESSION_WARMED:
        return _SESSION_WARMED[key]
    elapsed, data = _post_timed(client, "/night_out_suggestion", body)
    print(
        f"\n[warm restaurant] {elapsed:.1f}ms cache_hit={data.get('cache_hit')} "
        f"pool={data.get('pool_size')} name={data.get('name')!r}"
    )
    if elapsed > COLD_PATH_WARN_MS:
        print(f"  WARN: cold path {elapsed:.1f}ms > {COLD_PATH_WARN_MS:.0f}ms budget")
    _SESSION_WARMED[key] = data
    return data


def _warm_events(client: httpx.Client, body: dict) -> dict:
    key = f"evt:{sorted(body.get('interests') or [])}:{body.get('classification')}"
    if key in _SESSION_WARMED:
        return _SESSION_WARMED[key]
    elapsed, data = _post_timed(client, "/night_out_events", body)
    print(
        f"\n[warm events] {elapsed:.1f}ms cache_hit={data.get('cache_hit')} "
        f"pool={data.get('pool_size')} name={data.get('name')!r}"
    )
    if elapsed > COLD_PATH_WARN_MS:
        print(f"  WARN: cold path {elapsed:.1f}ms > {COLD_PATH_WARN_MS:.0f}ms budget")
    _SESSION_WARMED[key] = data
    return data


# ---------------------------------------------------------------------------
# Restaurant latency
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "label,overrides",
    [
        ("italian", {"cuisines": ["Italian"]}),
        ("chinese", {"cuisines": ["Chinese"]}),
        ("halal", {"cuisines": [], "dietary_preferences": ["Halal"]}),
        (
            "multi_cuisine",
            {"cuisines": ["Italian", "Chinese", "Japanese"]},
        ),
    ],
)
def test_restaurant_cache_hit_under_budget(
    client: httpx.Client, label: str, overrides: dict
):
    body = _restaurant_body(**overrides)
    _warm_restaurant(client, body)

    samples: list[float] = []
    names: list[str] = []
    for _ in range(5):
        elapsed, data = _post_timed(client, "/night_out_suggestion", body)
        samples.append(elapsed)
        names.append(data.get("name") or "")
        assert data.get("cache_hit") is True, f"{label}: expected cache_hit after warm"
        assert (data.get("pool_size") or 0) >= 5, f"{label}: thin pool {data.get('pool_size')}"

    p50 = statistics.median(samples)
    p95 = sorted(samples)[max(0, int(len(samples) * 0.95) - 1)]
    print(
        f"\n[restaurant/{label}] hit samples_ms={[round(s, 2) for s in samples]} "
        f"p50={p50:.2f} p95={p95:.2f} budget={CACHE_HIT_BUDGET_MS}"
    )
    assert p95 <= CACHE_HIT_BUDGET_MS, (
        f"{label} cache-hit p95 {p95:.1f}ms exceeds {CACHE_HIT_BUDGET_MS}ms budget"
    )
    # Same filters → same first pick for every user that day.
    assert len(set(names)) == 1, f"{label}: unstable pick across hits: {names}"


def test_restaurant_next_uses_cache(client: httpx.Client):
    body = _restaurant_body(cuisines=["Italian"])
    first = _warm_restaurant(client, body)
    body_next = dict(body)
    body_next["excluded_names"] = [first["name"]]

    elapsed, second = _post_timed(client, "/night_out_suggestion", body_next)
    print(
        f"\n[restaurant/next] {elapsed:.1f}ms "
        f"first={first['name']!r} next={second['name']!r} hit={second.get('cache_hit')}"
    )
    assert second.get("cache_hit") is True
    assert second["name"] != first["name"]
    assert elapsed <= CACHE_HIT_BUDGET_MS


# ---------------------------------------------------------------------------
# Activity latency
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "label,overrides",
    [
        ("live_music", {"interests": ["Live Music"]}),
        ("hiking", {"interests": ["Hiking"]}),
        ("museums", {"interests": ["Museums"]}),
        (
            "multi_interest",
            {"interests": ["Live Music", "Comedy Shows", "Theater"]},
        ),
    ],
)
def test_events_cache_hit_under_budget(
    client: httpx.Client, label: str, overrides: dict
):
    body = _events_body(**overrides)
    _warm_events(client, body)

    samples: list[float] = []
    names: list[str] = []
    for _ in range(5):
        elapsed, data = _post_timed(client, "/night_out_events", body)
        samples.append(elapsed)
        names.append(data.get("name") or "")
        assert data.get("cache_hit") is True, f"{label}: expected cache_hit after warm"
        assert (data.get("pool_size") or 0) >= 5, f"{label}: thin pool {data.get('pool_size')}"

    p50 = statistics.median(samples)
    p95 = sorted(samples)[max(0, int(len(samples) * 0.95) - 1)]
    print(
        f"\n[events/{label}] hit samples_ms={[round(s, 2) for s in samples]} "
        f"p50={p50:.2f} p95={p95:.2f} budget={CACHE_HIT_BUDGET_MS}"
    )
    assert p95 <= CACHE_HIT_BUDGET_MS, (
        f"{label} cache-hit p95 {p95:.1f}ms exceeds {CACHE_HIT_BUDGET_MS}ms budget"
    )
    assert len(set(names)) == 1, f"{label}: unstable pick across hits: {names}"


# ---------------------------------------------------------------------------
# Cold-path report (informational — does not fail on >50ms)
# ---------------------------------------------------------------------------
def test_cold_path_timing_report(client: httpx.Client):
    """Force unique cuisine keys so we measure real Places latency."""
    # Use an uncommon combo unlikely to be warm from earlier tests.
    body = _restaurant_body(
        cuisines=["Vietnamese", "Caribbean"],
        dietary_preferences=[],
    )
    # Bust by slightly nudging lon into a new cache bucket if needed —
    # Vietnamese+Caribbean is usually cold on a fresh server.
    elapsed, data = _post_timed(client, "/night_out_suggestion", body)
    print(
        f"\n[cold report] restaurant multi={elapsed:.1f}ms "
        f"cache_hit={data.get('cache_hit')} pool={data.get('pool_size')} "
        f"name={data.get('name')!r}"
    )
    assert (data.get("pool_size") or 0) >= 1
    # Soft budget only — print warn, don't fail the launch gate on cold path.
    if elapsed > COLD_PATH_WARN_MS:
        print(
            f"  WARN: cold restaurant path {elapsed:.1f}ms "
            f"(warn budget {COLD_PATH_WARN_MS:.0f}ms)"
        )
