"""
In-process launch-gate latency tests (no HTTP server required).

Exercises the same place/activity caches used by the FastAPI handlers.
Requires GOOGLE_PLACES_API_KEY (and Ticketmaster for events) via .env.

    make test-latency-local
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

CACHE_HIT_BUDGET_MS = float(os.getenv("CACHE_HIT_BUDGET_MS", "50"))
LAT = 40.83826525489168
LON = -74.04045383281529


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture(scope="module")
def app_module():
    if not os.getenv("GOOGLE_PLACES_API_KEY"):
        pytest.skip("GOOGLE_PLACES_API_KEY not set")
    import application as app

    with app._PLACE_POOL_LOCK:
        app._PLACE_POOL_CACHE.clear()
    with app._ACTIVITY_POOL_LOCK:
        app._ACTIVITY_POOL_CACHE.clear()
    return app


def _ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def test_local_restaurant_cache_hit_budget(app_module):
    app = app_module
    req = app.NightOutSuggestionRequest(
        party_size=1,
        latitude=LAT,
        longitude=LON,
        radius_meters=5 * 1609.34,
        cuisines=["Italian"],
        dietary_preferences=[],
        excluded_names=[],
    )

    t0 = time.perf_counter()
    miss = _run(app.get_night_out_suggestion(req))
    miss_ms = _ms(t0)
    assert miss.get("cache_hit") is False
    assert (miss.get("pool_size") or 0) >= 5

    samples = []
    for _ in range(5):
        t0 = time.perf_counter()
        hit = _run(app.get_night_out_suggestion(req))
        samples.append(_ms(t0))
        assert hit.get("cache_hit") is True
        assert hit["name"] == miss["name"]

    p95 = max(samples)
    print(
        f"\n[local restaurant] miss={miss_ms:.1f}ms "
        f"hits_ms={[round(s, 3) for s in samples]} p95={p95:.3f}ms "
        f"budget={CACHE_HIT_BUDGET_MS}"
    )
    assert p95 <= CACHE_HIT_BUDGET_MS


def test_local_halal_stable_pick(app_module):
    app = app_module
    req = app.NightOutSuggestionRequest(
        party_size=1,
        latitude=LAT,
        longitude=LON,
        radius_meters=5 * 1609.34,
        cuisines=[],
        dietary_preferences=["Halal"],
        excluded_names=[],
    )
    a = _run(app.get_night_out_suggestion(req))
    b = _run(app.get_night_out_suggestion(req))
    assert b.get("cache_hit") is True
    assert a["name"] == b["name"]
    assert (a.get("pool_size") or 0) >= 10


def test_local_events_cache_hit_budget(app_module):
    app = app_module
    req = app.NightOutEventRequest(
        party_size=1,
        latitude=LAT,
        longitude=LON,
        radius_miles=25,
        interests=["Hiking"],
        excluded_event_ids=[],
    )

    t0 = time.perf_counter()
    miss = _run(app.get_night_out_events(req))
    miss_ms = _ms(t0)
    assert (miss.get("pool_size") or 0) >= 5

    samples = []
    for _ in range(5):
        t0 = time.perf_counter()
        hit = _run(app.get_night_out_events(req))
        samples.append(_ms(t0))
        assert hit.get("cache_hit") is True
        assert hit["name"] == miss["name"]

    p95 = max(samples)
    print(
        f"\n[local events/hiking] miss={miss_ms:.1f}ms "
        f"hits_ms={[round(s, 3) for s in samples]} p95={p95:.3f}ms "
        f"budget={CACHE_HIT_BUDGET_MS}"
    )
    assert p95 <= CACHE_HIT_BUDGET_MS
