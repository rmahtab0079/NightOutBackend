"""
Home payload precomputation.

Bundles everything the Flutter home screen needs (curated events, top movies,
top TV shows, featured carousel) into a single Firestore document per user
under the `user_home_payload/{email}` collection.

The client subscribes to that document via Firestore streams so home renders
instantly from a single read instead of issuing 4 sequential HTTP round-trips
on every open.
"""

from __future__ import annotations

import math
import os
import random
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

import firebase_admin
from firebase_admin import credentials, firestore


HOME_PAYLOAD_COLLECTION = "user_home_payload"
# v2 adds `artist_picks`. The Flutter client invalidates its on-device cache
# whenever this version bumps, so older blobs without the new field are
# rebuilt rather than read.
SCHEMA_VERSION = 2

# Sizes mirror what home_screen.dart currently asks for.
TOP_MOVIES_COUNT = 16
TOP_TV_COUNT = 16
FEATURED_COUNT = 12

# Default radius (miles) used when filtering global catalog events for a single
# user. Mirrors the cron's scrape radius so the Featured carousel never shows
# events from a city the user can't reach.
DEFAULT_USER_RADIUS_MILES = 50.0

# Artist picks use a wider radius than the regular catalog feed: arena tours
# announce months ahead and fans regularly drive 1-2 hours for a favourite
# artist, so a 50-mile cap would silently empty the carousel.
ARTIST_PICK_RADIUS_MILES = 150.0

# Heavy TMDB fields the home cards never read. Stripping them keeps the
# Firestore document small (1 MB hard limit) and the stream fast.
_HEAVY_DETAIL_KEYS = {
    "production_companies",
    "production_countries",
    "spoken_languages",
    "credits",
    "videos",
    "images",
    "external_ids",
    "translations",
    "alternative_titles",
    "release_dates",
    "content_ratings",
    "keywords",
    "aggregate_credits",
    "recommendations",
    "similar",
    "lists",
    "watch/providers",
    "season_air_dates",
}


_db: firestore.Client | None = None


def _get_db() -> firestore.Client | None:
    global _db
    if _db is not None:
        return _db
    try:
        if not firebase_admin._apps:  # type: ignore[attr-defined]
            sa_path = os.getenv(
                "FIREBASE_SERVICE_ACCOUNT_PATH", "serviceAccount.json"
            )
            project_id = os.getenv("FIREBASE_PROJECT_ID")
            if os.path.exists(sa_path):
                cred = credentials.Certificate(sa_path)
                opts = {"projectId": project_id} if project_id else None
                firebase_admin.initialize_app(cred, opts)
            else:
                opts = {"projectId": project_id} if project_id else None
                firebase_admin.initialize_app(options=opts)
        _db = firestore.client()
    except Exception as e:
        print(f"[home_payload] Firestore init failed: {e}")
        _db = None
    return _db


# ---------------------------------------------------------------------------
# Filters (mirrors application._event_passes_filters so we can keep them in
# sync without importing the FastAPI module). Duplicated intentionally to
# avoid pulling in the entire web app from a background pipeline.
# ---------------------------------------------------------------------------
def _event_is_future_or_present(event: dict) -> bool:
    date_str = (event.get("date") or "").strip()
    if not date_str:
        return True

    time_str = (event.get("time") or "").strip()
    now_utc = datetime.now(timezone.utc)

    if time_str:
        iso = f"{date_str}T{time_str}"
        if iso.endswith("Z"):
            iso = iso[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(iso)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc) >= now_utc
        except ValueError:
            pass

    try:
        event_date = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
    except ValueError:
        return True
    return event_date >= now_utc.date()


def _event_has_image(event: dict) -> bool:
    image_url = (event.get("image_url") or "").strip()
    if not image_url:
        return False
    lower = image_url.lower()
    if lower in {"none", "null", "false"}:
        return False
    return lower.startswith("http://") or lower.startswith("https://")


def _event_passes_filters(event: dict) -> bool:
    return _event_is_future_or_present(event) and _event_has_image(event)


# ---------------------------------------------------------------------------
# Location filtering — keeps catalog events anchored to the user's region.
# Without this, the Featured carousel pulls a global random sample and a user
# in Illinois ends up seeing concerts in NYC and SF.
# ---------------------------------------------------------------------------
def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 3958.8
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _coerce_coord(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def event_is_within_user_radius(
    event: dict,
    user_lat: Optional[float],
    user_lon: Optional[float],
    radius_miles: float = DEFAULT_USER_RADIUS_MILES,
) -> bool:
    """
    True if `event` is within `radius_miles` of the user's location.

    When the user has no known location (`user_lat`/`user_lon` is None), every
    event passes — we have no basis to discriminate. When the user *does* have
    a location, an event is rejected if it lacks coordinates: we cannot prove
    it's nearby, and historically these are the events that leak across cities.
    """
    if user_lat is None or user_lon is None:
        return True
    ev_lat = _coerce_coord(event.get("latitude"))
    ev_lon = _coerce_coord(event.get("longitude"))
    if ev_lat is None or ev_lon is None:
        return False
    return _haversine_miles(user_lat, user_lon, ev_lat, ev_lon) <= radius_miles


def filter_catalog_events_by_location(
    events: Iterable[dict],
    user_lat: Optional[float],
    user_lon: Optional[float],
    radius_miles: float = DEFAULT_USER_RADIUS_MILES,
) -> list[dict]:
    """Convenience wrapper used by application.py endpoints."""
    return [
        e for e in events
        if event_is_within_user_radius(e, user_lat, user_lon, radius_miles)
    ]


def _filter_events_by_category(events_by_category: dict) -> dict:
    if not isinstance(events_by_category, dict):
        return {}
    out: dict[str, list[dict]] = {}
    for cat, items in events_by_category.items():
        if isinstance(items, list):
            out[cat] = [e for e in items if _event_passes_filters(e)]
        else:
            out[cat] = items
    return out


def _strip_detail(detail: dict | None) -> dict | None:
    if not isinstance(detail, dict):
        return detail
    return {k: v for k, v in detail.items() if k not in _HEAVY_DETAIL_KEYS}


def _trim_movie_item(item: dict) -> dict:
    out = dict(item)
    if "detail" in out:
        out["detail"] = _strip_detail(out["detail"])
    for k in list(out.keys()):
        if k in _HEAVY_DETAIL_KEYS:
            out.pop(k, None)
    return out


# ---------------------------------------------------------------------------
# Builders for each section of the payload.
# ---------------------------------------------------------------------------
def _load_curated(db: firestore.Client, email: str) -> dict[str, list[dict]]:
    doc = db.collection("user_curated_events").document(email).get()
    if not doc.exists:
        return {}
    data = doc.to_dict() or {}
    return _filter_events_by_category(data.get("events_by_category", {}))


def _load_artist_picks(
    db: firestore.Client,
    email: str,
    user_lat: Optional[float],
    user_lon: Optional[float],
    radius_miles: float = ARTIST_PICK_RADIUS_MILES,
) -> list[dict]:
    """
    Load the artist_picks list from user_curated_events, then re-apply the
    standard date/image/location filters so picks left over from a previous
    pipeline run never surface stale or out-of-region items.
    """
    doc = db.collection("user_curated_events").document(email).get()
    if not doc.exists:
        return []
    data = doc.to_dict() or {}
    raw = data.get("artist_picks", []) or []
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        if not _event_passes_filters(item):
            continue
        if not event_is_within_user_radius(item, user_lat, user_lon, radius_miles):
            continue
        out.append(item)
    return out


def _precomputed_genre_ids(
    db: firestore.Client, email: str, asset_type: str
) -> list[int]:
    # Lazy import — keeps module import cheap and avoids any chance of a
    # circular dependency in the test runner.
    from service.user_genre_recommendations import (
        MOVIE_RECS_COLLECTION,
        TV_RECS_COLLECTION,
    )

    coll = MOVIE_RECS_COLLECTION if asset_type == "movie" else TV_RECS_COLLECTION
    doc = db.collection(coll).document(email).get()
    if not doc.exists:
        return []
    data = doc.to_dict() or {}
    genre_map = data.get("genres", {}) or {}
    ordered: list[int] = []
    seen: set[int] = set()
    for ids in genre_map.values():
        for asset_id in ids or []:
            try:
                aid = int(asset_id)
            except Exception:
                continue
            if aid in seen:
                continue
            seen.add(aid)
            ordered.append(aid)
    return ordered


def _build_top_assets(
    db: firestore.Client,
    email: str,
    asset_type: str,
    excluded_ids: set[int],
    count: int,
) -> list[dict]:
    from models.process_tv_and_movies import get_movie_detail

    candidate_ids = _precomputed_genre_ids(db, email, asset_type)
    if not candidate_ids:
        return []

    results: list[dict] = []
    total = len(candidate_ids)
    for idx, asset_id in enumerate(candidate_ids):
        if asset_id in excluded_ids:
            continue
        try:
            detail = get_movie_detail(asset_id, asset_type, include_providers=False)
        except Exception:
            detail = None
        if not detail:
            continue
        slim = _strip_detail(detail) or {}
        title = (
            slim.get("title")
            or slim.get("original_title")
            or slim.get("name")
            or slim.get("original_name")
        )
        item = {
            "movie_id" if asset_type == "movie" else "tv_id": asset_id,
            "id": asset_id,
            "title": title,
            "similarity": max(0.0, 1.0 - (idx / max(total, 1))),
            "detail": slim,
        }
        results.append(item)
        if len(results) >= count:
            break
    return results


def _build_featured(
    db: firestore.Client,
    catalog_events: list[dict],
    top_movies: list[dict],
    top_tv: list[dict],
    count: int,
) -> list[dict]:
    """Blend events + 1 movie + 1 TV for the home carousel."""
    event_slots = max(1, count - 2)
    by_cat: dict[str, list[dict]] = {}
    for ev in catalog_events:
        cat = ev.get("category") or "other"
        if cat == "food" and ev.get("source") != "google_places":
            cat = "other"
        by_cat.setdefault(cat, []).append(ev)

    cats = list(by_cat.keys())
    random.shuffle(cats)
    per_cat = max(1, event_slots // max(len(cats), 1))

    featured: list[dict] = []
    for cat in cats:
        pool = by_cat[cat]
        sample_size = min(per_cat, len(pool))
        for ev in random.sample(pool, sample_size):
            ev = dict(ev)
            ev["_type"] = "event"
            featured.append(ev)

    random.shuffle(featured)
    featured = featured[:event_slots]

    if top_movies:
        m = dict(top_movies[0])
        m["_type"] = "movie"
        featured.append(m)
    if top_tv:
        t = dict(top_tv[0])
        t["_type"] = "tv"
        featured.append(t)

    random.shuffle(featured)
    return featured[:count]


def _load_catalog_events(
    db: firestore.Client,
    user_lat: Optional[float] = None,
    user_lon: Optional[float] = None,
    radius_miles: float = DEFAULT_USER_RADIUS_MILES,
) -> list[dict]:
    """
    Stream the global parsed_events_catalog and keep only events that pass our
    standard date/image filters AND fall within `radius_miles` of the user's
    stored location.
    """
    out: list[dict] = []
    for cdoc in db.collection("parsed_events_catalog").stream():
        ev = cdoc.to_dict() or {}
        if not _event_passes_filters(ev):
            continue
        if not event_is_within_user_radius(ev, user_lat, user_lon, radius_miles):
            continue
        out.append(ev)
    return out


# ---------------------------------------------------------------------------
# Public API — build + write.
# ---------------------------------------------------------------------------
def build_home_payload(email: str) -> Optional[dict[str, Any]]:
    """
    Build (but do not persist) the precomputed home payload for `email`.
    Returns None if Firestore is unreachable; otherwise always returns a dict
    (which may have empty sections for brand-new users).
    """
    db = _get_db()
    if db is None:
        return None

    # User preferences -- needed for excluded movie/tv ids and for clamping
    # the global catalog down to events near the user.
    excluded_movie_ids: set[int] = set()
    excluded_tv_ids: set[int] = set()
    user_lat: Optional[float] = None
    user_lon: Optional[float] = None
    try:
        user_doc = db.collection("user_preferences").document(email).get()
        if user_doc.exists:
            prefs = user_doc.to_dict() or {}
            for v in prefs.get("dislikedMovieIds", []) or []:
                try:
                    excluded_movie_ids.add(int(v))
                except Exception:
                    pass
            for v in prefs.get("dislikedTvShowIds", []) or []:
                try:
                    excluded_tv_ids.add(int(v))
                except Exception:
                    pass
            user_lat = _coerce_coord(prefs.get("last_latitude"))
            user_lon = _coerce_coord(prefs.get("last_longitude"))
    except Exception as e:
        print(f"[home_payload] could not load prefs for {email}: {e}")

    events_by_category = _load_curated(db, email)
    artist_picks = _load_artist_picks(db, email, user_lat, user_lon)
    top_movies = [
        _trim_movie_item(m)
        for m in _build_top_assets(
            db, email, "movie", excluded_movie_ids, TOP_MOVIES_COUNT
        )
    ]
    top_tv = [
        _trim_movie_item(t)
        for t in _build_top_assets(
            db, email, "tv", excluded_tv_ids, TOP_TV_COUNT
        )
    ]

    catalog_events = _load_catalog_events(db, user_lat, user_lon)
    featured = _build_featured(
        db, catalog_events, top_movies, top_tv, FEATURED_COUNT
    )

    total_events = sum(
        len(v) for v in events_by_category.values() if isinstance(v, list)
    )
    return {
        "email": email,
        "version": SCHEMA_VERSION,
        "last_updated": datetime.utcnow().isoformat() + "Z",
        "events_by_category": events_by_category,
        "total_events": total_events,
        "top_movies": top_movies,
        "top_tv": top_tv,
        "featured": featured,
        "artist_picks": artist_picks,
    }


def write_home_payload(email: str) -> Optional[dict[str, Any]]:
    """
    Build and persist the home payload for `email`. Returns the payload dict
    (or None if Firestore is unavailable).

    Safe to call from background threads -- failures are logged, never raised,
    so they never bring down the parser pipeline or an HTTP request handler.
    """
    payload = build_home_payload(email)
    if payload is None:
        return None
    db = _get_db()
    if db is None:
        return None
    try:
        db.collection(HOME_PAYLOAD_COLLECTION).document(email).set(payload)
        print(
            f"  [home_payload] wrote {payload['total_events']} events, "
            f"{len(payload['top_movies'])} movies, {len(payload['top_tv'])} tv, "
            f"{len(payload.get('artist_picks', []))} artist picks for {email}"
        )
        return payload
    except Exception as e:
        print(f"[home_payload] write failed for {email}: {e}")
        return None


def write_home_payload_safe(email: str) -> None:
    """Fire-and-forget helper that swallows all exceptions."""
    try:
        write_home_payload(email)
    except Exception as e:
        print(f"[home_payload] background write failed for {email}: {e}")
