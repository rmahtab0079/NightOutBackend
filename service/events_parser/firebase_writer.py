"""Write curated events to Firebase for each user."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import firebase_admin
from firebase_admin import credentials, firestore

_db = None


def _get_db():
    global _db
    if _db is not None:
        return _db

    service_account_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH", "serviceAccount.json")
    project_id = os.getenv("FIREBASE_PROJECT_ID", "nightoutclient-7931e")

    if not firebase_admin._apps:
        if os.path.exists(service_account_path):
            cred = credentials.Certificate(service_account_path)
            firebase_admin.initialize_app(cred, {"projectId": project_id})
        else:
            firebase_admin.initialize_app()

    _db = firestore.client()
    return _db


def get_all_users_with_preferences() -> list[dict]:
    """
    Fetch all user documents from the user_preferences collection.

    Returns a list of dicts, each containing at minimum:
      - email (str, document ID)
      - interests (list[str])
      - cuisines (list[str])
    And optionally location data if stored.
    """
    db = _get_db()
    docs = db.collection("user_preferences").stream()
    users = []
    for doc in docs:
        data = doc.to_dict()
        data["email"] = doc.id
        users.append(data)
    return users


def get_user_swipe_categories(email: str) -> set[str]:
    """
    Derive the categories a user has previously liked from their swipe history.

    Checks both event and place swipe collections.
    """
    db = _get_db()
    liked_categories: set[str] = set()

    from .models import CLASSIFICATION_TO_CATEGORY

    try:
        event_doc = db.collection("user_event_swipes").document(email).get()
        if event_doc.exists:
            data = event_doc.to_dict()
            for ev in data.get("likedEvents", []):
                segment = (ev.get("segment") or "").lower()
                if segment in CLASSIFICATION_TO_CATEGORY:
                    liked_categories.add(CLASSIFICATION_TO_CATEGORY[segment])
                genre = (ev.get("genre") or "").lower()
                if genre in CLASSIFICATION_TO_CATEGORY:
                    liked_categories.add(CLASSIFICATION_TO_CATEGORY[genre])
    except Exception:
        pass

    try:
        place_doc = db.collection("user_place_swipes").document(email).get()
        if place_doc.exists:
            data = place_doc.to_dict()
            for pl in data.get("likedPlaces", []):
                ptype = (pl.get("place_type") or "").lower()
                if "restaurant" in ptype or "cafe" in ptype or "bar" in ptype:
                    liked_categories.add("food")
                elif "gym" in ptype or "sport" in ptype:
                    liked_categories.add("sports")
                elif "museum" in ptype or "gallery" in ptype or "theater" in ptype:
                    liked_categories.add("arts")
                elif "park" in ptype:
                    liked_categories.add("outdoors")
    except Exception:
        pass

    return liked_categories


# Collection for storing parsed events by composite id (source:source_id) for lookup by id
PARSED_EVENTS_CATALOG_COLLECTION = "parsed_events_catalog"


def _event_catalog_id(event: dict) -> str | None:
    """Generate a stable document id for the event catalog (source:source_id)."""
    source = (event.get("source") or "").strip()
    source_id = (event.get("source_id") or event.get("name") or "").strip()
    if not source or not source_id:
        return None
    return f"{source}:{source_id}"


def write_parsed_events_to_catalog(events: list[dict]) -> None:
    """
    Upsert parsed events into the parsed_events_catalog collection.
    Each document id = source:source_id; body = full event dict + last_updated.
    """
    db = _get_db()
    coll = db.collection(PARSED_EVENTS_CATALOG_COLLECTION)
    now = datetime.utcnow().isoformat() + "Z"
    for event in events:
        doc_id = _event_catalog_id(event)
        if not doc_id:
            continue
        payload = {**event, "last_updated": now}
        coll.document(doc_id).set(payload)
    if events:
        print(f"  [firebase] Wrote {len(events)} events to {PARSED_EVENTS_CATALOG_COLLECTION}")


def write_curated_events(
    email: str,
    grouped_events: dict[str, list[dict]],
    run_id: str,
    *,
    artist_picks: list[dict] | None = None,
) -> None:
    """
    Write curated events to the user_curated_events collection in Firebase.
    Also upserts each event into parsed_events_catalog for lookup by id.

    Document structure (keyed by email):
    {
      "email": "user@example.com",
      "last_updated": "2026-02-11T...",
      "run_id": "...",
      "events_by_category": {
        "sports": [ { ...event... }, ... ],
        "music": [ ... ],
        "arts": [ ... ],
        "food": [ ... ],
        "outdoors": [ ... ],
        "other": [ ... ]
      },
      "total_events": 42,
      "artist_picks": [ { ...event..., "matched_artist": "Taylor Swift" }, ... ]
    }

    `artist_picks` powers the home screen's "Artists You Love" carousel and is
    a flat, pre-sorted list (highest relevance first).
    """
    db = _get_db()
    total = sum(len(v) for v in grouped_events.values())

    doc_ref = db.collection("user_curated_events").document(email)
    doc_ref.set({
        "email": email,
        "last_updated": datetime.utcnow().isoformat() + "Z",
        "run_id": run_id,
        "events_by_category": grouped_events,
        "total_events": total,
        "artist_picks": artist_picks or [],
    })

    # Store each parsed event in the catalog table for lookup by id
    all_events: list[dict] = []
    for event_list in grouped_events.values():
        all_events.extend(event_list)
    if artist_picks:
        all_events.extend(artist_picks)
    write_parsed_events_to_catalog(all_events)

    print(f"  [firebase] Wrote {total} curated events for {email}")

    # Re-emit the precomputed home payload so the client's Firestore stream
    # sees the new curated events without an extra HTTP round trip.
    try:
        from service.home_payload import write_home_payload_safe

        write_home_payload_safe(email)
    except Exception as e:
        print(f"  [firebase] home_payload refresh failed for {email}: {e}")


def update_user_location(email: str, latitude: float, longitude: float) -> None:
    """Store last-known location for a user (used by cron to know where to scrape)."""
    db = _get_db()
    db.collection("user_preferences").document(email).update({
        "last_latitude": latitude,
        "last_longitude": longitude,
        "location_updated_at": datetime.utcnow().isoformat() + "Z",
    })


def clear_user_curated_events() -> int:
    """
    Delete all documents in user_curated_events so the next pipeline run
    repopulates with correct data. Use after fixing pipeline bugs (e.g. wrong
    events in food category).
    Returns the number of documents deleted.
    """
    db = _get_db()
    coll = db.collection("user_curated_events")
    deleted = 0
    for doc in coll.stream():
        doc.reference.delete()
        deleted += 1
    if deleted:
        print(f"  [firebase] Deleted {deleted} documents from user_curated_events")
    return deleted


def _active_user_locations(radius_miles: float) -> list[tuple[float, float]]:
    """
    Return (lat, lon) for every user_preferences doc that has a stored
    location. Used by `purge_catalog_outside_user_radius` to know which
    catalog entries are still relevant to *some* current user.
    """
    db = _get_db()
    locs: list[tuple[float, float]] = []
    for doc in db.collection("user_preferences").stream():
        data = doc.to_dict() or {}
        lat = data.get("last_latitude")
        lon = data.get("last_longitude")
        if lat is None or lon is None:
            continue
        try:
            locs.append((float(lat), float(lon)))
        except (TypeError, ValueError):
            continue
    return locs


def purge_catalog_outside_user_radius(
    radius_miles: float = 50.0,
) -> dict:
    """
    Drop parsed_events_catalog entries that aren't within `radius_miles` of
    *any* current user's stored location.

    Catalog docs missing lat/lon are kept (we cannot prove they're far away);
    they are filtered out at render time by the per-user location guard. If
    no users have a stored location, this is a no-op so we don't wipe data.
    """
    from service.home_payload import _haversine_miles  # local import keeps deps light

    db = _get_db()
    coll = db.collection(PARSED_EVENTS_CATALOG_COLLECTION)

    user_locs = _active_user_locations(radius_miles)
    if not user_locs:
        print("  [firebase] catalog purge: no users with location, skipping")
        return {"kept": 0, "deleted_far": 0, "kept_no_coord": 0}

    deleted_far = 0
    kept = 0
    kept_no_coord = 0
    for doc in coll.stream():
        data = doc.to_dict() or {}
        lat_raw = data.get("latitude")
        lon_raw = data.get("longitude")
        if lat_raw is None or lon_raw is None:
            kept_no_coord += 1
            continue
        try:
            ev_lat = float(lat_raw)
            ev_lon = float(lon_raw)
        except (TypeError, ValueError):
            kept_no_coord += 1
            continue
        in_range = any(
            _haversine_miles(ulat, ulon, ev_lat, ev_lon) <= radius_miles
            for ulat, ulon in user_locs
        )
        if in_range:
            kept += 1
        else:
            doc.reference.delete()
            deleted_far += 1

    print(
        f"  [firebase] catalog purge: kept {kept}, removed {deleted_far} "
        f"out-of-range, kept {kept_no_coord} without coords"
    )
    return {
        "kept": kept,
        "deleted_far": deleted_far,
        "kept_no_coord": kept_no_coord,
    }


def _doc_is_stale_event(event: dict) -> tuple[bool, str]:
    """
    Cheap (offline) staleness check. Returns (True, reason) when the catalog
    entry should be purged for one of:
      - "past": event date is in the past
      - "no_image": missing or non-http(s) image_url

    A second pass (`_doc_image_is_unreachable`) probes the network and is run
    only on docs that pass this cheap check.
    """
    from datetime import datetime as _dt, timezone as _tz

    image_url_raw = (event.get("image_url") or "").strip()
    image_url_lower = image_url_raw.lower()
    has_usable_image = (
        bool(image_url_raw)
        and image_url_lower not in {"none", "null", "false"}
        and (
            image_url_lower.startswith("http://")
            or image_url_lower.startswith("https://")
        )
    )
    if not has_usable_image:
        return True, "no_image"

    date_str = (event.get("date") or "").strip()
    if not date_str:
        return False, ""

    time_str = (event.get("time") or "").strip()
    now_utc = _dt.now(_tz.utc)

    if time_str:
        iso = f"{date_str}T{time_str}"
        if iso.endswith("Z"):
            iso = iso[:-1] + "+00:00"
        try:
            dt = _dt.fromisoformat(iso)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=_tz.utc)
            if dt.astimezone(_tz.utc) < now_utc:
                return True, "past"
            return False, ""
        except ValueError:
            pass

    try:
        event_date = _dt.strptime(date_str[:10], "%Y-%m-%d").date()
    except ValueError:
        return False, ""
    if event_date < now_utc.date():
        return True, "past"
    return False, ""


def purge_stale_parsed_events(
    *,
    probe_network: bool = True,
    max_workers: int = 16,
) -> dict:
    """
    Remove entries from parsed_events_catalog that are:
      - past events,
      - missing a usable http(s) image_url, or
      - (if probe_network) point at an image_url that no longer responds 2xx.

    The network probe is what kills lingering Partiful 403 URLs that look
    valid but can't actually be loaded by the client.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    # Lazy import to avoid a circular dep with pipeline.py.
    from .pipeline import _url_is_reachable  # type: ignore

    db = _get_db()
    coll = db.collection(PARSED_EVENTS_CATALOG_COLLECTION)

    deleted_past = 0
    deleted_no_image = 0
    deleted_unreachable = 0
    survivors: list[tuple] = []  # (doc_ref, image_url)

    for doc in coll.stream():
        data = doc.to_dict() or {}
        is_stale, reason = _doc_is_stale_event(data)
        if is_stale:
            doc.reference.delete()
            if reason == "past":
                deleted_past += 1
            elif reason == "no_image":
                deleted_no_image += 1
            continue
        survivors.append((doc.reference, (data.get("image_url") or "").strip()))

    if probe_network and survivors:
        kept = 0
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_url_is_reachable, url): (ref, url)
                for ref, url in survivors
                if url
            }
            for fut in as_completed(futures):
                ref, _url = futures[fut]
                try:
                    ok = fut.result()
                except Exception:
                    ok = False
                if ok:
                    kept += 1
                else:
                    ref.delete()
                    deleted_unreachable += 1
    else:
        kept = len(survivors)

    print(
        f"  [firebase] parsed_events_catalog purge: "
        f"kept {kept}, removed {deleted_past} past, "
        f"{deleted_no_image} imageless, {deleted_unreachable} unreachable"
    )
    return {
        "kept": kept,
        "deleted_past": deleted_past,
        "deleted_no_image": deleted_no_image,
        "deleted_unreachable": deleted_unreachable,
    }
