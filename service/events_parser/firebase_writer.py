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
      "total_events": 42
    }
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
    })

    # Store each parsed event in the catalog table for lookup by id
    all_events: list[dict] = []
    for event_list in grouped_events.values():
        all_events.extend(event_list)
    write_parsed_events_to_catalog(all_events)

    print(f"  [firebase] Wrote {total} curated events for {email}")


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
