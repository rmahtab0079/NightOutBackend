"""
Compose and publish hourly-picks notifications for all users.

Each run picks **one** event suggestion per user — randomly chosen from the
top-N events per category in their `user_curated_events` doc, filtered to
exclude any event id we've already sent in a previous notification (tracked
in `user_notification_history/{email}`). When the user's eligible pool is
exhausted, the history is reset so notifications keep flowing instead of
going silent.
"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime
from typing import Optional

import firebase_admin
from firebase_admin import credentials, firestore

_db = None
_publisher = None

CATEGORY_EMOJI = {
    "food": "\U0001F355",
    "music": "\U0001F3B5",
    "sports": "\u26BD",
    "activities": "\U0001F3AE",
    "movies": "\U0001F3AC",
    "tv": "\U0001F4FA",
    "artist_picks": "\U0001F3A4",  # microphone
    "other": "\u2728",  # sparkles
}

# Collection that records which event ids we've already pushed to each user
# so we don't re-notify them about the same suggestion.
NOTIFICATION_HISTORY_COLLECTION = "user_notification_history"

# Categories considered "events" for notification purposes. Movies/TV live in
# their own carousels and aren't actionable from a push the same way.
EVENT_CATEGORIES = ("music", "sports", "activities", "food")

# How many top-scored items per category go into the eligible pool. The
# pipeline writes them already sorted by score, so the slice is the top of
# each list.
TOP_PICKS_PER_CATEGORY = 3

# Cap the per-user history so a long-tenured account's `sent_event_ids`
# array doesn't bloat the document past Firestore's 1MB limit. When we hit
# the cap we drop the oldest ids to make room.
MAX_HISTORY_EVENT_IDS = 500


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
            # No service-account file (e.g. running on Cloud Run with ADC).
            # We still need to pin the Firebase project explicitly — otherwise
            # firebase_admin falls back to the ADC project (the GCP project,
            # not the Firebase project), which sends Firestore traffic to a
            # database that doesn't exist there.
            firebase_admin.initialize_app(options={"projectId": project_id})

    _db = firestore.client()
    return _db


def _get_publisher():
    global _publisher
    if _publisher is not None:
        return _publisher
    try:
        from google.cloud import pubsub_v1
        _publisher = pubsub_v1.PublisherClient()
    except Exception as e:
        print(f"Pub/Sub publisher init failed: {e}")
    return _publisher


# ---------------------------------------------------------------------------
# Event id + pool helpers
# ---------------------------------------------------------------------------


def _event_id(event: dict) -> Optional[str]:
    """Return a stable id for `event` so we can dedupe across runs.

    Mirrors the scheme used by `firebase_writer._event_catalog_id` so the id
    stored in the per-user history matches the id in
    `parsed_events_catalog`. Falls back to `name|date` when source/source_id
    are missing so curated events scraped from less structured providers
    still get a stable key.
    """
    source = (event.get("source") or "").strip()
    sid = (event.get("source_id") or "").strip()
    if source and sid:
        return f"{source}:{sid}"

    name = (event.get("name") or event.get("title") or "").strip()
    if not name:
        return None
    when = (event.get("date") or event.get("start_date") or "").strip()
    return f"name:{name}|{when}"


def _build_event_pool(curated_doc: dict) -> list[dict]:
    """Collect the top picks across event categories + artist picks.

    Caller-side dedupe (via `_event_id`) prevents an event that lives in
    multiple buckets (e.g. an artist pick that also shows up under "music")
    from inflating its odds of being picked.
    """
    pool: list[dict] = []
    seen_ids: set[str] = set()

    events_by_cat = curated_doc.get("events_by_category") or {}
    for cat in EVENT_CATEGORIES:
        items = events_by_cat.get(cat) or []
        for ev in items[:TOP_PICKS_PER_CATEGORY]:
            if not isinstance(ev, dict):
                continue
            eid = _event_id(ev)
            if not eid or eid in seen_ids:
                continue
            seen_ids.add(eid)
            annotated = dict(ev)
            annotated.setdefault("_category", cat)
            pool.append(annotated)

    for ev in (curated_doc.get("artist_picks") or []):
        if not isinstance(ev, dict):
            continue
        eid = _event_id(ev)
        if not eid or eid in seen_ids:
            continue
        seen_ids.add(eid)
        annotated = dict(ev)
        annotated.setdefault("_category", "artist_picks")
        pool.append(annotated)

    return pool


# ---------------------------------------------------------------------------
# Per-user notification history
# ---------------------------------------------------------------------------


def _history_ref(db, email: str):
    return db.collection(NOTIFICATION_HISTORY_COLLECTION).document(email)


def _load_sent_ids(db, email: str) -> set[str]:
    snap = _history_ref(db, email).get()
    if not snap.exists:
        return set()
    return set((snap.to_dict() or {}).get("sent_event_ids", []) or [])


def _record_sent_id(db, email: str, event_id: str) -> None:
    """Append `event_id` to the user's notification history.

    Uses ArrayUnion in the common (uncapped) path so two concurrent runs
    can't lose each other's writes. When the array would exceed
    `MAX_HISTORY_EVENT_IDS` we fall back to a non-atomic trim — that's
    acceptable because we publish at most one notification per user per
    cron tick, so concurrent appends for the same user are not expected.
    """
    ref = _history_ref(db, email)
    snap = ref.get()
    now = datetime.utcnow().isoformat() + "Z"

    if not snap.exists:
        ref.set({
            "email": email,
            "sent_event_ids": [event_id],
            "last_sent_id": event_id,
            "last_sent_at": now,
        })
        return

    existing = (snap.to_dict() or {}).get("sent_event_ids", []) or []
    if len(existing) >= MAX_HISTORY_EVENT_IDS:
        # Drop the oldest entries to stay under the cap. The order in the
        # array reflects the order we wrote them in, so a tail slice keeps
        # the most recently sent ids — exactly what we want for dedupe.
        trimmed = existing[-(MAX_HISTORY_EVENT_IDS - 1):]
        if event_id not in trimmed:
            trimmed.append(event_id)
        ref.set({
            "email": email,
            "sent_event_ids": trimmed,
            "last_sent_id": event_id,
            "last_sent_at": now,
        }, merge=True)
        return

    ref.set({
        "email": email,
        "sent_event_ids": firestore.ArrayUnion([event_id]),
        "last_sent_id": event_id,
        "last_sent_at": now,
    }, merge=True)


def _reset_sent_ids(db, email: str) -> None:
    """Clear the user's history so the next notification can pick freely.

    Triggered when every event in the current pool has already been sent —
    typically because the curated list is small relative to the cron cadence.
    """
    ref = _history_ref(db, email)
    ref.set({
        "email": email,
        "sent_event_ids": [],
        "reset_at": datetime.utcnow().isoformat() + "Z",
    }, merge=True)


# ---------------------------------------------------------------------------
# Notification formatting
# ---------------------------------------------------------------------------


def _format_notification(event: dict) -> tuple[str, str, dict]:
    """Build (title, body, data) for a single picked event."""
    name = (
        event.get("name")
        or event.get("title")
        or "Something fun nearby"
    )
    cat = event.get("_category") or event.get("category") or "other"
    emoji = CATEGORY_EMOJI.get(cat, CATEGORY_EMOJI["other"])

    title = f"{emoji} New pick for you"

    venue = (event.get("venue_name") or event.get("venue") or "").strip()
    when = (event.get("date") or event.get("start_date") or "").strip()
    body_parts: list[str] = [str(name).strip()]
    if venue:
        body_parts.append(venue)
    if when:
        body_parts.append(when)
    body = " · ".join(p for p in body_parts if p)

    data: dict[str, str] = {
        "type": "hourly_picks",
        "event_id": _event_id(event) or "",
        "event_name": str(name),
        "event_category": str(cat),
    }
    url = event.get("url") or event.get("event_url")
    if url:
        data["event_url"] = str(url)
    return title, body, data


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def compose_hourly_picks() -> list[dict]:
    """For every user with curated events, compose **one** randomized
    notification payload they haven't seen before.

    Returns a list of dicts: { uid, email, title, body, data }. The history
    write happens here (before publish) so that even if the Pub/Sub publish
    later fails, we won't immediately re-pick the same event the next run —
    duplicates are worse than the rare missed delivery.
    """
    db = _get_db()
    payloads: list[dict] = []

    prefs_docs = {doc.id: doc.to_dict() for doc in db.collection("user_preferences").stream()}
    curated_docs = db.collection("user_curated_events").stream()

    for doc in curated_docs:
        email = doc.id
        curated = doc.to_dict() or {}
        if not curated:
            continue

        uid = (prefs_docs.get(email) or {}).get("uid")
        if not uid:
            continue

        pool = _build_event_pool(curated)
        if not pool:
            continue

        sent_ids = _load_sent_ids(db, email)
        candidates = [ev for ev in pool if (_event_id(ev) or "") not in sent_ids]
        reset_history = False
        if not candidates:
            # Pool is fully exhausted — wipe the per-user history so the
            # next cycle starts fresh instead of going silent.
            _reset_sent_ids(db, email)
            candidates = pool
            reset_history = True

        chosen = random.choice(candidates)
        eid = _event_id(chosen)
        if not eid:
            continue

        title, body, data = _format_notification(chosen)

        # Record before publish so retries / concurrent triggers can't pick
        # the same event again. If the recording itself fails we still send
        # — better one duplicate than a missed notification.
        try:
            _record_sent_id(db, email, eid)
        except Exception as e:
            print(f"Failed to record notification history for {email}: {e}")

        if reset_history:
            data["history_reset"] = "1"

        payloads.append({
            "uid": uid,
            "email": email,
            "title": title,
            "body": body,
            "data": data,
        })

    return payloads


def _resolve_pubsub_project_id() -> Optional[str]:
    """Resolve the GCP project id that owns the Pub/Sub topic.

    NOTE: This is the *GCP* project (e.g. ``optimal-aegis-470701-j8``), not
    the Firebase project (``nightoutclient-7931e``). Cloud Run does not set
    ``GOOGLE_CLOUD_PROJECT`` automatically, so we fall back to ADC and the
    metadata server before giving up.
    """
    pid = (
        os.getenv("PUBSUB_PROJECT_ID")
        or os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GCP_PROJECT")
    )
    if pid:
        return pid
    try:
        import google.auth
        _, detected = google.auth.default()
        if detected:
            return detected
    except Exception as e:
        print(f"google.auth.default() failed: {e}")
    return "optimal-aegis-470701-j8"


def publish_hourly_picks(payloads: list[dict]) -> int:
    """Publish one Pub/Sub message per user. Returns the count published."""
    publisher = _get_publisher()
    if not publisher:
        print("No Pub/Sub publisher available — skipping publish")
        return 0

    project_id = _resolve_pubsub_project_id()
    topic_name = os.getenv(
        "PUBSUB_HOURLY_PICKS_TOPIC",
        os.getenv("PUBSUB_DAILY_PICKS_TOPIC", "hourly-picks"),
    )
    if not project_id:
        print("No project ID configured — skipping publish")
        return 0

    topic_path = publisher.topic_path(project_id, topic_name)
    print(f"Publishing {len(payloads)} hourly picks to {topic_path}")
    published = 0
    futures = []

    for payload in payloads:
        try:
            message = json.dumps(payload).encode("utf-8")
            futures.append((payload, publisher.publish(topic_path, message)))
        except Exception as e:
            print(f"Failed to publish for {payload.get('email')}: {e}")

    for payload, fut in futures:
        try:
            msg_id = fut.result(timeout=30)
            published += 1
            print(f"Published msg_id={msg_id} for {payload.get('email')}")
        except Exception as e:
            print(f"Publish ack failed for {payload.get('email')}: {e}")

    print(f"Published {published}/{len(payloads)} hourly pick notifications")
    return published
