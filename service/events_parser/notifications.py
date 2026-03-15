"""
Compose and publish daily-picks notifications for all users.

Reads each user's curated events from Firestore, picks the top item per
category, and publishes a Pub/Sub message per user for the Cloud Function
to deliver via FCM.
"""

from __future__ import annotations

import json
import os
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
}


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


def _top_pick_name(events: list[dict]) -> Optional[str]:
    """Return the name of the highest-scored item in a category list."""
    if not events:
        return None
    best = events[0]
    return best.get("name") or best.get("title") or None


def compose_daily_picks() -> list[dict]:
    """
    For every user with curated events, compose a notification payload.
    Returns a list of dicts: { uid, email, title, body, data }.
    """
    db = _get_db()
    payloads: list[dict] = []

    prefs_docs = {doc.id: doc.to_dict() for doc in db.collection("user_preferences").stream()}
    curated_docs = db.collection("user_curated_events").stream()

    for doc in curated_docs:
        email = doc.id
        data = doc.to_dict()
        events_by_cat = data.get("events_by_category", {})

        if not events_by_cat:
            continue

        uid = prefs_docs.get(email, {}).get("uid")
        if not uid:
            for pref_email, pref_data in prefs_docs.items():
                if pref_email == email:
                    uid = pref_data.get("uid")
                    break
        if not uid:
            continue

        highlights: list[str] = []
        top_data: dict[str, str] = {}
        for cat in ("food", "music", "sports", "activities", "movies", "tv"):
            items = events_by_cat.get(cat, [])
            name = _top_pick_name(items)
            if name:
                emoji = CATEGORY_EMOJI.get(cat, "")
                highlights.append(f"{emoji} {name}")
                top_data[f"top_{cat}"] = name

        if not highlights:
            continue

        body = " · ".join(highlights[:3])
        if len(highlights) > 3:
            body += f" +{len(highlights) - 3} more"

        payloads.append({
            "uid": uid,
            "email": email,
            "title": "Your daily picks are ready!",
            "body": body,
            "data": {
                "type": "daily_picks",
                **top_data,
            },
        })

    return payloads


def publish_daily_picks(payloads: list[dict]) -> int:
    """Publish one Pub/Sub message per user. Returns the count published."""
    publisher = _get_publisher()
    if not publisher:
        print("No Pub/Sub publisher available — skipping publish")
        return 0

    project_id = os.getenv("FIREBASE_PROJECT_ID", os.getenv("GOOGLE_CLOUD_PROJECT"))
    topic_name = os.getenv("PUBSUB_DAILY_PICKS_TOPIC", "daily-picks")
    if not project_id:
        print("No project ID configured — skipping publish")
        return 0

    topic_path = publisher.topic_path(project_id, topic_name)
    published = 0

    for payload in payloads:
        try:
            message = json.dumps(payload).encode("utf-8")
            publisher.publish(topic_path, message)
            published += 1
        except Exception as e:
            print(f"Failed to publish for {payload.get('email')}: {e}")

    print(f"Published {published}/{len(payloads)} daily pick notifications")
    return published
