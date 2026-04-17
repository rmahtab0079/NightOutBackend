"""
Cloud Function: subscribes to Pub/Sub topic 'friend-activity' and sends FCM
push notifications when a friend creates an event or accepts a suggestion.

Trigger: Pub/Sub topic (friend-activity).
Message payload: JSON with actor_uid, actor_name, activity_type, event_name,
                 event_id, notify_uids.
"""
import json
import base64
import os

import firebase_admin
from firebase_admin import credentials, firestore, messaging

FCM_TOKENS_COLLECTION = "user_fcm_tokens"
_app_initialized = False

ACTIVITY_MESSAGES = {
    "created_event": "{name} is going to {event}",
    "accepted_suggestion": "{name} accepted a suggestion: {event}",
    "join_request": "{name} wants to join your event: {event}",
}


def _get_firestore():
    global _app_initialized
    if not _app_initialized:
        if not firebase_admin._apps:
            project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("FIREBASE_PROJECT_ID")
            svc_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH", "")
            if svc_path and os.path.exists(svc_path):
                cred = credentials.Certificate(svc_path)
                firebase_admin.initialize_app(cred, {"projectId": project_id})
            else:
                firebase_admin.initialize_app()
        _app_initialized = True
    return firestore.client()


def friend_activity_notify(event, context):
    """Triggered by Pub/Sub message on the friend-activity topic."""
    try:
        raw = base64.b64decode(event["data"]).decode("utf-8")
        payload = json.loads(raw)
    except Exception as e:
        print(f"Failed to decode message: {e}")
        return

    actor_name = payload.get("actor_name", "A friend")
    activity_type = payload.get("activity_type", "created_event")
    event_name = payload.get("event_name", "an event")
    event_id = payload.get("event_id", "")
    notify_uids = payload.get("notify_uids", [])

    if not notify_uids:
        print("No UIDs to notify")
        return

    template = ACTIVITY_MESSAGES.get(activity_type, "{name} has a new activity: {event}")
    body = template.format(name=actor_name, event=event_name)
    title = "Friend Activity"

    db = _get_firestore()
    total_sent = 0
    stale_tokens_by_uid: dict[str, list[str]] = {}

    for uid in notify_uids:
        tokens_doc = db.collection(FCM_TOKENS_COLLECTION).document(uid).get()
        if not tokens_doc.exists:
            continue
        tokens = tokens_doc.to_dict().get("tokens") or []
        if not tokens:
            continue

        for token in tokens:
            try:
                messaging.send(
                    messaging.Message(
                        data={
                            "type": "friend_activity",
                            "activity_type": activity_type,
                            "event_id": event_id,
                            "actor_name": actor_name,
                            "event_name": event_name,
                        },
                        notification=messaging.Notification(title=title, body=body),
                        token=token,
                        android=messaging.AndroidConfig(
                            notification=messaging.AndroidNotification(
                                channel_id="friend_activity",
                            ),
                        ),
                        apns=messaging.APNSConfig(
                            payload=messaging.APNSPayload(
                                aps=messaging.Aps(sound="default", badge=1),
                            ),
                        ),
                    )
                )
                total_sent += 1
            except messaging.UnregisteredError:
                stale_tokens_by_uid.setdefault(uid, []).append(token)
            except Exception as e:
                print(f"FCM send failed for uid={uid}: {e}")

    for uid, stale in stale_tokens_by_uid.items():
        try:
            db.collection(FCM_TOKENS_COLLECTION).document(uid).update({
                "tokens": firestore.ArrayRemove(stale),
            })
        except Exception as e:
            print(f"Failed to clean stale tokens for uid={uid}: {e}")

    print(f"Sent {total_sent} notifications for activity '{activity_type}' by {actor_name}")
