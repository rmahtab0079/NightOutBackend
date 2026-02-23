"""
Cloud Function: subscribes to Pub/Sub topic 'friend-requests' and sends FCM
notifications to the recipient so they can accept or decline the friend request.

Trigger: Pub/Sub topic (friend-requests).
Message payload: JSON with to_uid, from_uid, from_display_name, request_id.
"""
import json
import os
import base64

import firebase_admin
from firebase_admin import credentials, firestore, messaging

# Firestore collection where we store FCM tokens per user (document id = uid)
FCM_TOKENS_COLLECTION = "user_fcm_tokens"

_app_initialized = False


def _get_firestore():
    global _app_initialized
    if not _app_initialized:
        if not firebase_admin._apps:
            project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("FIREBASE_PROJECT_ID")
            if os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH") and os.path.exists(os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH", "")):
                cred = credentials.Certificate(os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH"))
                firebase_admin.initialize_app(cred, {"projectId": project_id})
            else:
                firebase_admin.initialize_app()
        _app_initialized = True
    return firestore.client()


def friend_request_notify(event, context):
    """
    Triggered by Pub/Sub message. Decodes payload and sends FCM to recipient.
    """
    try:
        data = base64.b64decode(event["data"]).decode("utf-8")
        payload = json.loads(data)
    except Exception as e:
        print(f"Failed to decode message: {e}")
        return

    to_uid = payload.get("to_uid")
    from_display_name = payload.get("from_display_name") or "Someone"
    request_id = payload.get("request_id", "")

    if not to_uid:
        print("Missing to_uid in payload")
        return

    db = _get_firestore()
    tokens_doc = db.collection(FCM_TOKENS_COLLECTION).document(to_uid).get()
    if not tokens_doc.exists:
        print(f"No FCM tokens for uid={to_uid}")
        return

    tokens = tokens_doc.to_dict().get("tokens") or []
    if not tokens:
        print(f"Empty tokens list for uid={to_uid}")
        return

    title = "Friend request"
    body = f"{from_display_name} wants to be your friend"

    for token in tokens:
        try:
            messaging.send(
                messaging.Message(
                    data={
                        "type": "friend_request",
                        "request_id": request_id,
                        "from_display_name": from_display_name,
                    },
                    notification=messaging.Notification(title=title, body=body),
                    token=token,
                )
            )
        except Exception as e:
            print(f"FCM send failed for token: {e}")
