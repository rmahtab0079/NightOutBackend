"""
Cloud Function: subscribes to Pub/Sub topic 'hourly-picks' and sends FCM
push notifications with each user's top picks.

Trigger: Pub/Sub topic (hourly-picks).
Message payload: JSON with uid, title, body, data.
"""
import json
import base64
import os

import firebase_admin
from firebase_admin import credentials, firestore, messaging

FCM_TOKENS_COLLECTION = "user_fcm_tokens"
_app_initialized = False


def _get_firestore():
    """Initialize Firebase Admin pointed at the Firebase project (which owns
    Firestore + FCM), regardless of which GCP project this function runs in."""
    global _app_initialized
    if not _app_initialized:
        if not firebase_admin._apps:
            project_id = os.getenv("FIREBASE_PROJECT_ID", "nightoutclient-7931e")
            svc_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH", "serviceAccount.json")
            if svc_path and os.path.exists(svc_path):
                cred = credentials.Certificate(svc_path)
                firebase_admin.initialize_app(cred, {"projectId": project_id})
            else:
                print(f"WARNING: serviceAccount.json not found at {svc_path}; falling back to ADC")
                firebase_admin.initialize_app(options={"projectId": project_id})
        _app_initialized = True
    return firestore.client()


def hourly_picks_notify(event, context):
    """
    Triggered by a Pub/Sub message on the hourly-picks topic.
    Decodes the payload and sends an FCM notification to the user's devices.
    """
    try:
        raw = base64.b64decode(event["data"]).decode("utf-8")
        payload = json.loads(raw)
    except Exception as e:
        print(f"Failed to decode message: {e}")
        return

    uid = payload.get("uid")
    title = payload.get("title", "Your hourly picks are ready!")
    body = payload.get("body", "Check out today's suggestions")
    data = payload.get("data", {})

    if not uid:
        print("Missing uid in payload")
        return

    db = _get_firestore()
    tokens_doc = db.collection(FCM_TOKENS_COLLECTION).document(uid).get()
    if not tokens_doc.exists:
        print(f"No FCM tokens for uid={uid}")
        return

    tokens = tokens_doc.to_dict().get("tokens") or []
    if not tokens:
        print(f"Empty tokens list for uid={uid}")
        return

    str_data = {k: str(v) for k, v in data.items()}

    stale_tokens: list[str] = []
    sent = 0

    for token in tokens:
        try:
            messaging.send(
                messaging.Message(
                    data=str_data,
                    notification=messaging.Notification(title=title, body=body),
                    token=token,
                    android=messaging.AndroidConfig(
                        notification=messaging.AndroidNotification(
                            channel_id="hourly_picks",
                        ),
                    ),
                    apns=messaging.APNSConfig(
                        payload=messaging.APNSPayload(
                            aps=messaging.Aps(sound="default", badge=1),
                        ),
                    ),
                )
            )
            sent += 1
        except messaging.UnregisteredError:
            stale_tokens.append(token)
        except Exception as e:
            print(f"FCM send failed for token: {e}")

    if stale_tokens:
        try:
            db.collection(FCM_TOKENS_COLLECTION).document(uid).update({
                "tokens": firestore.ArrayRemove(stale_tokens),
            })
            print(f"Removed {len(stale_tokens)} stale tokens for uid={uid}")
        except Exception as e:
            print(f"Failed to clean stale tokens: {e}")

    print(f"Sent {sent}/{len(tokens)} notifications for uid={uid}")
