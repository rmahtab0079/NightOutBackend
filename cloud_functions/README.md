# Friend Request Notification (Pub/Sub → FCM)

This Cloud Function subscribes to the `friend-requests` Pub/Sub topic and sends FCM (push) notifications to the recipient so they can open the app and accept or decline.

## Prerequisites

1. **Create Pub/Sub topic** (if not exists):
   ```bash
   gcloud pubsub topics create friend-requests --project=YOUR_PROJECT_ID
   ```

2. **Backend** must set `PUBSUB_FRIEND_REQUEST_TOPIC=friend-requests` and have `google-cloud-pubsub` publish when a friend request is created.

3. **Firestore** collection `user_fcm_tokens`: document ID = user UID, field `tokens` = array of FCM device tokens. The client registers tokens via `POST /fcm_token`.

## Deploy (Google Cloud Functions 2nd gen)

```bash
cd cloud_functions/friend_request_notify
gcloud functions deploy friend_request_notify \
  --gen2 \
  --runtime=python311 \
  --region=us-central1 \
  --source=. \
  --entry-point=friend_request_notify \
  --trigger-topic=friend-requests \
  --set-env-vars=FIREBASE_SERVICE_ACCOUNT_PATH=/path/to/serviceAccount.json
```

Or use a build that injects the service account via Secret Manager.

## Message payload (from backend)

The backend publishes JSON:

```json
{
  "to_uid": "recipient-firebase-uid",
  "from_uid": "sender-uid",
  "from_display_name": "Alice",
  "request_id": "firestore-doc-id"
}
```

The function looks up FCM tokens for `to_uid` in `user_fcm_tokens/{to_uid}` and sends a data + notification message so the app can open the friend requests screen.
