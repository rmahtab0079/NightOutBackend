import os


# Initialize Firebase Admin (with placeholders)
FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "nightoutclient-7931e")
FIREBASE_SERVICE_ACCOUNT_PATH = os.getenv(
    "FIREBASE_SERVICE_ACCOUNT_PATH", "serviceAccount.json"
)

firebase_db = None

