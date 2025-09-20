from fastapi import FastAPI, HTTPException, Header
import random
from pydantic import BaseModel
from typing import Optional, List
from dotenv import load_dotenv
import requests
import os
from datetime import datetime

# Firebase Admin SDK imports
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import auth as firebase_auth

app = FastAPI()

# Load environment from mounted secret path if provided, else from local .env
dotenv_path = os.getenv("DOTENV_PATH")
if dotenv_path and os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
else:
    load_dotenv()

from models.movies import get_movies_tmdb

api_key = "AIzaSyDW0X1gO6uVSPkYIa3R6sjRwNQrz-afYU0"

places_of_interest = ["hiking_area", "restaurant", "bar", "cafe", "coffee_shop", "beach",
                      "historical_landmark", "movie_theater", "video_arcade", "karaoke", "night_club", "opera_house",
                      "dance_hall", "amusement_park"]

# Model for location data
class LocationData(BaseModel):
    latitude: float
    longitude: float
    accuracy: Optional[float] = None
    altitude: Optional[float] = None
    speed: Optional[float] = None
    speedAccuracy: Optional[float] = None
    heading: Optional[float] = None
    timestamp: Optional[str] = None


class Suggestion(BaseModel):
    suggestion: str

# User Preferences payload model (matches Flutter `UserPreferences.toJson()`)
class UserPreferences(BaseModel):
    uid: Optional[str] = None
    email: Optional[str] = None
    age: Optional[int] = None
    movies: List[str] = []
    activities: List[str] = []
    cuisines: List[str] = []
    dietaryRestrictions: List[str] = []
    drinksAlcohol: Optional[bool] = None


# Initialize Firebase Admin (with placeholders)
FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "nightoutclient-7931e")
FIREBASE_SERVICE_ACCOUNT_PATH = os.getenv(
    "FIREBASE_SERVICE_ACCOUNT_PATH", "serviceAccount.json"
)

firebase_db = None

def initialize_firebase_if_needed() -> None:
    global firebase_db
    if firebase_db is not None:
        return
    try:
        if not firebase_admin._apps:  # type: ignore[attr-defined]
            # Try initializing with explicit Service Account path (placeholder by default)
            if os.path.exists(FIREBASE_SERVICE_ACCOUNT_PATH):
                cred = credentials.Certificate(FIREBASE_SERVICE_ACCOUNT_PATH)
                firebase_admin.initialize_app(cred, {"projectId": FIREBASE_PROJECT_ID})
            else:
                # Fallback to Application Default Credentials if available
                firebase_admin.initialize_app()
        firebase_db = firestore.client()
    except Exception:
        # Leave firebase_db as None if not configured; endpoints will report 501
        firebase_db = None


def _verify_and_get_user(authorization: Optional[str]) -> dict:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.split(" ", 1)[1].strip()
    try:
        decoded = firebase_auth.verify_id_token(token)
        return decoded
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid ID token: {e}")


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

def get_user_preferences():
    initialize_firebase_if_needed()
    if firebase_db is None:
        raise HTTPException(
            status_code=501,
            detail=(
                "Firebase not configured. Set FIREBASE_SERVICE_ACCOUNT_PATH and "
                "FIREBASE_PROJECT_ID environment variables to enable writes."
            ),
        )

@app.get("/tmdb")
def call_tmdb():
    return get_movies_tmdb()

@app.post("/user_preferences")
async def submit_user_preferences(preferences: UserPreferences, authorization: Optional[str] = Header(default=None)):
    initialize_firebase_if_needed()
    if firebase_db is None:
        raise HTTPException(
            status_code=501,
            detail=(
                "Firebase not configured. Set FIREBASE_SERVICE_ACCOUNT_PATH and "
                "FIREBASE_PROJECT_ID environment variables to enable writes."
            ),
        )

    # Verify user and enforce UID/email from token
    decoded = _verify_and_get_user(authorization)
    uid = decoded.get("uid")
    email_from_token = decoded.get("email")

    payload = preferences.model_dump(exclude_none=True)
    payload["uid"] = uid
    if email_from_token:
        payload["email"] = email_from_token
    payload["createdAt"] = datetime.utcnow().isoformat() + "Z"

    try:
        collection_ref = firebase_db.collection("user_preferences")
        doc_ref = collection_ref.document(email_from_token)  # Use email as the document key
        doc_ref.set(payload)  # Overwrite on every call
        return {"message": "User preferences submitted", "documentId": doc_ref.id}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/random", response_model=Suggestion)
async def get_random_suggestion(location: LocationData):
    print(f"Received location data: {location}")
    nearby_places = search_nearby_places(location)
    print(nearby_places)
    # random_suggestions = [
    #     "Take a walk at a Overpeck park",
    #     "Go watch Bellarina at AMC Ridgefield Park",
    #     "Go to the fair at the American Dream Mall",
    # ]
    #
    new_suggestion = "Visit " + random.choice(nearby_places)

    return Suggestion(suggestion=new_suggestion)

def search_nearby_places(location: LocationData) -> List[str]:
    endpoint_url = "https://places.googleapis.com/v1/places:searchNearby"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "places.displayName"
    }
    json_payload = {
        "includedTypes": places_of_interest,
        "maxResultCount": 20,
        "locationRestriction": {
            "circle": {
                "center": {
                    "latitude": location.latitude,
                    "longitude": location.longitude},
                "radius": 10000.0
            }
        }
    }

    try:
        response = requests.post(endpoint_url, json=json_payload, headers=headers)
        response_dict = response.json()
        places = response_dict["places"]
        results = []
        for place in places:
            print(place)
            results.append(place['displayName']['text'])

        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



