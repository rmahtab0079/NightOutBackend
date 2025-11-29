from fastapi import FastAPI, HTTPException, Header, Request
from starlette.middleware.base import BaseHTTPMiddleware
import random
from pydantic import BaseModel
from typing import Optional, List
from dotenv import load_dotenv
import requests
import json
import os
from datetime import datetime

# Firebase Admin SDK imports
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import auth as firebase_auth

app = FastAPI()


class SimpleLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        method = request.method
        try:
            response = await call_next(request)
            try:
                # Avoid reading body for large responses
                body_preview = await response.body()
            except Exception:
                body_preview = b""
            print(f"{method} {path} -> {response.status_code}")
            return response
        except Exception as e:
            print(f"{method} {path} -> error: {e}")
            raise


app.add_middleware(SimpleLoggingMiddleware)

# Load environment from mounted secret path if provided, else from local .env
dotenv_path = os.getenv("DOTENV_PATH")
if dotenv_path and os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
else:
    load_dotenv()

from models.movies import get_movies
from models.tv import get_tv
from models.process_tv_and_movies import find_similar_asset

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
    movieIds: List[int] = []
    tvShowIds: List[int] = []
    activities: List[str] = []
    cuisines: List[str] = []
    dietaryRestrictions: List[str] = []


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


def _load_service_account_project_id(path: str | None) -> str | None:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("project_id")
    except Exception:
        return None


def _infer_bucket_name() -> str | None:
    # Priority: explicit env -> derive from FIREBASE_PROJECT_ID/GOOGLE_CLOUD_PROJECT -> derive from serviceAccount.json
    explicit = os.getenv("FIREBASE_STORAGE_BUCKET")
    if explicit:
        return explicit

    project_id = os.getenv("FIREBASE_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        project_id = _load_service_account_project_id(os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH"))

    if not project_id:
        return None

    # Try common default bucket domain patterns
    candidates = [
        f"{project_id}.appspot.com",
        f"{project_id}.firebasestorage.app",
    ]
    # Return first candidate; existence is checked later
    return candidates[0]


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

@app.get("/find_similar_movie")
def find_similar_movie_from_storage():
    movies = {278, 238, 240, 424}
    similar_movie = find_similar_asset(movies, "movie")
    return similar_movie


@app.get("/movies")
def movies():
    return get_movies()

@app.get("/tv")
def tv():
    return get_tv()


@app.get("/get_user_preferences")
async def get_user_preferences(authorization: Optional[str] = Header(default=None)):
    initialize_firebase_if_needed()
    if firebase_db is None:
        raise HTTPException(
            status_code=501,
            detail=(
                "Firebase not configured. Set FIREBASE_SERVICE_ACCOUNT_PATH and "
                "FIREBASE_PROJECT_ID environment variables to enable writes."
            ),
        )

    # Verify user and get email from token
    decoded = _verify_and_get_user(authorization)
    email_from_token = decoded.get("email")

    if not email_from_token:
        raise HTTPException(status_code=400, detail="Email not found in token")

    try:
        collection_ref = firebase_db.collection("user_preferences")
        doc_ref = collection_ref.document(email_from_token)  # Use email as the document key
        doc = doc_ref.get()

        if doc.exists:
            return doc.to_dict()
        else:
            raise HTTPException(status_code=404, detail="User preferences not found")
    except HTTPException:
        raise
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

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


@app.get("/get_movie_suggestion")
async def get_recommended_movie(authorization: Optional[str] = Header(default=None)):
    user_preferences = await get_user_preferences(authorization)
    movie_ids = user_preferences["movieIds"]
    movie_id_set = set(movie_ids)
    similar_movie = find_similar_asset(movie_id_set, "movie")

    return similar_movie

@app.get("/get_tv_suggestion")
async def get_recommended_tv(authorization: Optional[str] = Header(default=None)):
    user_preferences = await get_user_preferences(authorization)
    tv_ids = user_preferences["tvShowIds"]
    tv_id_set = set(tv_ids)
    similar_movie = find_similar_asset(tv_id_set, "tv")

    return similar_movie

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


@app.get("/random", response_model=Suggestion)
async def get_random_suggestion_get(latitude: float, longitude: float):
    location = LocationData(latitude=latitude, longitude=longitude)
    print(f"Received location data (GET): {location}")
    nearby_places = search_nearby_places(location)
    print(nearby_places)
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



