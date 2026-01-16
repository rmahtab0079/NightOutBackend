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
from models.write_movies_csv import get_movie_genres, get_tv_genres
from service.cache_database_layer import (
    get_similar_assets,
    clear_memory_cache,
    clear_expired_firestore_cache,
    get_cache_stats
)

api_key = "AIzaSyDW0X1gO6uVSPkYIa3R6sjRwNQrz-afYU0"

places_of_interest = ["hiking_area", "restaurant", "bar", "cafe", "coffee_shop", "beach",
                      "historical_landmark", "movie_theater", "video_arcade", "karaoke", "night_club", "opera_house",
                      "dance_hall", "amusement_park"]

movie_type = "movie"
tv_type = "tv"

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
    streamingServices: List[str] = []

class SearchAssetParams(BaseModel):
    start_year: int = 1970
    end_year: int = 2026
    genres: List[int] = []  # Genre IDs to filter by (empty = all genres)


class SimilarAssetParams(BaseModel):
    asset_ids: List[int]  # IDs of movies or TV shows to find similar content for
    asset_type: str = "movie"  # 'movie' or 'tv'
    genres: List[int] = []  # Optional genre filter
    start_year: Optional[int] = None  # Optional start year filter
    end_year: Optional[int] = None  # Optional end year filter
    top_n: int = 5  # Number of recommendations to return
    bypass_cache: bool = False  # If True, skip cache and fetch fresh results

class UpdatePreferenceRequest(BaseModel):
    key: str  # Firestore document key (email)
    kind: str  # 'movies' or 'tv'
    itemId: int  # ID to add


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


@app.post("/movies")
def movies(params: SearchAssetParams):
    return get_movies(start_year=params.start_year, end_year=params.end_year, genres=params.genres)

@app.post("/tv")
def tv(params: SearchAssetParams):
    tv_show = get_tv(start_year=params.start_year, end_year=params.end_year, genres=params.genres)
    print(tv_show)
    return tv_show


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

@app.post("/user_preferences/update")
async def update_user_preferences(payload: UpdatePreferenceRequest, authorization: Optional[str] = Header(default=None)):
    initialize_firebase_if_needed()
    if firebase_db is None:
        raise HTTPException(
            status_code=501,
            detail=(
                "Firebase not configured. Set FIREBASE_SERVICE_ACCOUNT_PATH and "
                "FIREBASE_PROJECT_ID environment variables to enable writes."
            ),
        )

    # Verify user; enforce that caller matches the document key
    decoded = _verify_and_get_user(authorization)
    email_from_token = decoded.get("email")
    if not email_from_token:
        raise HTTPException(status_code=400, detail="Email not found in token")
    if email_from_token != payload.key:
        raise HTTPException(status_code=403, detail="Not authorized to modify this user preferences")

    field_name = None
    if payload.kind.lower() == "movies":
        field_name = "movieIds"
    elif payload.kind.lower() == "tv":
        field_name = "tvShowIds"
    else:
        raise HTTPException(status_code=400, detail="Invalid kind. Use 'movies' or 'tv'.")

    try:
        collection_ref = firebase_db.collection("user_preferences")
        doc_ref = collection_ref.document(payload.key)
        doc = doc_ref.get()
        if not doc.exists:
            raise HTTPException(status_code=404, detail="User preferences document not found")

        # Use Firestore ArrayUnion to add if not present
        doc_ref.update({field_name: firestore.ArrayUnion([payload.itemId])})
        return {"message": "User preferences updated", "field": field_name, "id": payload.itemId}
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


@app.get('/movie_genres')
def movie_genres():
    return get_movie_genres()

@app.get('/tv_genres')
def tv_genres():
    return get_tv_genres()


# ============================================================================
# CACHED SIMILAR ASSETS ENDPOINTS
# ============================================================================

@app.post("/similar_assets")
def find_similar_assets_cached(params: SimilarAssetParams):
    """
    Find similar movies or TV shows with caching support.
    
    This endpoint uses a multi-layer caching strategy:
    1. In-memory cache (1 hour TTL)
    2. Firestore database for persistent storage
    3. API fallback via find_similar_asset_v2
    
    Request body:
    - asset_ids: List of movie or TV show IDs to find similar content for
    - asset_type: 'movie' or 'tv' (default: 'movie')
    - genres: Optional list of genre IDs to filter results
    - start_year: Optional minimum release year filter
    - end_year: Optional maximum release year filter
    - top_n: Number of recommendations to return (default: 5)
    - bypass_cache: If True, skip cache and fetch fresh results
    """
    if not params.asset_ids:
        raise HTTPException(status_code=400, detail="asset_ids must not be empty")
    
    if params.asset_type not in ["movie", "tv"]:
        raise HTTPException(status_code=400, detail="asset_type must be 'movie' or 'tv'")
    
    try:
        results = get_similar_assets(
            asset_ids=set(params.asset_ids),
            asset_type=params.asset_type,
            genres=params.genres if params.genres else None,
            start_year=params.start_year,
            end_year=params.end_year,
            top_n=params.top_n,
            read_from_local=False,
            bypass_cache=params.bypass_cache
        )
        return {"results": results, "count": len(results)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error finding similar assets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get_movie_suggestion_v2")
async def get_recommended_movie_v2(
    authorization: Optional[str] = Header(default=None),
    genres: Optional[str] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    top_n: int = 5,
    bypass_cache: bool = False
):
    """
    Get movie recommendations based on user preferences with caching and filtering.
    
    Query params:
    - genres: Comma-separated genre IDs (e.g., "28,12,878")
    - start_year: Minimum release year filter
    - end_year: Maximum release year filter
    - top_n: Number of recommendations (default: 5)
    - bypass_cache: Skip cache for fresh results
    """
    user_preferences = await get_user_preferences(authorization)
    movie_ids = user_preferences.get("movieIds", [])
    
    if not movie_ids:
        raise HTTPException(status_code=400, detail="No movie preferences found for user")
    
    # Parse genres from comma-separated string
    genre_list = None
    if genres:
        try:
            genre_list = [int(g.strip()) for g in genres.split(",")]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid genres format. Use comma-separated integers.")
    
    try:
        results = get_similar_assets(
            asset_ids=set(movie_ids),
            asset_type="movie",
            genres=genre_list,
            start_year=start_year,
            end_year=end_year,
            top_n=top_n,
            read_from_local=False,
            bypass_cache=bypass_cache
        )
        return {"results": results, "count": len(results)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error getting movie suggestions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get_tv_suggestion_v2")
async def get_recommended_tv_v2(
    authorization: Optional[str] = Header(default=None),
    genres: Optional[str] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    top_n: int = 5,
    bypass_cache: bool = False
):
    """
    Get TV show recommendations based on user preferences with caching and filtering.
    
    Query params:
    - genres: Comma-separated genre IDs (e.g., "18,35,10765")
    - start_year: Minimum first air date year filter
    - end_year: Maximum first air date year filter
    - top_n: Number of recommendations (default: 5)
    - bypass_cache: Skip cache for fresh results
    """
    user_preferences = await get_user_preferences(authorization)
    tv_ids = user_preferences.get("tvShowIds", [])
    
    if not tv_ids:
        raise HTTPException(status_code=400, detail="No TV show preferences found for user")
    
    # Parse genres from comma-separated string
    genre_list = None
    if genres:
        try:
            genre_list = [int(g.strip()) for g in genres.split(",")]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid genres format. Use comma-separated integers.")
    
    try:
        results = get_similar_assets(
            asset_ids=set(tv_ids),
            asset_type="tv",
            genres=genre_list,
            start_year=start_year,
            end_year=end_year,
            top_n=top_n,
            read_from_local=False,
            bypass_cache=bypass_cache
        )
        return {"results": results, "count": len(results)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error getting TV suggestions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cache/stats")
def cache_statistics():
    """Get cache statistics including memory and Firestore cache counts."""
    return get_cache_stats()


@app.post("/cache/clear/memory")
def clear_memory_cache_endpoint():
    """Clear the in-memory cache. Returns number of entries cleared."""
    count = clear_memory_cache()
    return {"cleared_entries": count, "cache": "memory"}


@app.post("/cache/clear/expired")
def clear_expired_cache_endpoint():
    """Clear expired entries from Firestore cache. Returns number of entries deleted."""
    count = clear_expired_firestore_cache()
    return {"cleared_entries": count, "cache": "firestore"}
