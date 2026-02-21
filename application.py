from fastapi import FastAPI, HTTPException, Header, Request
from starlette.middleware.base import BaseHTTPMiddleware
import random
from pydantic import BaseModel
from typing import Optional, List, Set
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
from models.process_tv_and_movies import find_similar_asset, get_movie_detail
from models.write_movies_csv import get_movie_genres, get_tv_genres
from service.cache_database_layer import (
    get_similar_assets,
    clear_memory_cache,
    clear_expired_firestore_cache,
    get_cache_stats
)
from service.user_genre_recommendations import (
    MOVIE_RECS_COLLECTION,
    TV_RECS_COLLECTION,
    precompute_all_user_genre_recommendations,
    precompute_user_genre_recommendations_for_email,
    start_genre_recommendation_scheduler,
)
from service.wiki_plot_job import generate_and_upload_plots

# Start background genre recommendation job (runs every 5 minutes)
if os.getenv("ENABLE_GENRE_RECS_JOB", "true").lower() == "true":
    @app.on_event("startup")
    async def _start_genre_recs_job():
        start_genre_recommendation_scheduler(interval_seconds=300)

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
    dietaryPreferences: List[str] = []
    cuisines: List[str] = []

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


class AddAssetRequest(BaseModel):
    asset_type: str  # 'movie' or 'tv'
    asset_id: int  # ID of the movie or TV show


class RecomputeGenreRecsRequest(BaseModel):
    email: Optional[str] = None
    asset_type: Optional[str] = None  # 'movie', 'tv', or None for both
    limit_per_genre: int = 100


class PlotRebuildRequest(BaseModel):
    asset_type: str = "all"  # 'movies', 'tv', or 'all'
    max_items: Optional[int] = None
    read_from_storage: bool = True


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


def _verify_admin_token(admin_token: Optional[str]) -> None:
    expected = os.getenv("ADMIN_RECS_TOKEN")
    if not expected:
        raise HTTPException(status_code=501, detail="ADMIN_RECS_TOKEN is not configured")
    if not admin_token or admin_token != expected:
        raise HTTPException(status_code=403, detail="Invalid admin token")


def _get_precomputed_genre_ids(
    email: str,
    asset_type: str,
    genre_ids: List[int],
) -> List[int]:
    """
    Fetch precomputed genre recommendations for a user.
    Returns a deduped list of asset IDs for the requested genres.
    """
    initialize_firebase_if_needed()
    if firebase_db is None:
        return []

    collection = MOVIE_RECS_COLLECTION if asset_type == "movie" else TV_RECS_COLLECTION
    doc = firebase_db.collection(collection).document(email).get()
    if not doc.exists:
        return []

    data = doc.to_dict() or {}
    genre_map = data.get("genres", {})

    ordered_ids: List[int] = []
    seen: Set[int] = set()
    for gid in genre_ids:
        ids = genre_map.get(str(gid), [])
        for asset_id in ids:
            try:
                aid = int(asset_id)
            except Exception:
                continue
            if aid not in seen:
                seen.add(aid)
                ordered_ids.append(aid)

    return ordered_ids


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


class NightInSuggestionRequest(BaseModel):
    party_size: int
    genres: List[int] = []


class NightOutSuggestionRequest(BaseModel):
    party_size: int
    budget: Optional[str] = None
    latitude: float
    longitude: float
    radius_meters: float = 10000.0
    dietary_preferences: List[str] = []
    cuisines: List[str] = []
    excluded_names: List[str] = []


BUDGET_TO_MAX_PRICE_LEVEL = {
    "Free": 0,
    "Under $20": 1,
    "Under $50": 2,
    "Under $100": 3,
    "$100+": 4,
}


@app.get("/night_in_genres")
def night_in_genres():
    """Return movie + TV genres merged and deduplicated by name."""
    movie_map = get_movie_genres()   # {id: name}
    tv_map = get_tv_genres()         # {id: name}
    seen_names: dict[str, int] = {}
    for gid, name in {**movie_map, **tv_map}.items():
        gid_int = int(gid) if isinstance(gid, str) else gid
        if name not in seen_names:
            seen_names[name] = gid_int
    return {str(v): k for k, v in seen_names.items()}


@app.post("/night_in_suggestion")
async def get_night_in_suggestion(req: NightInSuggestionRequest):
    from datetime import date
    current_year = date.today().year

    asset_type = random.choice(["movie", "tv"])
    if asset_type == "movie":
        data = get_movies(start_year=2000, end_year=current_year, genres=req.genres)
    else:
        data = get_tv(start_year=2000, end_year=current_year, genres=req.genres)

    results = data.get("results", [])
    if not results:
        raise HTTPException(status_code=404, detail="No suggestions found")

    pick = random.choice(results)
    pick["asset_type"] = asset_type

    try:
        from service.watch_providers_job import get_providers_for_asset
        providers = get_providers_for_asset(pick["id"], asset_type, read_from_local=False)
        pick["watch_providers"] = providers
    except Exception as e:
        print(f"Could not fetch watch providers: {e}")
        pick["watch_providers"] = []

    return pick


CUISINE_TO_PLACE_TYPES = {
    "Italian": ["italian_restaurant"],
    "Chinese": ["chinese_restaurant"],
    "Japanese": ["japanese_restaurant", "sushi_restaurant", "ramen_restaurant"],
    "Mexican": ["mexican_restaurant"],
    "Indian": ["indian_restaurant"],
    "Thai": ["thai_restaurant"],
    "American": ["american_restaurant", "hamburger_restaurant", "steak_house"],
    "Mediterranean": ["mediterranean_restaurant", "greek_restaurant", "lebanese_restaurant"],
    "Korean": ["korean_restaurant"],
    "French": ["french_restaurant"],
    "Vietnamese": ["vietnamese_restaurant"],
    "Caribbean": ["caribbean_restaurant"],
    "Middle Eastern": ["middle_eastern_restaurant"],
}


@app.post("/night_out_suggestion")
async def get_night_out_suggestion(req: NightOutSuggestionRequest):
    included_types = list(places_of_interest)
    if req.cuisines:
        cuisine_types = []
        for c in req.cuisines:
            cuisine_types.extend(CUISINE_TO_PLACE_TYPES.get(c, []))
        if cuisine_types:
            included_types = cuisine_types

    location = LocationData(latitude=req.latitude, longitude=req.longitude)
    excluded = set(req.excluded_names)
    max_price = BUDGET_TO_MAX_PRICE_LEVEL.get(req.budget)

    MAX_RADIUS = 50000.0
    current_radius = req.radius_meters

    while current_radius <= MAX_RADIUS:
        places = search_nearby_places(location, radius=current_radius, included_types=included_types)

        if excluded:
            places = [p for p in places if p["name"] not in excluded]

        if max_price is not None:
            places = [
                p for p in places
                if p.get("price_level") is None or p["price_level"] <= max_price
            ]

        if places:
            break

        current_radius *= 2
        if current_radius > MAX_RADIUS:
            raise HTTPException(status_code=404, detail="No places found nearby")

    pick = random.choice(places)
    pick["party_size"] = req.party_size
    pick["budget"] = req.budget
    pick["radius_used"] = current_radius
    return pick

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


@app.post("/user_preferences/add_watched")
async def add_watched_asset(payload: AddAssetRequest, authorization: Optional[str] = Header(default=None)):
    """
    Add a movie or TV show to the user's watched list.
    - For movies: adds to 'movieIds' field
    - For TV shows: adds to 'tvShowIds' field
    """
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

    # Determine field name based on asset type
    if payload.asset_type.lower() == "movie":
        field_name = "movieIds"
    elif payload.asset_type.lower() == "tv":
        field_name = "tvShowIds"
    else:
        raise HTTPException(status_code=400, detail="Invalid asset_type. Use 'movie' or 'tv'.")

    try:
        collection_ref = firebase_db.collection("user_preferences")
        doc_ref = collection_ref.document(email_from_token)
        doc = doc_ref.get()
        
        if not doc.exists:
            # Create the document if it doesn't exist
            doc_ref.set({
                "email": email_from_token,
                field_name: [payload.asset_id],
                "createdAt": datetime.utcnow().isoformat() + "Z"
            })
        else:
            # Use Firestore ArrayUnion to add if not present
            doc_ref.update({field_name: firestore.ArrayUnion([payload.asset_id])})
        
        return {
            "message": "Asset added to watched list",
            "field": field_name,
            "asset_id": payload.asset_id
        }
    except HTTPException:
        raise
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/user_preferences/add_disliked")
async def add_disliked_asset(payload: AddAssetRequest, authorization: Optional[str] = Header(default=None)):
    """
    Add a movie or TV show to the user's disliked list.
    - For movies: adds to 'dislikedMovieIds' field
    - For TV shows: adds to 'dislikedTvShowIds' field
    
    These disliked items should be excluded from future recommendations.
    """
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

    # Determine field name based on asset type
    if payload.asset_type.lower() == "movie":
        field_name = "dislikedMovieIds"
    elif payload.asset_type.lower() == "tv":
        field_name = "dislikedTvShowIds"
    else:
        raise HTTPException(status_code=400, detail="Invalid asset_type. Use 'movie' or 'tv'.")

    try:
        collection_ref = firebase_db.collection("user_preferences")
        doc_ref = collection_ref.document(email_from_token)
        doc = doc_ref.get()
        
        if not doc.exists:
            # Create the document if it doesn't exist
            doc_ref.set({
                "email": email_from_token,
                field_name: [payload.asset_id],
                "createdAt": datetime.utcnow().isoformat() + "Z"
            })
        else:
            # Use Firestore ArrayUnion to add if not present
            doc_ref.update({field_name: firestore.ArrayUnion([payload.asset_id])})
        
        return {
            "message": "Asset added to disliked list",
            "field": field_name,
            "asset_id": payload.asset_id
        }
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
    nearby_places = search_nearby_places(location)
    if not nearby_places:
        raise HTTPException(status_code=404, detail="No places found nearby")
    pick = random.choice(nearby_places)
    return Suggestion(suggestion="Visit " + pick["name"])


@app.get("/random", response_model=Suggestion)
async def get_random_suggestion_get(latitude: float, longitude: float):
    location = LocationData(latitude=latitude, longitude=longitude)
    nearby_places = search_nearby_places(location)
    if not nearby_places:
        raise HTTPException(status_code=404, detail="No places found nearby")
    pick = random.choice(nearby_places)
    return Suggestion(suggestion="Visit " + pick["name"])


def search_nearby_places(
    location: LocationData,
    radius: float = 10000.0,
    included_types: List[str] | None = None,
) -> List[dict]:
    endpoint_url = "https://places.googleapis.com/v1/places:searchNearby"
    field_mask = ",".join([
        "places.displayName",
        "places.formattedAddress",
        "places.rating",
        "places.userRatingCount",
        "places.photos",
        "places.primaryType",
        "places.editorialSummary",
        "places.priceLevel",
    ])
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": field_mask,
    }
    json_payload = {
        "includedTypes": included_types or places_of_interest,
        "maxResultCount": 20,
        "locationRestriction": {
            "circle": {
                "center": {
                    "latitude": location.latitude,
                    "longitude": location.longitude},
                "radius": min(max(radius, 100.0), 50000.0)
            }
        }
    }

    try:
        response = requests.post(endpoint_url, json=json_payload, headers=headers)
        response_dict = response.json()
        places = response_dict.get("places", [])
        results = []
        for place in places:
            name = place.get("displayName", {}).get("text", "Unknown")
            photo_url = None
            photos = place.get("photos", [])
            if photos:
                photo_ref = photos[0].get("name", "")
                if photo_ref:
                    photo_url = (
                        f"https://places.googleapis.com/v1/{photo_ref}/media"
                        f"?maxWidthPx=800&key={api_key}"
                    )
            price_str = place.get("priceLevel")
            price_int = {
                "PRICE_LEVEL_FREE": 0,
                "PRICE_LEVEL_INEXPENSIVE": 1,
                "PRICE_LEVEL_MODERATE": 2,
                "PRICE_LEVEL_EXPENSIVE": 3,
                "PRICE_LEVEL_VERY_EXPENSIVE": 4,
            }.get(price_str)

            results.append({
                "name": name,
                "address": place.get("formattedAddress"),
                "rating": place.get("rating"),
                "user_rating_count": place.get("userRatingCount"),
                "photo_url": photo_url,
                "type": place.get("primaryType"),
                "summary": place.get("editorialSummary", {}).get("text"),
                "price_level": price_int,
            })
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
    offset: int = 0,
    bypass_cache: bool = False
):
    """
    Get movie recommendations based on user preferences with caching and filtering.
    
    Query params:
    - genres: Comma-separated genre IDs (e.g., "28,12,878")
    - start_year: Minimum release year filter
    - end_year: Maximum release year filter
    - top_n: Number of recommendations (default: 5)
    - offset: Number of results to skip for pagination (default: 0)
    - bypass_cache: Skip cache for fresh results
    
    Disliked movies (from dislikedMovieIds field) are automatically excluded.
    """
    user_preferences = await get_user_preferences(authorization)
    movie_ids = user_preferences.get("movieIds", [])
    
    if not movie_ids:
        raise HTTPException(status_code=400, detail="No movie preferences found for user")
    
    # Get disliked movie IDs to exclude from recommendations
    disliked_movie_ids = user_preferences.get("dislikedMovieIds", [])
    excluded_ids = set(disliked_movie_ids) if disliked_movie_ids else None
    
    # Parse genres from comma-separated string
    genre_list = None
    if genres:
        try:
            genre_list = [int(g.strip()) for g in genres.split(",")]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid genres format. Use comma-separated integers.")
    if not genre_list:
        raise HTTPException(status_code=400, detail="At least one genre is required.")
    if start_year is None or end_year is None:
        raise HTTPException(status_code=400, detail="start_year and end_year are required.")
    
    try:
        # Prefer precomputed genre recommendations
        email = user_preferences.get("email")
        precomputed_ids = _get_precomputed_genre_ids(email, "movie", genre_list) if email else []

        if precomputed_ids:
            results = []
            skipped = 0
            for idx, movie_id in enumerate(precomputed_ids):
                if excluded_ids and movie_id in excluded_ids:
                    continue
                detail = get_movie_detail(movie_id, "movie")
                if not detail:
                    continue
                release_date = detail.get("release_date") or ""
                if release_date:
                    try:
                        year = int(release_date.split("-")[0])
                        if year < start_year or year > end_year:
                            continue
                    except Exception:
                        pass
                # Handle offset for pagination
                if skipped < offset:
                    skipped += 1
                    continue
                results.append({
                    "movie_id": movie_id,
                    "title": detail.get("title") or detail.get("original_title"),
                    "similarity": max(0.0, 1.0 - (idx / max(len(precomputed_ids), 1))),
                    "detail": detail,
                })
                if len(results) >= top_n:
                    break
            return {"results": results, "count": len(results), "offset": offset, "has_more": (skipped + len(results)) < len(precomputed_ids)}

        # Fallback to dynamic recommendation
        results = get_similar_assets(
            asset_ids=set(movie_ids),
            asset_type="movie",
            genres=genre_list,
            start_year=start_year,
            end_year=end_year,
            top_n=top_n + offset,
            read_from_local=False,
            bypass_cache=bypass_cache,
            excluded_ids=excluded_ids
        )
        # Apply offset to fallback results
        paginated_results = results[offset:offset + top_n] if offset < len(results) else []
        return {"results": paginated_results, "count": len(paginated_results), "offset": offset, "has_more": len(results) > (offset + top_n)}
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
    
    Disliked TV shows (from dislikedTvShowIds field) are automatically excluded.
    """
    user_preferences = await get_user_preferences(authorization)
    tv_ids = user_preferences.get("tvShowIds", [])
    
    if not tv_ids:
        raise HTTPException(status_code=400, detail="No TV show preferences found for user")
    
    # Get disliked TV show IDs to exclude from recommendations
    disliked_tv_ids = user_preferences.get("dislikedTvShowIds", [])
    excluded_ids = set(disliked_tv_ids) if disliked_tv_ids else None
    
    # Parse genres from comma-separated string
    genre_list = None
    if genres:
        try:
            genre_list = [int(g.strip()) for g in genres.split(",")]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid genres format. Use comma-separated integers.")
    
    try:
        # Prefer precomputed genre recommendations
        email = user_preferences.get("email")
        precomputed_ids = _get_precomputed_genre_ids(email, "tv", genre_list) if email else []

        if precomputed_ids:
            results = []
            for idx, tv_id in enumerate(precomputed_ids):
                if excluded_ids and tv_id in excluded_ids:
                    continue
                detail = get_movie_detail(tv_id, "tv")
                if not detail:
                    continue
                first_air_date = detail.get("first_air_date") or ""
                if first_air_date:
                    try:
                        year = int(first_air_date.split("-")[0])
                        if year < start_year or year > end_year:
                            continue
                    except Exception:
                        pass
                results.append({
                    "movie_id": tv_id,
                    "title": detail.get("name") or detail.get("original_name"),
                    "similarity": max(0.0, 1.0 - (idx / max(len(precomputed_ids), 1))),
                    "detail": detail,
                })
                if len(results) >= top_n:
                    break
            return {"results": results, "count": len(results)}

        # Fallback to dynamic recommendation
        results = get_similar_assets(
            asset_ids=set(tv_ids),
            asset_type="tv",
            genres=genre_list,
            start_year=start_year,
            end_year=end_year,
            top_n=top_n,
            read_from_local=False,
            bypass_cache=bypass_cache,
            excluded_ids=excluded_ids
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


@app.get("/watch_providers/{asset_type}/{asset_id}")
async def get_watch_providers(asset_type: str, asset_id: int):
    """
    Get watch providers for a movie or TV show.
    
    Returns streaming platforms where the content is available.
    Data is fetched from precomputed parquet files.
    
    Path params:
    - asset_type: 'movie' or 'tv'
    - asset_id: TMDB movie or TV show ID
    """
    if asset_type not in ("movie", "tv"):
        raise HTTPException(status_code=400, detail="asset_type must be 'movie' or 'tv'")
    
    try:
        from service.watch_providers_job import get_providers_for_asset
        providers = get_providers_for_asset(asset_id, asset_type, read_from_local=False)
        return {
            "asset_id": asset_id,
            "asset_type": asset_type,
            "providers": providers,
            "count": len(providers),
        }
    except Exception as e:
        print(f"Error fetching watch providers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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


@app.post("/user/refresh_recommendations")
async def refresh_user_recommendations(
    authorization: Optional[str] = Header(default=None),
    asset_type: Optional[str] = None,
    limit_per_genre: int = 100,
):
    """
    Refresh the authenticated user's genre recommendations.
    
    This triggers recomputation of precomputed recommendations for the current user.
    Use this after the user has viewed many movies to get fresh recommendations.
    
    Query params:
    - asset_type: 'movie', 'tv', or omitted for both
    - limit_per_genre: Number of recommendations per genre (default: 100)
    """
    user_preferences = await get_user_preferences(authorization)
    email = user_preferences.get("email")
    
    if not email:
        raise HTTPException(status_code=400, detail="User email not found")
    
    if asset_type not in (None, "movie", "tv"):
        raise HTTPException(status_code=400, detail="asset_type must be 'movie', 'tv', or omitted")
    
    try:
        precompute_user_genre_recommendations_for_email(
            email=email,
            limit_per_genre=limit_per_genre,
            asset_type=asset_type,
        )
        return {
            "status": "ok",
            "email": email,
            "asset_type": asset_type or "both",
            "limit_per_genre": limit_per_genre,
        }
    except Exception as e:
        print(f"Error refreshing recommendations for {email}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/recompute_genre_recs")
def recompute_genre_recommendations(
    payload: RecomputeGenreRecsRequest,
    x_admin_token: Optional[str] = Header(default=None),
):
    """
    Trigger recomputation of precomputed genre recommendations.

    - Provide email to recompute for a single user, otherwise recomputes all users.
    - asset_type can be 'movie', 'tv', or omitted for both.
    """
    _verify_admin_token(x_admin_token)
    if payload.asset_type not in (None, "movie", "tv"):
        raise HTTPException(status_code=400, detail="asset_type must be 'movie', 'tv', or omitted")

    if payload.email:
        precompute_user_genre_recommendations_for_email(
            email=payload.email,
            limit_per_genre=payload.limit_per_genre,
            asset_type=payload.asset_type,
        )
    else:
        precompute_all_user_genre_recommendations(limit_per_genre=payload.limit_per_genre)

    return {
        "status": "ok",
        "email": payload.email,
        "asset_type": payload.asset_type or "both",
        "limit_per_genre": payload.limit_per_genre,
    }


@app.post("/admin/rebuild_plot_parquets")
def rebuild_plot_parquets(
    payload: PlotRebuildRequest,
    x_admin_token: Optional[str] = Header(default=None),
):
    _verify_admin_token(x_admin_token)
    if payload.asset_type not in ("movies", "tv", "all"):
        raise HTTPException(status_code=400, detail="asset_type must be 'movies', 'tv', or 'all'")

    results = []
    types = ["movies", "tv"] if payload.asset_type == "all" else [payload.asset_type]
    for asset_type in types:
        results.append(
            generate_and_upload_plots(
                asset_type=asset_type,
                output_dir=os.path.join(os.path.dirname(__file__), "datasets"),
                max_items=payload.max_items,
                read_from_storage=payload.read_from_storage,
            )
        )

    return {"status": "ok", "results": results}
