from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import logging
import random
import threading
import time
from pydantic import BaseModel
from typing import Any, Dict, Optional, List, Set
from dotenv import load_dotenv
import requests
import json
import os
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("nightout")

# Events parser: allow triggering from main app (on signup / before home load)
_events_parser_lock = threading.Lock()
_events_parser_last_run_time: Optional[float] = None
EVENTS_PARSER_COOLDOWN_SECONDS = 3600  # 1 hour cooldown between runs

# Firebase Admin SDK imports
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import auth as firebase_auth
from firebase_admin import messaging

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
ticketmaster_api_key = os.getenv("TICKETMASTER_API_KEY", "")
tmdb_api_key = os.getenv("TMDB_API_KEY", "")
# Watchmode API for streaming deeplinks (Netflix, Amazon Prime, etc.). Override with WATCHMODE_API_KEY in .env.
watchmode_api_key = os.getenv("WATCHMODE_API_KEY", "z4h3DYeMSEGbpvPgJUw3l8h6iJUuQOVGjov4hiqq")

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
    interests: List[str] = []
    favoriteTeams: List[str] = []
    favoriteArtists: List[str] = []

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


class PatchUserPreferencesRequest(BaseModel):
    """Partial update for `user_preferences/{email}`.

    The client sends only the fields it wants to change (one section at a
    time, e.g. `{"favoriteArtists": [...]}`). The endpoint validates each
    field name against an allow-list and merges them into Firestore so other
    fields stay untouched.
    """

    fields: Dict[str, Any]


class AddAssetRequest(BaseModel):
    asset_type: str  # 'movie' or 'tv'
    asset_id: int  # ID of the movie or TV show


class EventSwipe(BaseModel):
    event_id: str
    name: str
    liked: bool
    segment: Optional[str] = None
    genre: Optional[str] = None
    venue_name: Optional[str] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    date: Optional[str] = None


class EventSwipeBatch(BaseModel):
    swipes: List[EventSwipe]


class PlaceSwipe(BaseModel):
    name: str
    liked: bool
    place_type: Optional[str] = None
    address: Optional[str] = None
    rating: Optional[float] = None
    price_level: Optional[int] = None


class PlaceSwipeBatch(BaseModel):
    swipes: List[PlaceSwipe]


class SaveItemRequest(BaseModel):
    """Save / unsave a single discoverable item for later.

    `kind` is one of: 'movie', 'tv', 'event'. `item_id` is whatever id the
    client has on hand (TMDB id for movies/tv, internal event_id for curated
    events). It's coerced to a string before being used in the Firestore
    document key so int/str ids hash to the same bucket.

    `snapshot` is an optional shallow copy of the card's display fields so the
    Saved tab can render the item even if the source list later disappears.
    """

    kind: str
    item_id: str
    snapshot: Optional[dict] = None


class UnsaveItemRequest(BaseModel):
    kind: str
    item_id: str


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


# Names sent by the Flutter onboarding flow are normalized using the same
# mapping the watch_providers job uses, so equality on these strings always
# lines up with what TMDB returns.
_USER_SERVICE_ALIASES = {
    "amazon prime video": "amazon prime video",
    "amazon video": "amazon prime video",
    "prime video": "amazon prime video",
    "amazon": "amazon prime video",
    "disney plus": "disney+",
    "disney+": "disney+",
    "max": "hbo max",
    "hbo": "hbo max",
    "hbo max": "hbo max",
    "apple tv plus": "apple tv+",
    "apple tv+": "apple tv+",
    "apple tv": "apple tv+",
    "paramount plus": "paramount+",
    "paramount+": "paramount+",
    "peacock premium": "peacock",
    "peacock": "peacock",
    "youtube": "youtube premium",
    "youtube premium": "youtube premium",
    "tubi tv": "tubi",
    "tubi": "tubi",
    "fandango at home": "vudu",
    "vudu": "vudu",
    "amc plus": "amc+",
    "amc+": "amc+",
    "discovery plus": "discovery+",
    "discovery+": "discovery+",
    "espn plus": "espn+",
    "espn+": "espn+",
    "fubo": "fubo",
    "fubotv": "fubo",
    "fubo tv": "fubo",
    "fubo sports": "fubo",
    "fubo sports network": "fubo",
}


def _normalize_service_name(name: str) -> str:
    key = (name or "").strip().lower()
    return _USER_SERVICE_ALIASES.get(key, key)


def _detail_matches_user_services(detail: dict, user_services: Set[str]) -> bool:
    """Return True if the asset has at least one flatrate/free provider that
    overlaps with the user's selected streaming services. Treats missing
    provider data as a non-match so we don't surface titles we can't verify."""
    if not user_services:
        return True
    providers = (detail or {}).get("watch_providers") or []
    for p in providers:
        if not isinstance(p, dict):
            continue
        if p.get("provider_type") not in ("flatrate", "free"):
            continue
        candidate = _normalize_service_name(
            p.get("provider_name") or p.get("original_name") or ""
        )
        if candidate and candidate in user_services:
            return True
    return False


def _get_precomputed_genre_ids(
    email: str,
    asset_type: str,
    genre_ids: Optional[List[int]] = None,
) -> List[int]:
    """
    Fetch precomputed genre recommendations for a user.
    Returns a deduped list of asset IDs. If genre_ids is None or empty, returns all IDs from all genres.
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
    keys_to_use = [str(gid) for gid in genre_ids] if genre_ids else list(genre_map.keys())
    for gid_key in keys_to_use:
        ids = genre_map.get(gid_key, [])
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
    # When true, the backend filters candidates so the picked title is
    # available on at least one of `services`. Falls back to a normal pick
    # if nothing in the candidate pool matches, so the user always gets a
    # suggestion. Empty `services` makes the flag a no-op.
    only_my_services: bool = False
    services: List[str] = []


class NightOutSuggestionRequest(BaseModel):
    party_size: int = 1
    budget: Optional[str] = None
    latitude: float
    longitude: float
    radius_meters: float = 10000.0
    dietary_preferences: List[str] = []
    cuisines: List[str] = []
    interests: List[str] = []
    excluded_names: List[str] = []


BUDGET_TO_MAX_PRICE_LEVEL = {
    "Free": 0,
    "Under $20": 1,
    "Under $50": 2,
    "Under $100": 3,
    "$100+": 4,
}

BUDGET_TO_MAX_EVENT_PRICE = {
    "Free": 0.0,
    "Under $20": 20.0,
    "Under $50": 50.0,
    "Under $100": 100.0,
    "$100+": None,
}


class NightOutEventRequest(BaseModel):
    party_size: int
    budget: Optional[str] = None
    latitude: float
    longitude: float
    radius_miles: float = 10.0
    classification: Optional[str] = None
    keyword: Optional[str] = None
    interests: List[str] = []
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    days_ahead: Optional[int] = None  # horizon in days (e.g. 90 for onboarding diversity)
    excluded_event_ids: List[str] = []


def _encode_geohash(lat: float, lng: float, precision: int = 9) -> str:
    BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
    lat_range = [-90.0, 90.0]
    lng_range = [-180.0, 180.0]
    is_lng = True
    bits = 0
    char_idx = 0
    geohash: list[str] = []

    while len(geohash) < precision:
        if is_lng:
            mid = (lng_range[0] + lng_range[1]) / 2
            if lng >= mid:
                char_idx = char_idx * 2 + 1
                lng_range[0] = mid
            else:
                char_idx = char_idx * 2
                lng_range[1] = mid
        else:
            mid = (lat_range[0] + lat_range[1]) / 2
            if lat >= mid:
                char_idx = char_idx * 2 + 1
                lat_range[0] = mid
            else:
                char_idx = char_idx * 2
                lat_range[1] = mid

        is_lng = not is_lng
        bits += 1
        if bits == 5:
            geohash.append(BASE32[char_idx])
            bits = 0
            char_idx = 0

    return "".join(geohash)


def search_nearby_events(
    latitude: float,
    longitude: float,
    radius_miles: float = 10.0,
    classification: Optional[str] = None,
    keyword: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    size: int = 20,
) -> List[dict]:
    if not ticketmaster_api_key:
        raise HTTPException(status_code=501, detail="Ticketmaster API key not configured")

    geohash = _encode_geohash(latitude, longitude)
    params: dict[str, str] = {
        "apikey": ticketmaster_api_key,
        "geoPoint": geohash,
        "radius": str(int(max(radius_miles, 1))),
        "unit": "miles",
        "size": str(size),
        "sort": "date,asc",
    }

    if classification:
        params["classificationName"] = classification
    if keyword:
        params["keyword"] = keyword
    if start_date:
        params["startDateTime"] = start_date
    if end_date:
        params["endDateTime"] = end_date

    try:
        resp = requests.get(
            "https://app.ticketmaster.com/discovery/v2/events.json",
            params=params,
        )
        data = resp.json()

        if resp.status_code != 200:
            error_msg = data.get("fault", {}).get("faultstring", "Unknown error")
            raise HTTPException(status_code=502, detail=f"Ticketmaster API error: {error_msg}")

        events = data.get("_embedded", {}).get("events", [])
        results = []

        for event in events:
            images = event.get("images", [])
            image_url = None
            for img in sorted(images, key=lambda x: x.get("width", 0), reverse=True):
                if img.get("url"):
                    image_url = img["url"]
                    break

            venues = event.get("_embedded", {}).get("venues", [])
            venue_name = None
            venue_address = None
            if venues:
                v = venues[0]
                venue_name = v.get("name")
                parts = [
                    v.get("address", {}).get("line1", ""),
                    v.get("city", {}).get("name", ""),
                    v.get("state", {}).get("stateCode", ""),
                ]
                venue_address = ", ".join(p for p in parts if p)

            dates = event.get("dates", {})
            start_info = dates.get("start", {})
            date_str = start_info.get("localDate")
            time_str = start_info.get("localTime")

            price_ranges = event.get("priceRanges", [])
            min_price = None
            max_price = None
            currency = None
            if price_ranges:
                min_price = price_ranges[0].get("min")
                max_price = price_ranges[0].get("max")
                currency = price_ranges[0].get("currency", "USD")

            classifications = event.get("classifications", [])
            segment = None
            genre = None
            if classifications:
                c = classifications[0]
                segment = c.get("segment", {}).get("name")
                genre = c.get("genre", {}).get("name")

            description = (
                event.get("info")
                or event.get("description")
                or event.get("pleaseNote")
            )

            results.append({
                "id": event.get("id"),
                "name": event.get("name"),
                "description": description,
                "image_url": image_url,
                "date": date_str,
                "time": time_str,
                "venue_name": venue_name,
                "venue_address": venue_address,
                "min_price": min_price,
                "max_price": max_price,
                "currency": currency,
                "segment": segment,
                "genre": genre,
                "ticket_url": event.get("url"),
            })

        return results
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching events: {str(e)}")


@app.post("/night_out_events")
async def get_night_out_events(req: NightOutEventRequest):
    excluded = set(req.excluded_event_ids)
    max_event_price = BUDGET_TO_MAX_EVENT_PRICE.get(req.budget)

    classification = req.classification
    keyword = req.keyword

    if not classification and not keyword and req.interests:
        personalized = _personalized_event_params(req.interests)
        classification = personalized.get("classification")
        keyword = personalized.get("keyword")

    from datetime import timedelta
    today = datetime.utcnow()
    default_start = req.start_date or today.strftime("%Y-%m-%dT%H:%M:%SZ")
    default_end = req.end_date or (today + timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")

    MAX_RADIUS = 100.0
    current_radius = req.radius_miles

    while current_radius <= MAX_RADIUS:
        events = search_nearby_events(
            latitude=req.latitude,
            longitude=req.longitude,
            radius_miles=current_radius,
            classification=classification,
            keyword=keyword,
            start_date=default_start,
            end_date=default_end,
        )

        if excluded:
            events = [e for e in events if e["id"] not in excluded]

        if max_event_price is not None:
            events = [
                e for e in events
                if e.get("min_price") is None or e["min_price"] <= max_event_price
            ]

        if events:
            pick = random.choice(events)
            pick["party_size"] = req.party_size
            pick["budget"] = req.budget
            pick["radius_used_miles"] = current_radius
            return pick

        current_radius *= 2
        if current_radius > MAX_RADIUS:
            raise HTTPException(status_code=404, detail="No events found nearby")

    raise HTTPException(status_code=404, detail="No events found nearby")


@app.post("/nearby_events_batch")
async def get_nearby_events_batch(req: NightOutEventRequest):
    from datetime import timedelta
    today = datetime.utcnow()
    horizon_days = req.days_ahead if req.days_ahead is not None else 90
    default_start = req.start_date or today.strftime("%Y-%m-%dT%H:%M:%SZ")
    default_end = req.end_date or (today + timedelta(days=horizon_days)).strftime("%Y-%m-%dT%H:%M:%SZ")

    events = search_nearby_events(
        latitude=req.latitude,
        longitude=req.longitude,
        radius_miles=req.radius_miles,
        classification=req.classification,
        keyword=req.keyword,
        start_date=default_start,
        end_date=default_end,
        size=60,
    )

    excluded = set(req.excluded_event_ids)
    if excluded:
        events = [e for e in events if e["id"] not in excluded]

    return {"events": events}


class NearbyPlacesBatchRequest(BaseModel):
    latitude: float
    longitude: float
    radius_meters: float = 80467.0  # ~50 miles for onboarding diversity


@app.post("/nearby_places_batch")
async def get_nearby_places_batch(req: NearbyPlacesBatchRequest):
    location = LocationData(latitude=req.latitude, longitude=req.longitude)
    places = search_nearby_places(location, radius=req.radius_meters)
    return {"places": places}


@app.get("/event_classifications")
def event_classifications():
    return [
        {"id": "music", "name": "Music"},
        {"id": "sports", "name": "Sports"},
        {"id": "arts", "name": "Arts & Theatre"},
        {"id": "film", "name": "Film"},
        {"id": "miscellaneous", "name": "Other"},
    ]


@app.get("/interest_options")
def interest_options():
    return {
        "Sports & Fitness": [
            "Soccer", "Basketball", "Volleyball", "Tennis", "Golf",
            "Swimming", "Running", "Cycling", "Yoga", "Boxing", "Bowling",
        ],
        "Music & Entertainment": [
            "Live Music", "Comedy Shows", "Theater", "Dance Clubs", "Karaoke",
        ],
        "Arts & Culture": [
            "Art Galleries", "Museums", "Film Festivals",
        ],
        "Food & Drink": [
            "Wine Tasting", "Cooking Classes", "Brewery Tours", "Coffee Culture",
        ],
        "Outdoors & Adventure": [
            "Hiking", "Rock Climbing", "Kayaking", "Fishing", "Camping",
        ],
    }


def _get_tmdb_youtube_trailer(asset_id: int, asset_type: str) -> Optional[str]:
    """Fetch the YouTube trailer key for a movie or TV show from TMDB."""
    if not tmdb_api_key:
        return None
    media = "movie" if asset_type == "movie" else "tv"
    url = f"https://api.themoviedb.org/3/{media}/{asset_id}/videos?api_key={tmdb_api_key}&language=en-US"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code != 200:
            return None
        results = resp.json().get("results", [])
        for v in results:
            if v.get("site") == "YouTube" and v.get("type") == "Trailer":
                return v.get("key")
        for v in results:
            if v.get("site") == "YouTube" and v.get("type") == "Teaser":
                return v.get("key")
        for v in results:
            if v.get("site") == "YouTube":
                return v.get("key")
        return None
    except Exception:
        return None


def _get_tmdb_watch_url(asset_id: int, asset_type: str) -> str:
    media = "movie" if asset_type == "movie" else "tv"
    return f"https://www.themoviedb.org/{media}/{asset_id}/watch"


def _clean_deeplink(url) -> Optional[str]:
    """Return url only if it's a real URL, not a Watchmode paywall message."""
    if url is None:
        return None
    if not isinstance(url, str):
        url = str(url)
    if "paid plans" in url.lower() or not (url.startswith("http://") or url.startswith("https://")):
        return None
    return url


def _get_watchmode_sources(tmdb_id: int, asset_type: str) -> List[dict]:
    """Fetch streaming sources with deep links from Watchmode using TMDB ID."""
    if not watchmode_api_key:
        return []

    search_field = "tmdb_movie_id" if asset_type == "movie" else "tmdb_tv_id"
    search_url = (
        f"https://api.watchmode.com/v1/search/"
        f"?apiKey={watchmode_api_key}"
        f"&search_field={search_field}&search_value={tmdb_id}"
    )
    try:
        resp = requests.get(search_url, timeout=5)
        if resp.status_code != 200:
            return []
        results = resp.json().get("title_results", [])
        if not results:
            return []

        watchmode_id = results[0].get("id")
        if not watchmode_id:
            return []

        sources_url = (
            f"https://api.watchmode.com/v1/title/{watchmode_id}/sources/"
            f"?apiKey={watchmode_api_key}&regions=US"
        )
        resp = requests.get(sources_url, timeout=5)
        if resp.status_code != 200:
            return []

        raw_sources = resp.json()
        if isinstance(raw_sources, dict):
            raw_sources = raw_sources.get("sources", raw_sources.get("results", []))
        if not isinstance(raw_sources, list):
            return []

        sources = []
        seen = set()
        for s in raw_sources:
            name = s.get("name", "")
            stype = s.get("type", "")
            key = f"{name}_{stype}"
            if key in seen:
                continue
            seen.add(key)
            web_url = s.get("web_url") or s.get("url") or s.get("link")
            if web_url is not None and not isinstance(web_url, str):
                web_url = str(web_url)
            if web_url and not (web_url.startswith("http://") or web_url.startswith("https://")):
                web_url = None

            ios_url = _clean_deeplink(s.get("ios_url"))
            android_url = _clean_deeplink(s.get("android_url"))
            deeplink = _clean_deeplink(s.get("deeplink"))
            ios_url = ios_url or deeplink or web_url
            android_url = android_url or deeplink or web_url

            if not web_url and not ios_url and not android_url:
                continue

            sources.append({
                "source_id": s.get("source_id"),
                "name": name,
                "type": stype,
                "web_url": web_url,
                "ios_url": ios_url,
                "android_url": android_url,
                "price": s.get("price"),
                "format": s.get("format"),
            })
        return sources
    except Exception as e:
        print(f"Watchmode error: {e}")
        return []


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

    # Try a sequence of progressively-looser fetches so the user always gets
    # *something* instead of a "No matches" wall:
    #   1. requested genres on the chosen asset type
    #   2. requested genres on the *other* asset type
    #   3. no genre filter on the chosen asset type
    #   4. no genre filter on the other asset type
    primary = random.choice(["movie", "tv"])
    other = "tv" if primary == "movie" else "movie"

    def _fetch(asset_type: str, genres):
        if asset_type == "movie":
            return get_movies(
                start_year=2000, end_year=current_year, genres=genres or []
            )
        return get_tv(
            start_year=2000, end_year=current_year, genres=genres or []
        )

    attempts = [
        (primary, req.genres),
        (other, req.genres),
        (primary, []),
        (other, []),
    ]

    # Normalize the user's onboarding-style service names into the same
    # canonical form provider data uses, so equality comparisons line up.
    user_services_set: set = set()
    if req.only_my_services and req.services:
        user_services_set = {_normalize_service_name(s) for s in req.services if s}

    pick = None
    pick_providers: List[dict] = []
    asset_type = primary
    services_filter_applied = False
    services_filter_failed = False

    # Lazy import — avoids a circular when the watch_providers job is unused.
    try:
        from service.watch_providers_job import get_providers_for_asset
    except Exception:
        get_providers_for_asset = None  # type: ignore

    def _candidate_providers(candidate_id: int, atype: str) -> List[dict]:
        """Fetch providers for a candidate. Returns [] on any failure so the
        outer logic can treat the candidate as "not on user's services"
        without blowing up the request."""
        if get_providers_for_asset is None:
            return []
        try:
            return get_providers_for_asset(
                candidate_id, atype, read_from_local=False
            ) or []
        except Exception as exc:
            print(
                f"Could not fetch watch providers for {atype} {candidate_id}: {exc}"
            )
            return []

    for attempt_type, attempt_genres in attempts:
        data = _fetch(attempt_type, attempt_genres)
        results = data.get("results", [])
        if not results:
            continue

        if user_services_set:
            # Sample at most N candidates per attempt so we don't fan out to
            # hundreds of provider lookups. 25 is enough to find a match for
            # popular genres while keeping latency bounded.
            shuffled = random.sample(results, min(len(results), 25))
            for candidate in shuffled:
                providers = _candidate_providers(candidate["id"], attempt_type)
                if _detail_matches_user_services(
                    {"watch_providers": providers}, user_services_set
                ):
                    pick = candidate
                    pick_providers = providers
                    asset_type = attempt_type
                    services_filter_applied = True
                    if attempt_genres != req.genres:
                        pick["genre_fallback"] = True
                    break
            if pick is not None:
                break
        else:
            pick = random.choice(results)
            asset_type = attempt_type
            if attempt_genres != req.genres:
                pick["genre_fallback"] = True
            break

    # If the user asked for service-only and nothing matched anywhere, fall
    # back to a normal random pick so the screen never errors out, and tell
    # the client we couldn't honor the filter.
    if pick is None and user_services_set:
        services_filter_failed = True
        for attempt_type, attempt_genres in attempts:
            data = _fetch(attempt_type, attempt_genres)
            results = data.get("results", [])
            if results:
                pick = random.choice(results)
                asset_type = attempt_type
                if attempt_genres != req.genres:
                    pick["genre_fallback"] = True
                break

    if pick is None:
        raise HTTPException(status_code=404, detail="No suggestions found")

    pick["asset_type"] = asset_type
    pick["services_filter_applied"] = services_filter_applied
    pick["services_filter_failed"] = services_filter_failed

    if pick_providers:
        # Reuse the providers we already fetched while searching for a match.
        pick["watch_providers"] = pick_providers
    else:
        try:
            providers = _candidate_providers(pick["id"], asset_type)
            pick["watch_providers"] = providers
        except Exception as e:
            print(f"Could not fetch watch providers: {e}")
            pick["watch_providers"] = []

    try:
        youtube_key = _get_tmdb_youtube_trailer(pick["id"], asset_type)
        pick["youtube_trailer_key"] = youtube_key
    except Exception:
        pick["youtube_trailer_key"] = None

    pick["tmdb_watch_url"] = _get_tmdb_watch_url(pick["id"], asset_type)

    try:
        streaming_sources = _get_watchmode_sources(pick["id"], asset_type)
        pick["streaming_sources"] = streaming_sources
    except Exception:
        pick["streaming_sources"] = []

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

INTEREST_TO_PLACE_TYPES: dict[str, list[str]] = {
    "Soccer": ["sports_complex"],
    "Basketball": ["sports_complex"],
    "Volleyball": ["sports_complex"],
    "Tennis": ["sports_complex"],
    "Golf": ["golf_course"],
    "Swimming": ["swimming_pool"],
    "Running": ["park"],
    "Cycling": ["park"],
    "Yoga": ["yoga_studio"],
    "Boxing": ["gym"],
    "Bowling": ["bowling_alley"],
    "Live Music": ["night_club", "bar"],
    "Comedy Shows": ["performing_arts_theater"],
    "Theater": ["performing_arts_theater"],
    "Dance Clubs": ["night_club"],
    "Karaoke": ["bar", "night_club"],
    "Art Galleries": ["art_gallery"],
    "Museums": ["museum"],
    "Film Festivals": ["movie_theater"],
    "Wine Tasting": ["wine_bar"],
    "Cooking Classes": ["restaurant"],
    "Brewery Tours": ["bar"],
    "Coffee Culture": ["cafe"],
    "Hiking": ["national_park", "park"],
    "Rock Climbing": ["gym", "sports_complex"],
    "Kayaking": ["park"],
    "Fishing": ["park"],
    "Camping": ["national_park", "park"],
}

INTEREST_TO_TM_PARAMS: dict[str, dict[str, str]] = {
    "Soccer": {"classification": "sports", "keyword": "soccer"},
    "Basketball": {"classification": "sports", "keyword": "basketball"},
    "Volleyball": {"classification": "sports", "keyword": "volleyball"},
    "Tennis": {"classification": "sports", "keyword": "tennis"},
    "Golf": {"classification": "sports", "keyword": "golf"},
    "Swimming": {"classification": "sports", "keyword": "swimming"},
    "Running": {"classification": "sports", "keyword": "marathon running"},
    "Cycling": {"classification": "sports", "keyword": "cycling"},
    "Yoga": {"classification": "miscellaneous", "keyword": "yoga"},
    "Boxing": {"classification": "sports", "keyword": "boxing"},
    "Bowling": {"classification": "sports", "keyword": "bowling"},
    "Live Music": {"classification": "music"},
    "Comedy Shows": {"classification": "arts", "keyword": "comedy"},
    "Theater": {"classification": "arts", "keyword": "theatre"},
    "Dance Clubs": {"classification": "music", "keyword": "DJ dance"},
    "Karaoke": {"classification": "music", "keyword": "karaoke"},
    "Art Galleries": {"classification": "arts", "keyword": "art exhibition"},
    "Museums": {"classification": "arts", "keyword": "museum exhibit"},
    "Film Festivals": {"classification": "film", "keyword": "film festival"},
    "Wine Tasting": {"classification": "miscellaneous", "keyword": "wine tasting"},
    "Cooking Classes": {"classification": "miscellaneous", "keyword": "cooking"},
    "Brewery Tours": {"classification": "miscellaneous", "keyword": "brewery"},
    "Coffee Culture": {"classification": "miscellaneous", "keyword": "coffee"},
    "Hiking": {"classification": "miscellaneous", "keyword": "hiking outdoor"},
    "Rock Climbing": {"classification": "sports", "keyword": "climbing"},
    "Kayaking": {"classification": "sports", "keyword": "kayak"},
    "Fishing": {"classification": "sports", "keyword": "fishing"},
    "Camping": {"classification": "miscellaneous", "keyword": "camping outdoor"},
}


def _personalized_place_types(
    cuisines: List[str],
    interests: List[str],
) -> list[str]:
    types: list[str] = []
    for c in cuisines:
        types.extend(CUISINE_TO_PLACE_TYPES.get(c, []))
    for i in interests:
        types.extend(INTEREST_TO_PLACE_TYPES.get(i, []))
    return list(dict.fromkeys(types)) if types else list(places_of_interest)


def _personalized_event_params(interests: List[str]) -> dict[str, str | None]:
    if not interests:
        return {"classification": None, "keyword": None}
    pick = random.choice(interests)
    params = INTEREST_TO_TM_PARAMS.get(pick, {})
    return {
        "classification": params.get("classification"),
        "keyword": params.get("keyword"),
    }


@app.post("/night_out_suggestion")
async def get_night_out_suggestion(req: NightOutSuggestionRequest):
    included_types = _personalized_place_types(req.cuisines, req.interests)

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


# Allow-list of fields the section editors are permitted to overwrite via the
# /user_preferences/patch endpoint, paired with their expected Python type.
# Anything not listed here is rejected so a malicious or buggy client can't
# clobber server-managed fields like `last_latitude`, `createdAt`, swipe
# history, etc. Lists of strings/ints are checked element-by-element below.
_PATCHABLE_PREFERENCE_FIELDS: Dict[str, type] = {
    "age": int,
    "movieIds": list,
    "tvShowIds": list,
    "streamingServices": list,
    "dietaryPreferences": list,
    "cuisines": list,
    "interests": list,
    "favoriteTeams": list,
    "favoriteArtists": list,
}

# Subset of the patchable fields that change which events / restaurants /
# titles a user should see. When any of these is updated, kick off a per-user
# pipeline rerun + home payload rebuild in the background so the client sees
# the new picks without waiting for the next hourly cron.
_EVENT_AFFECTING_PREFERENCE_FIELDS: Set[str] = {
    "favoriteArtists",
    "favoriteTeams",
    "streamingServices",
    "cuisines",
    "dietaryPreferences",
    "interests",
    "age",
}

# Per-user cooldown so a user mashing the same edit button doesn't spawn a
# rescrape thread per tap. Lives next to the location-driven cooldown.
_PATCH_RESCRAPE_COOLDOWN_SECONDS = 60
_patch_rescrape_lock = threading.Lock()
_patch_rescrape_last_run: Dict[str, float] = {}


def _validate_patch_value(field: str, value: Any) -> Any:
    """Coerce / validate a single field from the patch payload.

    Returns the cleaned value to write to Firestore, or raises HTTPException
    with a 400 status when the value doesn't match the schema for `field`.
    """
    expected = _PATCHABLE_PREFERENCE_FIELDS[field]

    # `age` is the only scalar; everything else is a list of strings or ints.
    if field == "age":
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise HTTPException(
                status_code=400,
                detail=f"Field '{field}' must be an integer or null",
            )
        if value < 13 or value > 120:
            raise HTTPException(
                status_code=400,
                detail="Age must be between 13 and 120",
            )
        return int(value)

    if expected is list:
        if not isinstance(value, list):
            raise HTTPException(
                status_code=400,
                detail=f"Field '{field}' must be a list",
            )
        if field in {"movieIds", "tvShowIds"}:
            cleaned: List[int] = []
            for item in value:
                if isinstance(item, bool) or not isinstance(item, int):
                    raise HTTPException(
                        status_code=400,
                        detail=f"All items in '{field}' must be integers",
                    )
                cleaned.append(int(item))
            return cleaned
        cleaned_str: List[str] = []
        for item in value:
            if not isinstance(item, str):
                raise HTTPException(
                    status_code=400,
                    detail=f"All items in '{field}' must be strings",
                )
            stripped = item.strip()
            if stripped:
                cleaned_str.append(stripped)
        return cleaned_str

    raise HTTPException(status_code=400, detail=f"Unknown field '{field}'")


def _refresh_home_payload_safe(email: str) -> None:
    """Background-thread entrypoint that rebuilds and writes the user's home
    payload, swallowing any failures so the request handler never sees them.
    """
    try:
        from service.home_payload import write_home_payload_safe

        write_home_payload_safe(email)
    except Exception as e:
        print(f"[patch_prefs] home_payload refresh failed for {email}: {e}")


@app.post("/user_preferences/patch")
async def patch_user_preferences(
    payload: PatchUserPreferencesRequest,
    authorization: Optional[str] = Header(default=None),
):
    """Partial update for the signed-in user's preference document.

    Replaces only the fields included in `payload.fields`, leaving everything
    else (location, swipe history, watched lists, ...) untouched. Used by the
    Preferences screen's per-section editors so users can tweak a single
    interest/artist/etc. without re-running onboarding.

    When an event-affecting field (artists, teams, cuisines, ...) is changed,
    schedules a per-user pipeline rerun + home payload rebuild in the
    background so the home screen reflects the new picks shortly after.
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

    decoded = _verify_and_get_user(authorization)
    email = decoded.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="Email not found in token")

    if not payload.fields:
        raise HTTPException(status_code=400, detail="No fields provided")

    unknown = [f for f in payload.fields.keys() if f not in _PATCHABLE_PREFERENCE_FIELDS]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Field(s) not allowed: {', '.join(sorted(unknown))}",
        )

    cleaned: Dict[str, Any] = {
        field: _validate_patch_value(field, value)
        for field, value in payload.fields.items()
    }

    try:
        prefs_ref = firebase_db.collection("user_preferences").document(email)
        snapshot = prefs_ref.get()
        if not snapshot.exists:
            # New users hit /user_preferences (full submit) at the end of
            # onboarding; the patch endpoint is for tweaking an existing doc.
            raise HTTPException(
                status_code=404,
                detail="User preferences not found. Complete onboarding first.",
            )

        cleaned["updatedAt"] = datetime.utcnow().isoformat() + "Z"
        prefs_ref.set(cleaned, merge=True)

        affected = [f for f in cleaned.keys() if f in _EVENT_AFFECTING_PREFERENCE_FIELDS]
        rescrape_started = False
        if affected:
            with _patch_rescrape_lock:
                last = _patch_rescrape_last_run.get(email)
                now_ts = time.time()
                if last is None or (now_ts - last) >= _PATCH_RESCRAPE_COOLDOWN_SECONDS:
                    _patch_rescrape_last_run[email] = now_ts
                    rescrape_started = True

            if rescrape_started:
                print(
                    f"[patch_prefs] {email} updated {affected} -- "
                    "starting per-user pipeline + home payload refresh"
                )
                threading.Thread(
                    target=_per_user_pipeline_safe,
                    args=(email,),
                    daemon=True,
                ).start()
            # Always rebuild the home payload (lightweight) so even when the
            # rescrape is skipped by cooldown, the new prefs are reflected in
            # what the home screen pulls back from Firestore.
            threading.Thread(
                target=_refresh_home_payload_safe,
                args=(email,),
                daemon=True,
            ).start()

        return {
            "message": "Preferences updated",
            "updated_fields": sorted(cleaned.keys() - {"updatedAt"}),
            "rescrape_started": rescrape_started,
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"patch_user_preferences error: {e}")
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


@app.post("/user_preferences/event_swipes")
async def save_event_swipes(batch: EventSwipeBatch, authorization: Optional[str] = Header(default=None)):
    initialize_firebase_if_needed()
    if firebase_db is None:
        raise HTTPException(status_code=501, detail="Firebase not configured")

    decoded = _verify_and_get_user(authorization)
    email = decoded.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="Email not found in token")

    try:
        collection = firebase_db.collection("user_event_swipes")
        doc_ref = collection.document(email)
        doc = doc_ref.get()

        liked_events: list[dict] = []
        disliked_events: list[dict] = []

        for swipe in batch.swipes:
            entry = {
                "event_id": swipe.event_id,
                "name": swipe.name,
                "segment": swipe.segment,
                "genre": swipe.genre,
                "venue_name": swipe.venue_name,
                "min_price": swipe.min_price,
                "max_price": swipe.max_price,
                "date": swipe.date,
                "swiped_at": datetime.utcnow().isoformat() + "Z",
            }
            if swipe.liked:
                liked_events.append(entry)
            else:
                disliked_events.append(entry)

        if doc.exists:
            updates: dict = {}
            if liked_events:
                updates["likedEvents"] = firestore.ArrayUnion(liked_events)
            if disliked_events:
                updates["dislikedEvents"] = firestore.ArrayUnion(disliked_events)
            if updates:
                doc_ref.update(updates)
        else:
            doc_ref.set({
                "email": email,
                "likedEvents": liked_events,
                "dislikedEvents": disliked_events,
                "createdAt": datetime.utcnow().isoformat() + "Z",
            })

        return {
            "message": "Event swipes saved",
            "liked_count": len(liked_events),
            "disliked_count": len(disliked_events),
        }
    except HTTPException:
        raise
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/user_event_history")
async def get_user_event_history(authorization: Optional[str] = Header(default=None)):
    initialize_firebase_if_needed()
    if firebase_db is None:
        raise HTTPException(status_code=501, detail="Firebase not configured")

    decoded = _verify_and_get_user(authorization)
    email = decoded.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="Email not found in token")

    try:
        doc = firebase_db.collection("user_event_swipes").document(email).get()
        if not doc.exists:
            return {"seen_event_ids": [], "liked_event_ids": []}

        data = doc.to_dict()
        liked_ids = [e["event_id"] for e in data.get("likedEvents", [])]
        disliked_ids = [e["event_id"] for e in data.get("dislikedEvents", [])]
        return {
            "seen_event_ids": liked_ids + disliked_ids,
            "liked_event_ids": liked_ids,
        }
    except HTTPException:
        raise
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/user_preferences/place_swipes")
async def save_place_swipes(batch: PlaceSwipeBatch, authorization: Optional[str] = Header(default=None)):
    initialize_firebase_if_needed()
    if firebase_db is None:
        raise HTTPException(status_code=501, detail="Firebase not configured")

    decoded = _verify_and_get_user(authorization)
    email = decoded.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="Email not found in token")

    try:
        collection = firebase_db.collection("user_place_swipes")
        doc_ref = collection.document(email)
        doc = doc_ref.get()

        liked_places: list[dict] = []
        disliked_places: list[dict] = []

        for swipe in batch.swipes:
            entry = {
                "name": swipe.name,
                "place_type": swipe.place_type,
                "address": swipe.address,
                "rating": swipe.rating,
                "price_level": swipe.price_level,
                "swiped_at": datetime.utcnow().isoformat() + "Z",
            }
            if swipe.liked:
                liked_places.append(entry)
            else:
                disliked_places.append(entry)

        if doc.exists:
            updates: dict = {}
            if liked_places:
                updates["likedPlaces"] = firestore.ArrayUnion(liked_places)
            if disliked_places:
                updates["dislikedPlaces"] = firestore.ArrayUnion(disliked_places)
            if updates:
                doc_ref.update(updates)
        else:
            doc_ref.set({
                "email": email,
                "likedPlaces": liked_places,
                "dislikedPlaces": disliked_places,
                "createdAt": datetime.utcnow().isoformat() + "Z",
            })

        return {
            "message": "Place swipes saved",
            "liked_count": len(liked_places),
            "disliked_count": len(disliked_places),
        }
    except HTTPException:
        raise
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/user_place_history")
async def get_user_place_history(authorization: Optional[str] = Header(default=None)):
    initialize_firebase_if_needed()
    if firebase_db is None:
        raise HTTPException(status_code=501, detail="Firebase not configured")

    decoded = _verify_and_get_user(authorization)
    email = decoded.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="Email not found in token")

    try:
        doc = firebase_db.collection("user_place_swipes").document(email).get()
        if not doc.exists:
            return {"seen_place_names": [], "liked_place_names": []}

        data = doc.to_dict()
        liked = [p["name"] for p in data.get("likedPlaces", [])]
        disliked = [p["name"] for p in data.get("dislikedPlaces", [])]
        return {
            "seen_place_names": liked + disliked,
            "liked_place_names": liked,
        }
    except HTTPException:
        raise
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------------------------
# Saved-for-later items
#
# Lightweight "bookmark" feature: any movie / TV show / curated event card on
# the discover surface (and on the matching detail screens) exposes a save
# action that writes a single doc into
#   user_saved_items/{email}/items/{kind}_{item_id}
# We store a `snapshot` so the Saved tab can render the card without re-
# resolving the source list (which may have rotated or been filtered out).
# Reads/writes are scoped to the authenticated caller's email so users can
# only see and mutate their own saved list.
# -----------------------------------------------------------------------------

_ALLOWED_SAVE_KINDS = {"movie", "tv", "event"}


def _save_doc_id(kind: str, item_id: str) -> str:
    """Stable Firestore doc id for a (kind, id) pair. Normalising both sides
    means saving from a card with `id: 550` (int) and unsaving from a card
    with `id: "550"` (string) hit the same document."""
    safe_id = str(item_id).strip()
    return f"{kind}_{safe_id}"


@app.post("/saved_items/save")
async def save_item_for_later(
    payload: SaveItemRequest,
    authorization: Optional[str] = Header(default=None),
):
    initialize_firebase_if_needed()
    if firebase_db is None:
        raise HTTPException(status_code=501, detail="Firebase not configured")

    decoded = _verify_and_get_user(authorization)
    email = decoded.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="Email not found in token")

    kind = (payload.kind or "").strip().lower()
    if kind not in _ALLOWED_SAVE_KINDS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid kind '{payload.kind}'. Use one of {sorted(_ALLOWED_SAVE_KINDS)}.",
        )

    item_id = (payload.item_id or "").strip()
    if not item_id:
        raise HTTPException(status_code=400, detail="item_id is required")

    try:
        doc_ref = (
            firebase_db.collection("user_saved_items")
            .document(email)
            .collection("items")
            .document(_save_doc_id(kind, item_id))
        )
        doc_ref.set(
            {
                "kind": kind,
                "item_id": item_id,
                "snapshot": payload.snapshot or {},
                "saved_at": firestore.SERVER_TIMESTAMP,
            },
            merge=True,
        )
        return {"message": "saved", "kind": kind, "item_id": item_id}
    except HTTPException:
        raise
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/saved_items/unsave")
async def unsave_item_for_later(
    payload: UnsaveItemRequest,
    authorization: Optional[str] = Header(default=None),
):
    initialize_firebase_if_needed()
    if firebase_db is None:
        raise HTTPException(status_code=501, detail="Firebase not configured")

    decoded = _verify_and_get_user(authorization)
    email = decoded.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="Email not found in token")

    kind = (payload.kind or "").strip().lower()
    if kind not in _ALLOWED_SAVE_KINDS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid kind '{payload.kind}'. Use one of {sorted(_ALLOWED_SAVE_KINDS)}.",
        )

    item_id = (payload.item_id or "").strip()
    if not item_id:
        raise HTTPException(status_code=400, detail="item_id is required")

    try:
        doc_ref = (
            firebase_db.collection("user_saved_items")
            .document(email)
            .collection("items")
            .document(_save_doc_id(kind, item_id))
        )
        doc_ref.delete()
        return {"message": "unsaved", "kind": kind, "item_id": item_id}
    except HTTPException:
        raise
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/saved_items")
async def list_saved_items(authorization: Optional[str] = Header(default=None)):
    """Return the authenticated user's saved cards grouped by kind.

    Newest first within each bucket. Documents missing a `saved_at` (e.g.
    legacy rows written before this endpoint shipped) are pushed to the end
    rather than dropped so users never lose data silently.
    """

    initialize_firebase_if_needed()
    if firebase_db is None:
        raise HTTPException(status_code=501, detail="Firebase not configured")

    decoded = _verify_and_get_user(authorization)
    email = decoded.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="Email not found in token")

    try:
        items_ref = (
            firebase_db.collection("user_saved_items")
            .document(email)
            .collection("items")
        )
        docs = list(items_ref.stream())

        movies: list[dict] = []
        tv: list[dict] = []
        events: list[dict] = []

        for doc in docs:
            data = doc.to_dict() or {}
            entry = {
                "kind": data.get("kind"),
                "item_id": data.get("item_id"),
                "snapshot": data.get("snapshot") or {},
                "saved_at": data.get("saved_at"),
            }
            kind = (entry["kind"] or "").lower()
            if kind == "movie":
                movies.append(entry)
            elif kind == "tv":
                tv.append(entry)
            elif kind == "event":
                events.append(entry)

        def _sort_key(entry: dict):
            ts = entry.get("saved_at")
            # Firestore timestamps expose `.timestamp()`; fall back to 0 so
            # legacy rows sort last instead of crashing the comparator.
            try:
                return -float(ts.timestamp())  # type: ignore[union-attr]
            except Exception:
                return 0.0

        movies.sort(key=_sort_key)
        tv.sort(key=_sort_key)
        events.sort(key=_sort_key)

        # Strip the (non-JSON-serializable) Firestore timestamp from the
        # response. The client already has its own ordering and doesn't need
        # the raw timestamp field.
        for bucket in (movies, tv, events):
            for entry in bucket:
                entry.pop("saved_at", None)

        return {
            "movies": movies,
            "tv": tv,
            "events": events,
            "total": len(movies) + len(tv) + len(events),
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


def _run_events_parser_sync(
    radius_miles: float = 50.0,
    days_ahead: int = 14,
    max_events_per_user: int = 80,
) -> None:
    """Run the events parser pipeline (blocking). Used from a background thread."""
    try:
        from service.events_parser.pipeline import run_pipeline
        run_pipeline(
            radius_miles=radius_miles,
            days_ahead=days_ahead,
            max_events_per_user=max_events_per_user,
        )
    except Exception as e:
        print(f"Events parser pipeline error: {e}")


def _run_genre_precompute_sync(email: str, limit_per_genre: int = 100) -> None:
    """Run genre recommendations precompute for one user (blocking). Used from a background thread."""
    try:
        from service.user_genre_recommendations import precompute_user_genre_recommendations_for_email
        precompute_user_genre_recommendations_for_email(email, limit_per_genre=limit_per_genre)
    except Exception as e:
        print(f"Genre precompute error for {email}: {e}")


@app.post("/trigger_events_parser")
async def trigger_events_parser(
    authorization: Optional[str] = Header(default=None),
    radius_miles: float = 50.0,
    days_ahead: int = 14,
    force_refresh: bool = False,
):
    """
    Start the events parser pipeline and genre recommendations precompute in the background (in parallel).
    For new signups and before home load. Requires auth. Returns 202 immediately.
    Both run at the same time:
    - Events parser: scrapes events/restaurants and refreshes user_curated_events for all users (1 hour cooldown unless force_refresh=True).
    - Genre precompute: refreshes movie and TV recommendations for the current user only (no cooldown).
    """
    decoded = _verify_and_get_user(authorization)
    email = decoded.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="Email not found in token")
    initialize_firebase_if_needed()
    if firebase_db is None:
        raise HTTPException(status_code=501, detail="Firebase not configured")

    def run_events_parser_background() -> None:
        global _events_parser_last_run_time
        with _events_parser_lock:
            now = time.time()
            if not force_refresh and (
                _events_parser_last_run_time is not None
                and (now - _events_parser_last_run_time) < EVENTS_PARSER_COOLDOWN_SECONDS
            ):
                print("Events parser: skipping (cooldown; runs at most once per hour)")
                return
            _events_parser_last_run_time = now
        _run_events_parser_sync(radius_miles=radius_miles, days_ahead=days_ahead)
        with _events_parser_lock:
            _events_parser_last_run_time = time.time()

    def run_genre_precompute_background() -> None:
        _run_genre_precompute_sync(email, limit_per_genre=100)

    # Run events parser in background (curated events for all users)
    thread_parser = threading.Thread(target=run_events_parser_background, daemon=True)
    thread_parser.start()
    # Run genre precompute for this user (movie/TV suggestions for new users)
    thread_genre = threading.Thread(target=run_genre_precompute_background, daemon=True)
    thread_genre.start()

    return JSONResponse(
        status_code=202,
        content={
            "message": "Events parser and genre precompute started in background",
            "status": "accepted",
        },
    )


@app.get("/user_curated_events")
async def get_user_curated_events(authorization: Optional[str] = Header(default=None)):
    """Return the curated events for the authenticated user."""
    initialize_firebase_if_needed()
    if firebase_db is None:
        raise HTTPException(status_code=501, detail="Firebase not configured")

    decoded = _verify_and_get_user(authorization)
    email = decoded.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="Email not found in token")

    try:
        doc = firebase_db.collection("user_curated_events").document(email).get()
        if not doc.exists:
            return {"events_by_category": {}, "total_events": 0}
        data = doc.to_dict()
        filtered = _filter_events_by_category(data.get("events_by_category", {}))
        data["events_by_category"] = filtered
        data["total_events"] = sum(len(v) for v in filtered.values() if isinstance(v, list))
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


PARSED_EVENTS_CATALOG_COLLECTION = "parsed_events_catalog"


def _event_is_future_or_present(event: dict) -> bool:
    """
    Return True if the event is today or in the future.

    Filters out stale events returned from Firestore that predate the
    parser-side date filter. Items without a `date` field (e.g. restaurants
    from Google Places) are always kept — they aren't time-bound.
    """
    date_str = (event.get("date") or "").strip()
    if not date_str:
        return True

    time_str = (event.get("time") or "").strip()
    now_utc = datetime.now(timezone.utc)

    if time_str:
        iso = f"{date_str}T{time_str}"
        if iso.endswith("Z"):
            iso = iso[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(iso)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc) >= now_utc
        except ValueError:
            pass

    try:
        event_date = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
    except ValueError:
        return True
    return event_date >= now_utc.date()


def _event_has_image(event: dict) -> bool:
    """
    Return True if the item has a usable image URL.

    Applied uniformly to events AND restaurants — anything without a real
    image_url is dropped so the UI never has to render a placeholder card.
    A "usable" URL must start with http(s) so we don't accept sentinels like
    "None", "null", or relative paths that would 404 in Image.network.
    """
    image_url = (event.get("image_url") or "").strip()
    if not image_url:
        return False
    lower = image_url.lower()
    if lower in {"none", "null", "false"}:
        return False
    return lower.startswith("http://") or lower.startswith("https://")


def _event_passes_filters(event: dict) -> bool:
    """True if the event is not in the past AND has a usable image."""
    return _event_is_future_or_present(event) and _event_has_image(event)


def _filter_events_by_category(
    events_by_category: dict,
) -> dict:
    """Return a copy of `events_by_category` with past/imageless events dropped."""
    if not isinstance(events_by_category, dict):
        return events_by_category
    filtered: dict = {}
    for cat, items in events_by_category.items():
        if isinstance(items, list):
            filtered[cat] = [e for e in items if _event_passes_filters(e)]
        else:
            filtered[cat] = items
    return filtered


@app.get("/event_catalog/{catalog_id:path}")
async def get_event_from_catalog(
    catalog_id: str,
    authorization: Optional[str] = Header(default=None),
):
    """
    Return a single parsed event from the catalog by id (source:source_id).
    Events are written to this table by the events parser pipeline.
    """
    initialize_firebase_if_needed()
    if firebase_db is None:
        raise HTTPException(status_code=501, detail="Firebase not configured")
    _verify_and_get_user(authorization)
    doc = firebase_db.collection(PARSED_EVENTS_CATALOG_COLLECTION).document(catalog_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Event not found in catalog")
    return doc.to_dict()


class CuratedPickRequest(BaseModel):
    category: Optional[str] = None  # sports, music, arts, food, dining, outdoors, other
    excluded_ids: List[str] = []
    excluded_names: List[str] = []


def _interleave_by_key(items: list[dict], key_fn) -> list[dict]:
    """
    Round-robin interleave items by an arbitrary grouping key.
    Within each group, items retain their original order.
    """
    from collections import OrderedDict
    groups: OrderedDict[str, list[dict]] = OrderedDict()
    for item in items:
        k = key_fn(item)
        groups.setdefault(k, []).append(item)

    result: list[dict] = []
    while groups:
        empty_keys = []
        for k in list(groups.keys()):
            bucket = groups[k]
            if bucket:
                result.append(bucket.pop(0))
            if not bucket:
                empty_keys.append(k)
        for k in empty_keys:
            del groups[k]
    return result


def _weighted_shuffle(items: list[dict], weight_fn) -> list[dict]:
    """
    Return ``items`` in a randomized order. Higher-``weight_fn`` items have
    a bigger chance of being placed earlier in the result, but every item
    can still appear anywhere — it's a soft bias, not a sort.

    Implementation: assign each item a random key of ``random()**(1/weight)``
    (Efraimidis–Spirakis weighted reservoir), then sort descending by key.
    Items with weight ``<= 0`` get a tiny floor so they don't always sink
    to the bottom.
    """
    import math

    if not items:
        return items

    keyed: list[tuple[float, int, dict]] = []
    for idx, item in enumerate(items):
        try:
            w = float(weight_fn(item))
        except Exception:
            w = 0.0
        if w <= 0:
            w = 0.01
        # log-domain to avoid underflow when weight is large
        u = random.random() or 1e-12
        key = math.log(u) / w
        keyed.append((key, idx, item))

    keyed.sort(key=lambda t: t[0], reverse=True)
    return [t[2] for t in keyed]


def _interleave_by_category(events_by_cat: dict) -> list[dict]:
    """
    Build a round-robin list that alternates categories so the user
    sees a diverse mix rather than all Thai restaurants in a row.
    Each category's items are still ordered by relevance_score internally.
    """
    queues: dict[str, list[dict]] = {}
    for cat, items in events_by_cat.items():
        sorted_items = sorted(items, key=lambda e: e.get("relevance_score", 0), reverse=True)
        if sorted_items:
            queues[cat] = sorted_items

    result: list[dict] = []
    while queues:
        empty_cats = []
        for cat in list(queues.keys()):
            items = queues[cat]
            if items:
                result.append(items.pop(0))
            if not items:
                empty_cats.append(cat)
        for cat in empty_cats:
            del queues[cat]

    return result


@app.post("/curated_event_pick")
async def curated_event_pick(
    req: CuratedPickRequest,
    authorization: Optional[str] = Header(default=None),
):
    """
    Return a single curated event/restaurant for the user, respecting exclusions.
    Interleaves categories for variety. Deduplicates by both ID and name.
    """
    initialize_firebase_if_needed()
    if firebase_db is None:
        raise HTTPException(status_code=501, detail="Firebase not configured")

    decoded = _verify_and_get_user(authorization)
    email = decoded.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="Email not found in token")

    try:
        doc = firebase_db.collection("user_curated_events").document(email).get()
        if not doc.exists:
            raise HTTPException(status_code=404, detail="No curated events available")

        data = doc.to_dict()
        events_by_cat = _filter_events_by_category(data.get("events_by_category", {}))

        excluded_ids = set(req.excluded_ids)
        excluded_names = {n.strip().lower() for n in req.excluded_names if n.strip()}

        if req.category:
            sorted_cat = sorted(
                events_by_cat.get(req.category, []),
                key=lambda e: e.get("relevance_score", 0),
                reverse=True,
            )
            candidates = _interleave_by_key(
                sorted_cat,
                key_fn=lambda e: (e.get("genre") or e.get("source") or "other").lower(),
            )
        else:
            candidates = _interleave_by_category(events_by_cat)

        # Randomize the candidate order so the user doesn't see the same
        # sequence on every refresh / "next" tap. We still want a slight
        # bias toward higher-relevance picks, so we use a Fisher-Yates
        # shuffle weighted by relevance_score (a small constant is added
        # to avoid zero-weight items being permanently sunk to the end).
        candidates = _weighted_shuffle(
            candidates, weight_fn=lambda e: e.get("relevance_score", 0) or 0
        )

        seen_names: set[str] = set()
        for event in candidates:
            eid = event.get("source_id", event.get("name", ""))
            ename = (event.get("name") or "").strip().lower()

            if eid in excluded_ids:
                continue
            if ename and ename in excluded_names:
                continue
            if ename and ename in seen_names:
                continue
            seen_names.add(ename)
            return event

        raise HTTPException(status_code=404, detail="No more curated events")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/curated_categories")
async def curated_categories(authorization: Optional[str] = Header(default=None)):
    """Return the available curated categories and their event counts for the user."""
    initialize_firebase_if_needed()
    if firebase_db is None:
        raise HTTPException(status_code=501, detail="Firebase not configured")

    decoded = _verify_and_get_user(authorization)
    email = decoded.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="Email not found in token")

    try:
        doc = firebase_db.collection("user_curated_events").document(email).get()
        if not doc.exists:
            return {"categories": {}, "total_events": 0}

        data = doc.to_dict()
        events_by_cat = _filter_events_by_category(data.get("events_by_category", {}))
        categories = {cat: len(events) for cat, events in events_by_cat.items()}
        return {
            "categories": categories,
            "total_events": sum(categories.values()),
            "last_updated": data.get("last_updated"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reshuffle_curated")
async def reshuffle_curated(
    authorization: Optional[str] = Header(default=None),
    novelty_ratio: float = 0.3,
):
    """
    Reshuffle the user's curated events by swapping out novelty items with
    fresh random picks from the parsed_events_catalog.  This is a cheap
    operation that doesn't re-run the full scraping pipeline.
    """
    import random as _rand
    from datetime import datetime as _dt

    initialize_firebase_if_needed()
    if firebase_db is None:
        raise HTTPException(status_code=501, detail="Firebase not configured")

    decoded = _verify_and_get_user(authorization)
    email = decoded.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="Email not found in token")

    try:
        doc = firebase_db.collection("user_curated_events").document(email).get()
        if not doc.exists:
            raise HTTPException(status_code=404, detail="No curated events to reshuffle")

        data = doc.to_dict()
        events_by_cat = data.get("events_by_category", {})

        # Anchor the reshuffle pool to the user's region. Without this the
        # novelty injection silently mixes in events from other cities.
        from service.home_payload import (
            DEFAULT_USER_RADIUS_MILES,
            event_is_within_user_radius,
        )

        user_lat: Optional[float] = None
        user_lon: Optional[float] = None
        try:
            prefs_doc = firebase_db.collection("user_preferences").document(email).get()
            if prefs_doc.exists:
                prefs = prefs_doc.to_dict() or {}
                lat_raw = prefs.get("last_latitude")
                lon_raw = prefs.get("last_longitude")
                if lat_raw is not None and lon_raw is not None:
                    user_lat = float(lat_raw)
                    user_lon = float(lon_raw)
        except (TypeError, ValueError):
            user_lat = None
            user_lon = None

        catalog_docs = firebase_db.collection("parsed_events_catalog").stream()
        catalog_by_cat: dict[str, list[dict]] = {}
        for cdoc in catalog_docs:
            ev = cdoc.to_dict()
            if not _event_passes_filters(ev):
                continue
            if not event_is_within_user_radius(
                ev, user_lat, user_lon, DEFAULT_USER_RADIUS_MILES
            ):
                continue
            cat = ev.get("category") or "other"
            if cat == "food" and ev.get("source") != "google_places":
                cat = "other"
            catalog_by_cat.setdefault(cat, []).append(ev)

        swapped = 0
        for cat, items in events_by_cat.items():
            existing_ids = {
                (e.get("source", "") + ":" + (e.get("source_id") or e.get("name", "")))
                for e in items
            }

            for e in items:
                e.pop("is_novelty", None)

            non_novelty = [e for e in items if not e.get("is_novelty")]
            novelty_count = max(1, int(len(items) * novelty_ratio))

            pool = [
                e for e in catalog_by_cat.get(cat, [])
                if (e.get("source", "") + ":" + (e.get("source_id") or e.get("name", ""))) not in existing_ids
            ]

            if pool:
                sample_size = min(novelty_count, len(pool))
                new_novelty = _rand.sample(pool, sample_size)
                for e in new_novelty:
                    e["is_novelty"] = True
                    e["relevance_score"] = round(_rand.uniform(0.3, 1.5), 2)
                swapped += sample_size
            else:
                new_novelty = []

            _rand.shuffle(non_novelty)
            events_by_cat[cat] = non_novelty + new_novelty

        total = sum(len(v) for v in events_by_cat.values())
        firebase_db.collection("user_curated_events").document(email).update({
            "events_by_category": events_by_cat,
            "total_events": total,
            "last_updated": _dt.utcnow().isoformat() + "Z",
            "last_reshuffled": _dt.utcnow().isoformat() + "Z",
        })

        # Re-emit the home payload so the client's Firestore stream picks up
        # the reshuffled list without needing to re-fetch over HTTP.
        try:
            from service.home_payload import write_home_payload_safe

            write_home_payload_safe(email)
        except Exception as e:
            print(f"home_payload refresh failed for {email}: {e}")

        return {
            "status": "reshuffled",
            "items_swapped": swapped,
            "total_events": total,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/featured_picks")
async def featured_picks(
    authorization: Optional[str] = Header(default=None),
    count: int = 12,
    exclude_ids: Optional[str] = None,
):
    """
    Return a random sample of events + movies + TV for the Featured carousel.
    Read-only -- no Firestore writes.
    """
    import random as _rand

    initialize_firebase_if_needed()
    if firebase_db is None:
        raise HTTPException(status_code=501, detail="Firebase not configured")

    decoded = _verify_and_get_user(authorization)
    email = decoded.get("email")

    exclude_set: set[str] = set()
    if exclude_ids:
        exclude_set = {eid.strip() for eid in exclude_ids.split(",") if eid.strip()}

    event_slots = max(1, count - 2)
    movie_slots = 1
    tv_slots = 1

    # Resolve the user's stored location so the carousel only surfaces events
    # near them. Without this, /featured_picks returns a global random sample.
    from service.home_payload import (
        DEFAULT_USER_RADIUS_MILES,
        event_is_within_user_radius,
    )

    user_lat: Optional[float] = None
    user_lon: Optional[float] = None
    if email:
        try:
            prefs_doc = firebase_db.collection("user_preferences").document(email).get()
            if prefs_doc.exists:
                prefs = prefs_doc.to_dict() or {}
                lat_raw = prefs.get("last_latitude")
                lon_raw = prefs.get("last_longitude")
                if lat_raw is not None and lon_raw is not None:
                    user_lat = float(lat_raw)
                    user_lon = float(lon_raw)
        except (TypeError, ValueError):
            user_lat = None
            user_lon = None

    # --- Events from parsed_events_catalog ---
    catalog_by_cat: dict[str, list[dict]] = {}
    for cdoc in firebase_db.collection("parsed_events_catalog").stream():
        ev = cdoc.to_dict()
        if not _event_passes_filters(ev):
            continue
        if not event_is_within_user_radius(
            ev, user_lat, user_lon, DEFAULT_USER_RADIUS_MILES
        ):
            continue
        eid = ev.get("source", "") + ":" + (ev.get("source_id") or ev.get("name", ""))
        if eid in exclude_set:
            continue
        cat = ev.get("category") or "other"
        if cat == "food" and ev.get("source") != "google_places":
            cat = "other"
        catalog_by_cat.setdefault(cat, []).append(ev)

    featured: list[dict] = []
    cats = list(catalog_by_cat.keys())
    _rand.shuffle(cats)
    per_cat = max(1, event_slots // max(len(cats), 1))
    for cat in cats:
        pool = catalog_by_cat[cat]
        sample_size = min(per_cat, len(pool))
        picks = _rand.sample(pool, sample_size)
        for p in picks:
            p["_type"] = "event"
        featured.extend(picks)

    _rand.shuffle(featured)
    featured = featured[:event_slots]

    # --- Movies ---
    if email:
        movie_ids = _get_precomputed_genre_ids(email, "movie")
        if movie_ids:
            sample_ids = _rand.sample(movie_ids, min(movie_slots * 3, len(movie_ids)))
            for mid in sample_ids:
                mid_str = str(mid)
                if mid_str in exclude_set:
                    continue
                detail = get_movie_detail(mid, "movie")
                if detail:
                    featured.append({
                        "_type": "movie",
                        "id": mid,
                        "movie_id": mid,
                        "title": detail.get("title") or detail.get("original_title"),
                        **{k: v for k, v in detail.items() if v is not None},
                    })
                    if sum(1 for f in featured if f.get("_type") == "movie") >= movie_slots:
                        break

    # --- TV ---
    if email:
        tv_ids = _get_precomputed_genre_ids(email, "tv")
        if tv_ids:
            sample_ids = _rand.sample(tv_ids, min(tv_slots * 3, len(tv_ids)))
            for tid in sample_ids:
                tid_str = str(tid)
                if tid_str in exclude_set:
                    continue
                detail = get_movie_detail(tid, "tv")
                if detail:
                    featured.append({
                        "_type": "tv",
                        "id": tid,
                        "tv_id": tid,
                        "name": detail.get("name") or detail.get("original_name"),
                        **{k: v for k, v in detail.items() if v is not None},
                    })
                    if sum(1 for f in featured if f.get("_type") == "tv") >= tv_slots:
                        break

    return {"featured": featured, "count": len(featured)}


@app.get("/home_payload")
async def get_home_payload(
    authorization: Optional[str] = Header(default=None),
    refresh: bool = False,
):
    """
    One-shot fallback for the precomputed home payload.

    Normally the Flutter client subscribes directly to Firestore at
    `user_home_payload/{email}` for live updates. This endpoint is used:
      * on first-ever signup, before any cron run has produced a doc, or
      * by clients that can't talk to Firestore directly, or
      * when `refresh=true` is passed to force a rebuild on demand.
    """
    initialize_firebase_if_needed()
    if firebase_db is None:
        raise HTTPException(status_code=501, detail="Firebase not configured")

    decoded = _verify_and_get_user(authorization)
    email = decoded.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="Email not found in token")

    from service.home_payload import (
        HOME_PAYLOAD_COLLECTION,
        build_home_payload,
        write_home_payload,
    )

    if refresh:
        payload = write_home_payload(email)
        if payload is None:
            raise HTTPException(status_code=500, detail="Failed to build home payload")
        return payload

    doc = firebase_db.collection(HOME_PAYLOAD_COLLECTION).document(email).get()
    if doc.exists:
        return doc.to_dict()

    # No cached payload yet (e.g. brand-new signup). Build it on demand and
    # persist so subsequent stream subscribers see the same data.
    payload = write_home_payload(email)
    if payload is None:
        raise HTTPException(status_code=500, detail="Failed to build home payload")
    return payload


class UpdateLocationRequest(BaseModel):
    latitude: float
    longitude: float


# Per-user cooldown (seconds) for location-driven re-scrapes. Without this a
# noisy location stream from the client could trigger a fresh per-user pipeline
# every few seconds while the user is in transit.
_LOCATION_RESCRAPE_COOLDOWN_SECONDS = 5 * 60
# Distance (miles) the user has to move before we treat the location update as
# meaningful enough to re-scrape. Anything smaller is just GPS jitter.
_LOCATION_RESCRAPE_THRESHOLD_MILES = 10.0
_location_rescrape_last_run: dict[str, float] = {}
_location_rescrape_lock = threading.Lock()


def _haversine_miles_simple(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    import math

    R = 3958.8
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _per_user_pipeline_safe(email: str) -> None:
    """
    Background entrypoint for the location-driven per-user re-scrape. Always
    swallows exceptions so a failure here can never bring down the request
    handler that scheduled it.
    """
    try:
        from service.events_parser.pipeline import run_pipeline_for_user

        run_pipeline_for_user(email)
    except Exception as e:
        print(f"[per_user_pipeline] {email} failed: {e}")


@app.post("/user_preferences/location")
async def update_user_location(
    req: UpdateLocationRequest,
    authorization: Optional[str] = Header(default=None),
):
    """Store the user's last known location for background event curation.

    If the location moved by more than `_LOCATION_RESCRAPE_THRESHOLD_MILES`
    since the last stored position (or there was no prior position), kicks off
    a per-user pipeline rerun in a background thread so the home payload
    reflects the new region without waiting for the next hourly cron.
    """
    initialize_firebase_if_needed()
    if firebase_db is None:
        raise HTTPException(status_code=501, detail="Firebase not configured")

    decoded = _verify_and_get_user(authorization)
    email = decoded.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="Email not found in token")

    try:
        prefs_ref = firebase_db.collection("user_preferences").document(email)

        prev_snapshot = prefs_ref.get()
        prev = prev_snapshot.to_dict() or {} if prev_snapshot.exists else {}
        prev_lat = prev.get("last_latitude")
        prev_lon = prev.get("last_longitude")

        prefs_ref.update({
            "last_latitude": req.latitude,
            "last_longitude": req.longitude,
            "location_updated_at": datetime.utcnow().isoformat() + "Z",
        })

        moved_miles: Optional[float] = None
        try:
            if prev_lat is not None and prev_lon is not None:
                moved_miles = _haversine_miles_simple(
                    float(prev_lat), float(prev_lon),
                    req.latitude, req.longitude,
                )
        except (TypeError, ValueError):
            moved_miles = None

        meaningful_move = (
            prev_lat is None
            or prev_lon is None
            or (moved_miles is not None and moved_miles >= _LOCATION_RESCRAPE_THRESHOLD_MILES)
        )

        rescrape_started = False
        if meaningful_move:
            with _location_rescrape_lock:
                last = _location_rescrape_last_run.get(email)
                now_ts = time.time()
                if last is None or (now_ts - last) >= _LOCATION_RESCRAPE_COOLDOWN_SECONDS:
                    _location_rescrape_last_run[email] = now_ts
                    rescrape_started = True

            if rescrape_started:
                print(
                    f"[location] {email} moved "
                    f"{'(no prior coords)' if moved_miles is None else f'{moved_miles:.1f}mi'} "
                    "-- starting per-user pipeline"
                )
                threading.Thread(
                    target=_per_user_pipeline_safe,
                    args=(email,),
                    daemon=True,
                ).start()

        return {
            "message": "Location updated",
            "moved_miles": moved_miles,
            "rescrape_started": rescrape_started,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Friends & Auth Users ----------
# Firestore: friend_requests (doc id auto), friends/{uid} with map "friends": { friendUid: { addedAt } }
# Pub/Sub topic for friend request notifications (subscription can send FCM).

FRIEND_REQUESTS_COLLECTION = "friend_requests"
FRIENDS_COLLECTION = "friends"
PUBSUB_FRIEND_REQUEST_TOPIC = os.getenv("PUBSUB_FRIEND_REQUEST_TOPIC", "friend-requests")

_pubsub_publisher = None


def _get_pubsub_publisher():
    global _pubsub_publisher
    if _pubsub_publisher is not None:
        return _pubsub_publisher
    try:
        from google.cloud import pubsub_v1
        project_id = os.getenv("FIREBASE_PROJECT_ID", os.getenv("GOOGLE_CLOUD_PROJECT"))
        if project_id:
            _pubsub_publisher = pubsub_v1.PublisherClient()
            return _pubsub_publisher
    except Exception as e:
        print(f"Pub/Sub publisher init skipped: {e}")
    return None


def _publish_friend_request_notification(to_uid: str, from_uid: str, from_display_name: str, request_id: str) -> None:
    """Publish a message to the friend-requests topic for push notification delivery."""
    publisher = _get_pubsub_publisher()
    if not publisher:
        return
    try:
        import json
        project_id = os.getenv("FIREBASE_PROJECT_ID", os.getenv("GOOGLE_CLOUD_PROJECT"))
        if not project_id:
            return
        topic_path = publisher.topic_path(project_id, PUBSUB_FRIEND_REQUEST_TOPIC)
        payload = json.dumps({
            "to_uid": to_uid,
            "from_uid": from_uid,
            "from_display_name": from_display_name or "",
            "request_id": request_id,
        }).encode("utf-8")
        publisher.publish(topic_path, payload)
    except Exception as e:
        print(f"Pub/Sub publish failed: {e}")


class SendFriendRequestRequest(BaseModel):
    to_uid: str


class RespondFriendRequestRequest(BaseModel):
    request_id: str
    accept: bool


@app.get("/auth_users")
async def list_auth_users(
    authorization: Optional[str] = Header(default=None),
    page_size: int = 100,
    page_token: Optional[str] = None,
):
    """
    List Firebase Auth users for the "add friends" flow.
    Returns uid, email, display_name, photo_url. Excludes current user and users already friends or with pending request.
    """
    initialize_firebase_if_needed()
    decoded = _verify_and_get_user(authorization)
    current_uid = decoded.get("uid")
    if not current_uid:
        raise HTTPException(status_code=400, detail="UID not found in token")

    # First page: pass page_token=None. Firebase Auth rejects empty string.
    next_token = (page_token or "").strip() or None
    try:
        page = firebase_auth.list_users(
            max_results=min(page_size, 100),
            page_token=next_token,
        )
    except Exception as e:
        print(f"auth_users list_users error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list users: {e}")

    # Load current user's friends and pending (sent + received) to exclude
    my_friend_uids: Set[str] = set()
    pending_with: Set[str] = set()
    if firebase_db:
        try:
            friend_doc = firebase_db.collection(FRIENDS_COLLECTION).document(current_uid).get()
            if friend_doc.exists:
                friends_map = friend_doc.to_dict().get("friends") or {}
                my_friend_uids = set(friends_map.keys())
            # Pending: from_uid or to_uid in friend_requests where status == pending
            reqs = firebase_db.collection(FRIEND_REQUESTS_COLLECTION).where(
                "status", "==", "pending"
            ).stream()
            for r in reqs:
                data = r.to_dict()
                from_uid = data.get("from_uid")
                to_uid = data.get("to_uid")
                if from_uid == current_uid:
                    pending_with.add(to_uid)
                elif to_uid == current_uid:
                    pending_with.add(from_uid)
        except Exception:
            pass

    exclude_uids = my_friend_uids | pending_with | {current_uid}
    users = []
    for user in page.users:
        if user.uid in exclude_uids:
            continue
        users.append({
            "uid": user.uid,
            "email": user.email or "",
            "display_name": (user.display_name or "").strip() or (user.email or "Unknown"),
            "photo_url": user.photo_url or "",
        })
    next_page_token = page.next_page_token if hasattr(page, "next_page_token") else None
    return {"users": users, "next_page_token": next_page_token}


@app.post("/friend_requests")
async def send_friend_request(
    req: SendFriendRequestRequest,
    authorization: Optional[str] = Header(default=None),
):
    """Create a pending friend request and publish to Pub/Sub for notification."""
    initialize_firebase_if_needed()
    if firebase_db is None:
        raise HTTPException(status_code=501, detail="Firebase not configured")

    decoded = _verify_and_get_user(authorization)
    from_uid = decoded.get("uid")
    from_email = decoded.get("email") or ""
    from_display_name = (decoded.get("name") or from_email or "Someone").strip()
    if not from_uid:
        raise HTTPException(status_code=400, detail="UID not found in token")
    to_uid = (req.to_uid or "").strip()
    if not to_uid or to_uid == from_uid:
        raise HTTPException(status_code=400, detail="Invalid to_uid")

    # Already friends?
    from_friend_doc = firebase_db.collection(FRIENDS_COLLECTION).document(from_uid).get()
    if from_friend_doc.exists:
        friends_map = from_friend_doc.to_dict().get("friends") or {}
        if to_uid in friends_map:
            return {"message": "Already friends", "request_id": None}

    # Existing pending request (either direction)?
    existing = firebase_db.collection(FRIEND_REQUESTS_COLLECTION).where(
        "from_uid", "==", from_uid
    ).where("to_uid", "==", to_uid).where("status", "==", "pending").limit(1).stream()
    for _ in existing:
        return {"message": "Request already sent", "request_id": None}
    existing = firebase_db.collection(FRIEND_REQUESTS_COLLECTION).where(
        "from_uid", "==", to_uid
    ).where("to_uid", "==", from_uid).where("status", "==", "pending").limit(1).stream()
    for _ in existing:
        return {"message": "They already sent you a request", "request_id": None}

    now = datetime.utcnow().isoformat() + "Z"
    ref = firebase_db.collection(FRIEND_REQUESTS_COLLECTION).document()
    ref.set({
        "from_uid": from_uid,
        "to_uid": to_uid,
        "status": "pending",
        "created_at": now,
        "from_display_name": from_display_name,
        "from_email": from_email,
    })
    request_id = ref.id
    _publish_friend_request_notification(to_uid, from_uid, from_display_name, request_id)
    return {"message": "Friend request sent", "request_id": request_id}


@app.get("/friend_requests/incoming")
async def get_incoming_friend_requests(authorization: Optional[str] = Header(default=None)):
    """List pending friend requests where current user is the recipient."""
    initialize_firebase_if_needed()
    if firebase_db is None:
        raise HTTPException(status_code=501, detail="Firebase not configured")

    decoded = _verify_and_get_user(authorization)
    to_uid = decoded.get("uid")
    if not to_uid:
        raise HTTPException(status_code=400, detail="UID not found in token")

    snapshot = firebase_db.collection(FRIEND_REQUESTS_COLLECTION).where(
        "to_uid", "==", to_uid
    ).where("status", "==", "pending").stream()
    requests = []
    for doc in snapshot:
        data = doc.to_dict()
        data["request_id"] = doc.id
        requests.append(data)
    return {"requests": requests}


@app.post("/friend_requests/respond")
async def respond_to_friend_request(
    req: RespondFriendRequestRequest,
    authorization: Optional[str] = Header(default=None),
):
    """Accept or decline a friend request. On accept, add both users to each other's friends map."""
    initialize_firebase_if_needed()
    if firebase_db is None:
        raise HTTPException(status_code=501, detail="Firebase not configured")

    decoded = _verify_and_get_user(authorization)
    current_uid = decoded.get("uid")
    if not current_uid:
        raise HTTPException(status_code=400, detail="UID not found in token")

    doc_ref = firebase_db.collection(FRIEND_REQUESTS_COLLECTION).document(req.request_id)
    doc = doc_ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Request not found")
    data = doc.to_dict()
    if data.get("to_uid") != current_uid:
        raise HTTPException(status_code=403, detail="Not the recipient")
    if data.get("status") != "pending":
        return {"message": "Request already handled", "accepted": False}

    status = "accepted" if req.accept else "declined"
    doc_ref.update({"status": status, "responded_at": datetime.utcnow().isoformat() + "Z"})

    if req.accept:
        from_uid = data.get("from_uid")
        to_uid = data.get("to_uid")
        now = datetime.utcnow().isoformat() + "Z"
        for uid, other_uid in [(from_uid, to_uid), (to_uid, from_uid)]:
            friend_ref = firebase_db.collection(FRIENDS_COLLECTION).document(uid)
            doc = friend_ref.get()
            friends_map = (doc.to_dict().get("friends") or {}) if doc.exists else {}
            friends_map[other_uid] = {"addedAt": now}
            friend_ref.set({"friends": friends_map})

    return {"message": "Accepted" if req.accept else "Declined", "accepted": req.accept}


@app.get("/friends")
async def get_my_friends(authorization: Optional[str] = Header(default=None)):
    """Return the current user's friends map: { friendUid: { addedAt } }."""
    initialize_firebase_if_needed()
    if firebase_db is None:
        raise HTTPException(status_code=501, detail="Firebase not configured")

    decoded = _verify_and_get_user(authorization)
    uid = decoded.get("uid")
    if not uid:
        raise HTTPException(status_code=400, detail="UID not found in token")

    doc = firebase_db.collection(FRIENDS_COLLECTION).document(uid).get()
    if not doc.exists:
        return {"friends": {}}
    data = doc.to_dict()
    return {"friends": data.get("friends") or {}}


@app.get("/friends/list")
async def get_friends_list(authorization: Optional[str] = Header(default=None)):
    """Return list of friends with display names for invite UIs. Each item: { uid, display_name, email }."""
    initialize_firebase_if_needed()
    if firebase_db is None:
        raise HTTPException(status_code=501, detail="Firebase not configured")

    decoded = _verify_and_get_user(authorization)
    uid = decoded.get("uid")
    if not uid:
        raise HTTPException(status_code=400, detail="UID not found in token")

    doc = firebase_db.collection(FRIENDS_COLLECTION).document(uid).get()
    friends_map = (doc.to_dict().get("friends") or {}) if doc.exists else {}
    result = []
    for friend_uid in friends_map:
        try:
            user_record = firebase_auth.get_user(friend_uid)
            result.append({
                "uid": friend_uid,
                "display_name": (user_record.display_name or "").strip() or (user_record.email or "Friend"),
                "email": user_record.email or "",
            })
        except Exception:
            result.append({"uid": friend_uid, "display_name": "Friend", "email": ""})
    return {"friends": result}


# ---------- Friend activity feed ----------

PUBSUB_FRIEND_ACTIVITY_TOPIC = os.getenv("PUBSUB_FRIEND_ACTIVITY_TOPIC", "friend-activity")


def _publish_friend_activity(actor_uid: str, actor_name: str, activity_type: str, event_name: str, event_id: str, friend_uids: list[str]) -> None:
    """Publish a friend-activity message for each friend who should be notified."""
    publisher = _get_pubsub_publisher()
    if not publisher or not friend_uids:
        return
    try:
        import json as _json
        project_id = os.getenv("FIREBASE_PROJECT_ID", os.getenv("GOOGLE_CLOUD_PROJECT"))
        if not project_id:
            return
        topic_path = publisher.topic_path(project_id, PUBSUB_FRIEND_ACTIVITY_TOPIC)
        payload = _json.dumps({
            "actor_uid": actor_uid,
            "actor_name": actor_name,
            "activity_type": activity_type,
            "event_name": event_name,
            "event_id": event_id,
            "notify_uids": friend_uids,
        }).encode("utf-8")
        publisher.publish(topic_path, payload)
    except Exception as e:
        print(f"Friend activity publish failed: {e}")


@app.get("/friends/feed")
async def friends_feed(authorization: Optional[str] = Header(default=None)):
    """
    Return a chronological activity feed of friends' planned events.
    Shows events created by friends and events friends are invited to.
    """
    initialize_firebase_if_needed()
    if firebase_db is None:
        raise HTTPException(status_code=501, detail="Firebase not configured")

    decoded = _verify_and_get_user(authorization)
    uid = decoded.get("uid")
    if not uid:
        raise HTTPException(status_code=400, detail="UID not found in token")

    friend_doc = firebase_db.collection(FRIENDS_COLLECTION).document(uid).get()
    if not friend_doc.exists:
        return {"feed": [], "friends_count": 0}
    friends_map = friend_doc.to_dict().get("friends") or {}
    friend_uids = list(friends_map.keys())
    if not friend_uids:
        return {"feed": [], "friends_count": 0}

    feed: list[dict] = []
    seen_ids: set[str] = set()

    for batch_start in range(0, len(friend_uids), 10):
        batch = friend_uids[batch_start:batch_start + 10]
        created = firebase_db.collection("planned_events").where(
            "created_by_uid", "in", batch
        ).stream()
        for doc in created:
            if doc.id in seen_ids:
                continue
            seen_ids.add(doc.id)
            d = doc.to_dict()
            d["event_id"] = doc.id
            d["feed_type"] = "created"
            feed.append(d)

        invited = firebase_db.collection("planned_events").where(
            "invited_uids", "array_contains_any", batch
        ).stream()
        for doc in invited:
            if doc.id in seen_ids:
                continue
            seen_ids.add(doc.id)
            d = doc.to_dict()
            d["event_id"] = doc.id
            d["feed_type"] = "invited"
            feed.append(d)

    feed.sort(key=lambda e: e.get("created_at", ""), reverse=True)
    return {"feed": feed[:50], "friends_count": len(friend_uids)}


# ---------- Planned events (create from event/place/restaurant, invite friends) ----------
PLANNED_EVENTS_COLLECTION = "planned_events"


class CreatePlannedEventRequest(BaseModel):
    name: str
    event_date: str  # ISO date or YYYY-MM-DD
    event_time: str  # HH:MM or HH:MM:SS
    description: Optional[str] = None
    venue_name: Optional[str] = None
    venue_address: Optional[str] = None
    image_url: Optional[str] = None
    url: Optional[str] = None
    source_type: Optional[str] = None  # event | place | restaurant
    invited_uids: List[str] = []


@app.post("/planned_events")
async def create_planned_event(
    req: CreatePlannedEventRequest,
    authorization: Optional[str] = Header(default=None),
):
    """Create a planned event and invite friends. Stored in Firestore planned_events."""
    initialize_firebase_if_needed()
    if firebase_db is None:
        raise HTTPException(status_code=501, detail="Firebase not configured")

    decoded = _verify_and_get_user(authorization)
    created_by_uid = decoded.get("uid")
    created_by_name = (decoded.get("name") or decoded.get("email") or "Someone").strip()
    if not created_by_uid:
        raise HTTPException(status_code=400, detail="UID not found in token")

    invited_uids = [u for u in (req.invited_uids or []) if u and u != created_by_uid]
    now = datetime.utcnow().isoformat() + "Z"
    ref = firebase_db.collection(PLANNED_EVENTS_COLLECTION).document()
    ref.set({
        "created_by_uid": created_by_uid,
        "created_by_name": created_by_name,
        "name": req.name,
        "event_date": req.event_date,
        "event_time": req.event_time,
        "description": req.description,
        "venue_name": req.venue_name,
        "venue_address": req.venue_address,
        "image_url": req.image_url,
        "url": req.url,
        "source_type": req.source_type or "event",
        "invited_uids": invited_uids,
        "created_at": now,
    })

    all_friends_to_notify = list(invited_uids)
    if firebase_db:
        try:
            friend_doc = firebase_db.collection(FRIENDS_COLLECTION).document(created_by_uid).get()
            if friend_doc.exists:
                all_friends = list((friend_doc.to_dict().get("friends") or {}).keys())
                all_friends_to_notify = list(set(all_friends_to_notify + all_friends))
        except Exception:
            pass

    if all_friends_to_notify:
        _publish_friend_activity(
            actor_uid=created_by_uid,
            actor_name=created_by_name,
            activity_type="created_event",
            event_name=req.name,
            event_id=ref.id,
            friend_uids=[u for u in all_friends_to_notify if u != created_by_uid],
        )

    return {"event_id": ref.id, "message": "Event created"}


@app.get("/planned_events")
async def get_planned_events(authorization: Optional[str] = Header(default=None)):
    """Return events created by the user and events where the user is invited."""
    initialize_firebase_if_needed()
    if firebase_db is None:
        raise HTTPException(status_code=501, detail="Firebase not configured")

    decoded = _verify_and_get_user(authorization)
    uid = decoded.get("uid")
    if not uid:
        raise HTTPException(status_code=400, detail="UID not found in token")

    created = firebase_db.collection(PLANNED_EVENTS_COLLECTION).where(
        "created_by_uid", "==", uid
    ).stream()
    invited = firebase_db.collection(PLANNED_EVENTS_COLLECTION).where(
        "invited_uids", "array_contains", uid
    ).stream()
    seen_ids = set()
    events = []
    for doc in created:
        if doc.id not in seen_ids:
            seen_ids.add(doc.id)
            d = doc.to_dict()
            d["event_id"] = doc.id
            d["is_creator"] = True
            events.append(d)
    for doc in invited:
        if doc.id not in seen_ids:
            seen_ids.add(doc.id)
            d = doc.to_dict()
            d["event_id"] = doc.id
            d["is_creator"] = False
            events.append(d)
    events.sort(key=lambda e: (e.get("event_date") or "", e.get("event_time") or ""))
    return {"events": events}


class JoinEventRequest(BaseModel):
    event_id: str


@app.post("/planned_events/request_join")
async def request_join_event(
    req: JoinEventRequest,
    authorization: Optional[str] = Header(default=None),
):
    """Request to join a friend's planned event."""
    initialize_firebase_if_needed()
    if firebase_db is None:
        raise HTTPException(status_code=501, detail="Firebase not configured")

    decoded = _verify_and_get_user(authorization)
    uid = decoded.get("uid")
    display_name = (decoded.get("name") or decoded.get("email") or "Someone").strip()
    if not uid:
        raise HTTPException(status_code=400, detail="UID not found in token")

    doc = firebase_db.collection(PLANNED_EVENTS_COLLECTION).document(req.event_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Event not found")

    data = doc.to_dict()
    creator_uid = data.get("created_by_uid")
    if uid == creator_uid:
        raise HTTPException(status_code=400, detail="You are the creator of this event")

    existing_requests = data.get("join_requests") or []
    if any(r.get("uid") == uid for r in existing_requests):
        return {"message": "Already requested"}

    invited = data.get("invited_uids") or []
    if uid in invited:
        return {"message": "Already invited"}

    now = datetime.utcnow().isoformat() + "Z"
    firebase_db.collection(PLANNED_EVENTS_COLLECTION).document(req.event_id).update({
        "join_requests": firestore.ArrayUnion([{
            "uid": uid,
            "display_name": display_name,
            "requested_at": now,
            "status": "pending",
        }]),
    })

    if creator_uid:
        _publish_friend_activity(
            actor_uid=uid,
            actor_name=display_name,
            activity_type="join_request",
            event_name=data.get("name", "an event"),
            event_id=req.event_id,
            friend_uids=[creator_uid],
        )

    return {"message": "Join request sent"}


class RespondJoinRequest(BaseModel):
    event_id: str
    requester_uid: str
    accept: bool


@app.post("/planned_events/respond_join")
async def respond_join_request(
    req: RespondJoinRequest,
    authorization: Optional[str] = Header(default=None),
):
    """Accept or decline a join request (only the event creator can do this)."""
    initialize_firebase_if_needed()
    if firebase_db is None:
        raise HTTPException(status_code=501, detail="Firebase not configured")

    decoded = _verify_and_get_user(authorization)
    uid = decoded.get("uid")
    if not uid:
        raise HTTPException(status_code=400, detail="UID not found in token")

    doc_ref = firebase_db.collection(PLANNED_EVENTS_COLLECTION).document(req.event_id)
    doc = doc_ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Event not found")

    data = doc.to_dict()
    if data.get("created_by_uid") != uid:
        raise HTTPException(status_code=403, detail="Only the event creator can respond")

    join_requests = data.get("join_requests") or []
    updated = []
    found = False
    for r in join_requests:
        if r.get("uid") == req.requester_uid:
            found = True
            r["status"] = "accepted" if req.accept else "declined"
            r["responded_at"] = datetime.utcnow().isoformat() + "Z"
        updated.append(r)

    if not found:
        raise HTTPException(status_code=404, detail="Join request not found")

    update_payload: dict = {"join_requests": updated}
    if req.accept:
        update_payload["invited_uids"] = firestore.ArrayUnion([req.requester_uid])

    doc_ref.update(update_payload)
    return {"message": "accepted" if req.accept else "declined"}


class FCMTokenRequest(BaseModel):
    token: str


FCM_TOKENS_COLLECTION = "user_fcm_tokens"


@app.post("/fcm_token")
async def register_fcm_token(
    req: FCMTokenRequest,
    authorization: Optional[str] = Header(default=None),
):
    """Register FCM token for the current user (used by Pub/Sub subscriber to send push notifications)."""
    initialize_firebase_if_needed()
    if firebase_db is None:
        raise HTTPException(status_code=501, detail="Firebase not configured")

    decoded = _verify_and_get_user(authorization)
    uid = decoded.get("uid")
    if not uid or not (req.token or "").strip():
        raise HTTPException(status_code=400, detail="UID or token missing")

    token = req.token.strip()
    firebase_db.collection(FCM_TOKENS_COLLECTION).document(uid).set({
        "tokens": firestore.ArrayUnion([token]),
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }, merge=True)
    return {"message": "Token registered"}


@app.get("/admin/fcm_token_count")
async def fcm_token_count(
    x_admin_token: Optional[str] = Header(default=None),
):
    """Return the number of users with registered FCM tokens."""
    _verify_admin_token(x_admin_token)
    initialize_firebase_if_needed()
    if firebase_db is None:
        raise HTTPException(status_code=501, detail="Firebase not configured")

    total_users = 0
    total_tokens = 0
    for doc in firebase_db.collection(FCM_TOKENS_COLLECTION).stream():
        tokens = (doc.to_dict() or {}).get("tokens") or []
        if tokens:
            total_users += 1
            total_tokens += len(tokens)
    return {"users": total_users, "tokens": total_tokens}


@app.get("/admin/list_fcm_tokens")
async def list_fcm_tokens(
    x_admin_token: Optional[str] = Header(default=None),
):
    """List every user that has at least one FCM token registered.

    Returns UID, email, display name, and the number of tokens for each.
    Useful for picking a UID to use with `make broadcast-uid`.
    """
    _verify_admin_token(x_admin_token)
    initialize_firebase_if_needed()
    if firebase_db is None:
        raise HTTPException(status_code=501, detail="Firebase not configured")

    users: list[dict] = []
    for doc in firebase_db.collection(FCM_TOKENS_COLLECTION).stream():
        tokens = (doc.to_dict() or {}).get("tokens") or []
        if not tokens:
            continue
        uid = doc.id
        try:
            user_record = firebase_auth.get_user(uid)
            email = user_record.email or "(no email)"
            display_name = user_record.display_name or ""
        except Exception:
            email = "(unknown)"
            display_name = ""
        users.append({
            "uid": uid,
            "email": email,
            "display_name": display_name,
            "token_count": len(tokens),
            "token_previews": [
                f"{t[:8]}…{t[-8:]}" if len(t) > 20 else t for t in tokens
            ],
        })

    users.sort(key=lambda u: u["email"])
    return {"users": users, "total_users": len(users)}


@app.post("/admin/purge_all_fcm_tokens")
async def purge_all_fcm_tokens(
    x_admin_token: Optional[str] = Header(default=None),
    confirm: Optional[str] = Header(default=None, alias="X-Confirm"),
):
    """Delete every document in the FCM tokens collection.

    Forces every device to re-register a fresh token on next app launch.
    Useful when stale/orphaned tokens are suspected (e.g. after a bundle ID
    or Firebase config change).

    Requires both X-Admin-Token AND X-Confirm: PURGE headers as a safety net.
    """
    _verify_admin_token(x_admin_token)
    if confirm != "PURGE":
        raise HTTPException(
            status_code=400,
            detail="Refusing to purge without 'X-Confirm: PURGE' header.",
        )
    initialize_firebase_if_needed()
    if firebase_db is None:
        raise HTTPException(status_code=501, detail="Firebase not configured")

    deleted_uids: list[str] = []
    batch = firebase_db.batch()
    batch_size = 0
    for doc in firebase_db.collection(FCM_TOKENS_COLLECTION).stream():
        deleted_uids.append(doc.id)
        batch.delete(doc.reference)
        batch_size += 1
        # Firestore batches are limited to 500 ops; commit and start a new one.
        if batch_size >= 400:
            batch.commit()
            batch = firebase_db.batch()
            batch_size = 0
    if batch_size > 0:
        batch.commit()

    logger.warning("Purged FCM tokens for %d users: %s", len(deleted_uids), deleted_uids)
    return {
        "deleted_users": len(deleted_uids),
        "uids": deleted_uids,
    }


class BroadcastNotificationRequest(BaseModel):
    title: str
    body: str


@app.post("/admin/broadcast_notification")
async def broadcast_notification(
    req: BroadcastNotificationRequest,
    x_admin_token: Optional[str] = Header(default=None),
):
    """Send a push notification to every user with a registered FCM token."""
    _verify_admin_token(x_admin_token)
    initialize_firebase_if_needed()
    if firebase_db is None:
        raise HTTPException(status_code=501, detail="Firebase not configured")

    token_docs = firebase_db.collection(FCM_TOKENS_COLLECTION).stream()
    sent = 0
    failed = 0
    stale_cleaned = 0
    recipients: list[dict] = []

    for doc in token_docs:
        uid = doc.id
        tokens = (doc.to_dict() or {}).get("tokens") or []

        try:
            user_record = firebase_auth.get_user(uid)
            email = user_record.email or "(no email)"
            display_name = user_record.display_name or ""
        except Exception:
            email = "(unknown)"
            display_name = ""

        stale: list[str] = []
        per_user_results: list[dict] = []
        for token in tokens:
            token_preview = f"{token[:8]}…{token[-8:]}" if len(token) > 20 else token
            try:
                message_id = messaging.send(messaging.Message(
                    notification=messaging.Notification(
                        title=req.title,
                        body=req.body,
                    ),
                    data={"type": "broadcast"},
                    token=token,
                    android=messaging.AndroidConfig(
                        notification=messaging.AndroidNotification(
                            channel_id="hourly_picks",
                        ),
                    ),
                    apns=messaging.APNSConfig(
                        headers={
                            "apns-priority": "10",
                            "apns-push-type": "alert",
                        },
                        payload=messaging.APNSPayload(
                            aps=messaging.Aps(
                                alert=messaging.ApsAlert(
                                    title=req.title,
                                    body=req.body,
                                ),
                                sound="default",
                                badge=1,
                            ),
                        ),
                    ),
                ))
                sent += 1
                per_user_results.append({
                    "token": token_preview,
                    "status": "sent",
                    "message_id": message_id,
                })
                logger.info(
                    "Broadcast sent: uid=%s email=%s token=%s message_id=%s",
                    uid, email, token_preview, message_id,
                )
            except messaging.UnregisteredError as e:
                stale.append(token)
                per_user_results.append({
                    "token": token_preview,
                    "status": "stale",
                    "error": str(e),
                })
                logger.warning(
                    "Broadcast stale token: uid=%s email=%s token=%s",
                    uid, email, token_preview,
                )
            except (messaging.ThirdPartyAuthError, messaging.SenderIdMismatchError) as e:
                # Token was generated by a build pinned to a different APNs
                # topic / Firebase sender. It can never deliver — treat as stale
                # and prune so we don't keep retrying on every broadcast.
                stale.append(token)
                per_user_results.append({
                    "token": token_preview,
                    "status": "stale",
                    "error": f"{type(e).__name__}: {e}",
                })
                logger.warning(
                    "Broadcast pruning unrecoverable token: uid=%s email=%s token=%s reason=%s",
                    uid, email, token_preview, type(e).__name__,
                )
            except Exception as e:
                failed += 1
                per_user_results.append({
                    "token": token_preview,
                    "status": "failed",
                    "error": f"{type(e).__name__}: {e}",
                })
                logger.error(
                    "Broadcast failed: uid=%s email=%s token=%s error=%s",
                    uid, email, token_preview, e,
                )

        if stale:
            stale_cleaned += len(stale)
            firebase_db.collection(FCM_TOKENS_COLLECTION).document(uid).update({
                "tokens": firestore.ArrayRemove(stale),
            })

        recipients.append({
            "uid": uid,
            "email": email,
            "display_name": display_name,
            "tokens": per_user_results,
        })

    return {
        "sent": sent,
        "failed": failed,
        "stale_tokens_cleaned": stale_cleaned,
        "recipients": recipients,
    }


class TargetedNotificationRequest(BaseModel):
    uid: str
    title: str
    body: str


@app.post("/admin/broadcast_to_uid")
async def broadcast_to_uid(
    req: TargetedNotificationRequest,
    x_admin_token: Optional[str] = Header(default=None),
):
    """Send a push notification to a single user's tokens (debug helper).

    Useful when triaging delivery problems on a specific device. The response
    includes per-token status with full FCM message IDs so you can correlate
    with APNs delivery receipts.
    """
    _verify_admin_token(x_admin_token)
    initialize_firebase_if_needed()
    if firebase_db is None:
        raise HTTPException(status_code=501, detail="Firebase not configured")

    doc = firebase_db.collection(FCM_TOKENS_COLLECTION).document(req.uid).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail=f"No FCM tokens found for uid {req.uid}")

    tokens = (doc.to_dict() or {}).get("tokens") or []
    if not tokens:
        raise HTTPException(status_code=404, detail=f"User {req.uid} has no registered tokens")

    try:
        user_record = firebase_auth.get_user(req.uid)
        email = user_record.email or "(no email)"
        display_name = user_record.display_name or ""
    except Exception:
        email = "(unknown)"
        display_name = ""

    sent = 0
    failed = 0
    stale: list[str] = []
    per_token: list[dict] = []

    for token in tokens:
        token_preview = f"{token[:8]}…{token[-8:]}" if len(token) > 20 else token
        try:
            message_id = messaging.send(messaging.Message(
                notification=messaging.Notification(
                    title=req.title,
                    body=req.body,
                ),
                data={"type": "broadcast"},
                token=token,
                android=messaging.AndroidConfig(
                    notification=messaging.AndroidNotification(
                        channel_id="hourly_picks",
                    ),
                ),
                apns=messaging.APNSConfig(
                    headers={
                        "apns-priority": "10",
                        "apns-push-type": "alert",
                    },
                    payload=messaging.APNSPayload(
                        aps=messaging.Aps(
                            alert=messaging.ApsAlert(
                                title=req.title,
                                body=req.body,
                            ),
                            sound="default",
                            badge=1,
                        ),
                    ),
                ),
            ))
            sent += 1
            per_token.append({
                "token_full": token,
                "token_preview": token_preview,
                "status": "sent",
                "message_id": message_id,
            })
            logger.info(
                "Targeted send: uid=%s email=%s token=%s message_id=%s",
                req.uid, email, token_preview, message_id,
            )
        except messaging.UnregisteredError as e:
            stale.append(token)
            per_token.append({
                "token_full": token,
                "token_preview": token_preview,
                "status": "stale",
                "error": str(e),
            })
            logger.warning(
                "Targeted send stale token: uid=%s email=%s token=%s",
                req.uid, email, token_preview,
            )
        except (messaging.ThirdPartyAuthError, messaging.SenderIdMismatchError) as e:
            stale.append(token)
            per_token.append({
                "token_full": token,
                "token_preview": token_preview,
                "status": "stale",
                "error": f"{type(e).__name__}: {e}",
            })
            logger.warning(
                "Targeted send pruning unrecoverable token: uid=%s email=%s token=%s reason=%s",
                req.uid, email, token_preview, type(e).__name__,
            )
        except Exception as e:
            failed += 1
            per_token.append({
                "token_full": token,
                "token_preview": token_preview,
                "status": "failed",
                "error": f"{type(e).__name__}: {e}",
            })
            logger.error(
                "Targeted send failed: uid=%s email=%s token=%s error=%s",
                req.uid, email, token_preview, e,
            )

    if stale:
        firebase_db.collection(FCM_TOKENS_COLLECTION).document(req.uid).update({
            "tokens": firestore.ArrayRemove(stale),
        })

    return {
        "uid": req.uid,
        "email": email,
        "display_name": display_name,
        "sent": sent,
        "failed": failed,
        "stale_tokens_cleaned": len(stale),
        "tokens": per_token,
    }


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
        "places.websiteUri",
        "places.googleMapsUri",
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
                "website_url": place.get("websiteUri"),
                "google_maps_url": place.get("googleMapsUri"),
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
    bypass_cache: bool = False,
    only_my_services: bool = False,
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
    movie_ids = user_preferences.get("movieIds", []) or []
    
    # Get disliked movie IDs to exclude from recommendations
    disliked_movie_ids = user_preferences.get("dislikedMovieIds", [])
    excluded_ids = set(disliked_movie_ids) if disliked_movie_ids else None

    # When the user has opted into the "only my services" filter, normalize
    # their selected services into the same lowercase canonical form we use
    # for provider matching. Empty set short-circuits the filter to no-op.
    user_services: Set[str] = set()
    if only_my_services:
        for svc in user_preferences.get("streamingServices", []) or []:
            norm = _normalize_service_name(svc)
            if norm:
                user_services.add(norm)
        if not user_services:
            print("only_my_services=True but user has no streamingServices set; skipping filter")
    
    # Parse genres from comma-separated string (optional; when omitted, no genre filter)
    genre_list = None
    if genres:
        try:
            genre_list = [int(g.strip()) for g in genres.split(",") if g.strip()]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid genres format. Use comma-separated integers.")
    
    # Default year range when not provided (e.g. home screen only sends top_n and offset)
    _current_year = datetime.now().year
    if start_year is None:
        start_year = 1970
    if end_year is None:
        end_year = _current_year + 1
    
    try:
        # Prefer precomputed genre recommendations (for new users or when genres filter provided)
        email = user_preferences.get("email")
        precomputed_ids = (
            _get_precomputed_genre_ids(email, "movie", genre_list)
            if email
            else []
        )

        if precomputed_ids:
            results = []
            skipped = 0
            scanned = 0
            for idx, movie_id in enumerate(precomputed_ids):
                scanned = idx + 1
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
                if user_services and not _detail_matches_user_services(detail, user_services):
                    continue
                # Handle offset for pagination
                if skipped < offset:
                    skipped += 1
                    continue
                # Normalize so client gets top-level id (Movie.fromJson expects id or movie_id + detail)
                results.append({
                    "movie_id": movie_id,
                    "id": movie_id,
                    "title": detail.get("title") or detail.get("original_title"),
                    "similarity": max(0.0, 1.0 - (idx / max(len(precomputed_ids), 1))),
                    "detail": detail,
                    **{k: v for k, v in detail.items() if v is not None},
                })
                if len(results) >= top_n:
                    break
            return {"results": results, "count": len(results), "offset": offset, "has_more": scanned < len(precomputed_ids)}

        # Fallback to dynamic recommendation (requires at least one movie preference)
        if not movie_ids:
            return {"results": [], "count": 0, "offset": offset, "has_more": False}
        # When filtering by services, request a wider pool so we still have
        # enough candidates after dropping titles the user can't stream.
        fetch_size = (top_n + offset) * (4 if user_services else 1)
        results = get_similar_assets(
            asset_ids=set(movie_ids),
            asset_type="movie",
            genres=genre_list,
            start_year=start_year,
            end_year=end_year,
            top_n=max(fetch_size, top_n + offset),
            read_from_local=False,
            bypass_cache=bypass_cache,
            excluded_ids=excluded_ids
        )
        if user_services:
            results = [
                r for r in results
                if _detail_matches_user_services(r.get("detail") or {}, user_services)
            ]
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
    bypass_cache: bool = False,
    only_my_services: bool = False,
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
    tv_ids = user_preferences.get("tvShowIds", []) or []
    
    # Get disliked TV show IDs to exclude from recommendations
    disliked_tv_ids = user_preferences.get("dislikedTvShowIds", [])
    excluded_ids = set(disliked_tv_ids) if disliked_tv_ids else None

    user_services: Set[str] = set()
    if only_my_services:
        for svc in user_preferences.get("streamingServices", []) or []:
            norm = _normalize_service_name(svc)
            if norm:
                user_services.add(norm)
        if not user_services:
            print("only_my_services=True but user has no streamingServices set; skipping filter")
    
    # Parse genres from comma-separated string
    genre_list = None
    if genres:
        try:
            genre_list = [int(g.strip()) for g in genres.split(",")]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid genres format. Use comma-separated integers.")
    
    try:
        # Prefer precomputed genre recommendations (for new users or when genres filter provided)
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
                if user_services and not _detail_matches_user_services(detail, user_services):
                    continue
                results.append({
                    "movie_id": tv_id,
                    "title": detail.get("name") or detail.get("original_name"),
                    "similarity": max(0.0, 1.0 - (idx / max(len(precomputed_ids), 1))),
                    "detail": detail,
                })
                if len(results) >= top_n:
                    break
            return {"results": results, "count": len(results)}

        # Fallback to dynamic recommendation (requires at least one TV preference)
        if not tv_ids:
            return {"results": [], "count": 0}
        fetch_size = top_n * (4 if user_services else 1)
        results = get_similar_assets(
            asset_ids=set(tv_ids),
            asset_type="tv",
            genres=genre_list,
            start_year=start_year,
            end_year=end_year,
            top_n=max(fetch_size, top_n),
            read_from_local=False,
            bypass_cache=bypass_cache,
            excluded_ids=excluded_ids
        )
        if user_services:
            results = [
                r for r in results
                if _detail_matches_user_services(r.get("detail") or {}, user_services)
            ]
            results = results[:top_n]
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
    Returns TMDB provider names plus Watchmode streaming_sources with deeplinks
    (web_url, ios_url, android_url) for Netflix, Amazon Prime, etc.
    """
    if asset_type not in ("movie", "tv"):
        raise HTTPException(status_code=400, detail="asset_type must be 'movie' or 'tv'")

    providers = []
    streaming_sources = []
    try:
        from service.watch_providers_job import get_providers_for_asset
        providers = get_providers_for_asset(asset_id, asset_type, read_from_local=False)
    except Exception as e:
        print(f"Watch providers (TMDB) error: {e}")

    try:
        streaming_sources = _get_watchmode_sources(asset_id, asset_type)
    except Exception as e:
        print(f"Watchmode streaming sources error: {e}")

    return {
        "asset_id": asset_id,
        "asset_type": asset_type,
        "providers": providers,
        "streaming_sources": streaming_sources,
        "count": len(providers) + len(streaming_sources),
    }


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
