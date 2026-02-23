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
ticketmaster_api_key = os.getenv("TICKETMASTER_API_KEY", "")
tmdb_api_key = os.getenv("TMDB_API_KEY", "")
watchmode_api_key = os.getenv("WATCHMODE_API_KEY", "")

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
    default_start = req.start_date or today.strftime("%Y-%m-%dT%H:%M:%SZ")
    default_end = req.end_date or (today + timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")

    events = search_nearby_events(
        latitude=req.latitude,
        longitude=req.longitude,
        radius_miles=req.radius_miles,
        classification=req.classification,
        keyword=req.keyword,
        start_date=default_start,
        end_date=default_end,
        size=20,
    )

    excluded = set(req.excluded_event_ids)
    if excluded:
        events = [e for e in events if e["id"] not in excluded]

    return {"events": events}


class NearbyPlacesBatchRequest(BaseModel):
    latitude: float
    longitude: float
    radius_meters: float = 16093.4  # ~10 miles


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
            sources.append({
                "source_id": s.get("source_id"),
                "name": name,
                "type": stype,
                "web_url": s.get("web_url"),
                "ios_url": s.get("ios_url"),
                "android_url": s.get("android_url"),
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
        return doc.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


PARSED_EVENTS_CATALOG_COLLECTION = "parsed_events_catalog"


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
        events_by_cat = data.get("events_by_category", {})

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
        events_by_cat = data.get("events_by_category", {})
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


class UpdateLocationRequest(BaseModel):
    latitude: float
    longitude: float


@app.post("/user_preferences/location")
async def update_user_location(
    req: UpdateLocationRequest,
    authorization: Optional[str] = Header(default=None),
):
    """Store the user's last known location for background event curation."""
    initialize_firebase_if_needed()
    if firebase_db is None:
        raise HTTPException(status_code=501, detail="Firebase not configured")

    decoded = _verify_and_get_user(authorization)
    email = decoded.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="Email not found in token")

    try:
        firebase_db.collection("user_preferences").document(email).update({
            "last_latitude": req.latitude,
            "last_longitude": req.longitude,
            "location_updated_at": datetime.utcnow().isoformat() + "Z",
        })
        return {"message": "Location updated"}
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
        # Prefer precomputed genre recommendations (only when genres filter is provided)
        email = user_preferences.get("email")
        precomputed_ids = (
            _get_precomputed_genre_ids(email, "movie", genre_list)
            if (email and genre_list)
            else []
        )

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
