"""
Restaurant scraper using Google Places API.

Fetches restaurants near a user's location, filtered by cuisine type.
Runs additional dietary-specific searches (e.g. halal_restaurant) when the
user has matching dietary preferences.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests

from .models import ScrapedEvent

# Mirrors CUISINE_TO_PLACE_TYPES from application.py
CUISINE_TO_PLACE_TYPES: dict[str, list[str]] = {
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

# Google Places types for dietary preferences that have first-class support.
DIETARY_TO_PLACE_TYPES: dict[str, list[str]] = {
    "Halal": ["halal_restaurant"],
    "Vegetarian": ["vegetarian_restaurant"],
    "Vegan": ["vegan_restaurant"],
}

# Cuisines that commonly serve halal food — searched as an extra signal
# when the user has the Halal dietary preference.
HALAL_ADJACENT_CUISINES: list[str] = [
    "pakistani_restaurant",
    "turkish_restaurant",
    "bangladeshi_restaurant",
    "moroccan_restaurant",
    "afghani_restaurant",
    "lebanese_restaurant",
]

# Dietary preference keywords used to tag restaurants from their names/summaries
DIETARY_KEYWORDS: dict[str, list[str]] = {
    "Vegetarian": ["vegetarian", "veggie", "plant-based", "meatless"],
    "Vegan": ["vegan", "plant-based", "plant based"],
    "Gluten-Free": ["gluten-free", "gluten free", "gf", "celiac"],
    "Halal": ["halal"],
    "Kosher": ["kosher"],
    "Dairy-Free": ["dairy-free", "dairy free", "lactose-free"],
    "Nut-Free": ["nut-free", "nut free", "allergy-friendly"],
}


def scrape_restaurants(
    latitude: float,
    longitude: float,
    radius_miles: float = 50.0,
    cuisines: Optional[list[str]] = None,
    dietary: Optional[list[str]] = None,
    max_per_cuisine: int = 10,
) -> list[ScrapedEvent]:
    """
    Fetch restaurants from Google Places API, grouped by cuisine type.

    If cuisines is provided, searches specifically for those cuisine types
    in ranked order. If dietary preferences are provided, runs additional
    dedicated searches (e.g. ``halal_restaurant``) and auto-tags results.
    """
    api_key = os.getenv("GOOGLE_PLACES_API_KEY", "")
    if not api_key:
        print("[restaurants] No Google Places API key, skipping")
        return []

    radius_meters = radius_miles * 1609.34
    results: list[ScrapedEvent] = []
    seen_names: set[str] = set()

    # --- Cuisine-based searches (existing behaviour) ---
    if cuisines:
        for cuisine in cuisines:
            place_types = CUISINE_TO_PLACE_TYPES.get(cuisine, ["restaurant"])
            places = _search_places(
                api_key, latitude, longitude, radius_meters,
                included_types=place_types,
                max_results=max_per_cuisine,
            )
            for p in places:
                if p["name"] not in seen_names:
                    seen_names.add(p["name"])
                    tags = ["food", "restaurant", cuisine.lower()]
                    results.append(_place_to_event(p, tags, cuisine))
    else:
        places = _search_places(
            api_key, latitude, longitude, radius_meters,
            included_types=["restaurant"],
            max_results=20,
        )
        for p in places:
            if p["name"] not in seen_names:
                seen_names.add(p["name"])
                cuisine_tag = _infer_cuisine_from_type(p.get("type", ""))
                tags = ["food", "restaurant"]
                if cuisine_tag:
                    tags.append(cuisine_tag)
                results.append(_place_to_event(p, tags, cuisine_tag))

    # --- Dietary-specific searches ---
    dietary_count = 0
    for diet in (dietary or []):
        diet_types = DIETARY_TO_PLACE_TYPES.get(diet)
        if not diet_types:
            continue

        places = _search_places(
            api_key, latitude, longitude, radius_meters,
            included_types=diet_types,
            max_results=20,
        )
        diet_tag = diet.lower()
        for p in places:
            if p["name"] not in seen_names:
                seen_names.add(p["name"])
                tags = ["food", "restaurant", diet_tag]
                results.append(_place_to_event(p, tags, None))
                dietary_count += 1

        # For Halal, also search cuisine types that *often* serve halal, but
        # only keep / tag a place when name/summary actually mentions "halal".
        # Blindly tagging every Pakistani/Turkish/etc. spot as halal was the
        # main source of non-halal restaurants leaking into curated dining.
        if diet == "Halal":
            for adj_type in HALAL_ADJACENT_CUISINES:
                adj_places = _search_places(
                    api_key, latitude, longitude, radius_meters,
                    included_types=[adj_type],
                    max_results=10,
                )
                for p in adj_places:
                    if p["name"] in seen_names:
                        continue
                    text = f"{p.get('name') or ''} {p.get('summary') or ''}".lower()
                    if "halal" not in text:
                        continue
                    seen_names.add(p["name"])
                    cuisine_label = adj_type.replace("_restaurant", "").replace("_", " ")
                    tags = ["food", "restaurant", "halal", cuisine_label]
                    results.append(_place_to_event(p, tags, cuisine_label))
                    dietary_count += 1

    if dietary_count:
        print(f"[restaurants] Dietary-specific searches added {dietary_count} restaurants")

    print(f"[restaurants] Scraped {len(results)} restaurants total")
    return results


def tag_dietary_matches(
    restaurants: list[ScrapedEvent],
    dietary_preferences: list[str],
) -> list[ScrapedEvent]:
    """
    Tag each restaurant with dietary preference matches based on
    name, description, and type.
    """
    if not dietary_preferences:
        return restaurants

    for r in restaurants:
        text = f"{r.name} {r.description or ''} {' '.join(r.tags)}".lower()
        existing_tags = {t.lower() for t in r.tags}
        for diet in dietary_preferences:
            if diet.lower() in existing_tags:
                continue
            keywords = DIETARY_KEYWORDS.get(diet, [diet.lower()])
            if any(kw in text for kw in keywords):
                r.tags.append(diet.lower())

    return restaurants


def _resolve_place_photo_url(api_key: str, photo_name: str) -> Optional[str]:
    """Resolve a Places photo resource to a durable googleusercontent CDN URL.

    Storing `.../media?key=...` directly fails later with
    `The photo resource in the request is invalid` once the resource name
    goes stale — and the client then hides every restaurant card. Asking for
    `skipHttpRedirect=true` returns a `photoUri` the app can load without an
    API key.
    """
    if not photo_name:
        return None
    try:
        media_url = (
            f"https://places.googleapis.com/v1/{photo_name}/media"
            f"?maxWidthPx=800&skipHttpRedirect=true"
        )
        resp = requests.get(
            media_url,
            headers={"X-Goog-Api-Key": api_key},
            timeout=10,
        )
        if resp.status_code == 200:
            uri = (resp.json() or {}).get("photoUri")
            if isinstance(uri, str) and uri.startswith("http"):
                return uri

        # Fallback: follow the redirect Location header.
        resp2 = requests.get(
            f"https://places.googleapis.com/v1/{photo_name}/media?maxWidthPx=800",
            headers={"X-Goog-Api-Key": api_key},
            allow_redirects=False,
            timeout=10,
        )
        loc = resp2.headers.get("location")
        if resp2.status_code in (301, 302, 303, 307, 308) and loc:
            return loc
    except Exception as e:
        print(f"[restaurants] photo resolve failed: {e}")
    return None


def _search_places(
    api_key: str,
    latitude: float,
    longitude: float,
    radius_meters: float,
    included_types: list[str],
    max_results: int = 20,
) -> list[dict]:
    """Call the Google Places searchNearby API."""
    endpoint = "https://places.googleapis.com/v1/places:searchNearby"
    field_mask = ",".join([
        "places.id",
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
        "places.location",
    ])
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": field_mask,
    }
    payload = {
        "includedTypes": included_types,
        "maxResultCount": min(max_results, 20),
        "locationRestriction": {
            "circle": {
                "center": {"latitude": latitude, "longitude": longitude},
                "radius": min(max(radius_meters, 100.0), 50000.0),
            }
        },
    }

    try:
        resp = requests.post(endpoint, json=payload, headers=headers, timeout=15)
        if resp.status_code != 200:
            print(f"[restaurants] Google Places returned {resp.status_code}")
            return []

        data = resp.json()
        places = data.get("places", [])
        # Resolve photo resource names -> durable CDN URLs in parallel so a
        # cuisine search with ~20 hits doesn't serialize 20 media round-trips.
        photo_jobs: dict[int, str] = {}
        draft: list[dict] = []
        for idx, place in enumerate(places):
            name = place.get("displayName", {}).get("text", "Unknown")
            photos = place.get("photos", [])
            if photos:
                ref = photos[0].get("name", "")
                if ref:
                    photo_jobs[idx] = ref

            price_str = place.get("priceLevel")
            price_int = {
                "PRICE_LEVEL_FREE": 0,
                "PRICE_LEVEL_INEXPENSIVE": 1,
                "PRICE_LEVEL_MODERATE": 2,
                "PRICE_LEVEL_EXPENSIVE": 3,
                "PRICE_LEVEL_VERY_EXPENSIVE": 4,
            }.get(price_str)

            loc = place.get("location", {})
            place_id = (place.get("id") or "").strip()

            draft.append({
                "place_id": place_id,
                "name": name,
                "address": place.get("formattedAddress"),
                "rating": place.get("rating"),
                "user_rating_count": place.get("userRatingCount"),
                "photo_url": None,
                "type": place.get("primaryType"),
                "summary": place.get("editorialSummary", {}).get("text"),
                "price_level": price_int,
                "website_url": place.get("websiteUri"),
                "google_maps_url": place.get("googleMapsUri"),
                "latitude": loc.get("latitude"),
                "longitude": loc.get("longitude"),
            })

        if photo_jobs:
            with ThreadPoolExecutor(max_workers=min(8, len(photo_jobs))) as pool:
                futures = {
                    pool.submit(_resolve_place_photo_url, api_key, ref): idx
                    for idx, ref in photo_jobs.items()
                }
                for fut in as_completed(futures):
                    idx = futures[fut]
                    try:
                        draft[idx]["photo_url"] = fut.result()
                    except Exception:
                        draft[idx]["photo_url"] = None

        return draft
    except Exception as e:
        print(f"[restaurants] Places API error: {e}")
        return []


def _place_to_event(
    place: dict,
    tags: list[str],
    cuisine: Optional[str],
) -> ScrapedEvent:
    """Convert a Google Places result dict into a ScrapedEvent."""
    place_id = (place.get("place_id") or "").strip()
    if place_id:
        source_id = f"gp_{place_id}"
    else:
        # Fallback for older payloads / missing ids — strip path separators so
        # Firestore catalog writes never see a `/` in the document id.
        safe_name = place["name"].replace(" ", "_").replace("/", "_").lower()
        source_id = f"gp_{safe_name}"

    return ScrapedEvent(
        source="google_places",
        source_id=source_id,
        name=place["name"],
        venue_name=place["name"],
        venue_address=place.get("address"),
        latitude=place.get("latitude"),
        longitude=place.get("longitude"),
        image_url=place.get("photo_url"),
        description=place.get("summary"),
        url=place.get("website_url") or place.get("google_maps_url"),
        tags=tags,
        category="dining",
        genre=cuisine,
        rating=place.get("rating"),
        review_count=place.get("user_rating_count"),
    )


def _infer_cuisine_from_type(primary_type: str) -> Optional[str]:
    """Reverse-map a Google Places type to a cuisine label."""
    type_lower = primary_type.lower()
    for cuisine, types in CUISINE_TO_PLACE_TYPES.items():
        if any(t in type_lower for t in types):
            return cuisine.lower()
    if "restaurant" in type_lower:
        return None
    return None
