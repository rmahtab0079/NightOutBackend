"""
Restaurant scraper using Google Places API.

Fetches restaurants near a user's location, filtered by cuisine type.
Respects dietary preferences when tagging and scoring.
"""

from __future__ import annotations

import os
import requests
from typing import Optional

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
    max_per_cuisine: int = 10,
) -> list[ScrapedEvent]:
    """
    Fetch restaurants from Google Places API, grouped by cuisine type.

    If cuisines is provided, searches specifically for those cuisine types
    in ranked order. Otherwise does a general restaurant search.
    """
    api_key = os.getenv("GOOGLE_PLACES_API_KEY", "")
    if not api_key:
        print("[restaurants] No Google Places API key, skipping")
        return []

    radius_meters = radius_miles * 1609.34
    results: list[ScrapedEvent] = []
    seen_names: set[str] = set()

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

    print(f"[restaurants] Scraped {len(results)} restaurants")
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
        matched_diets: list[str] = []
        for diet in dietary_preferences:
            keywords = DIETARY_KEYWORDS.get(diet, [diet.lower()])
            if any(kw in text for kw in keywords):
                matched_diets.append(diet)
        if matched_diets:
            r.tags.extend([d.lower() for d in matched_diets])

    return restaurants


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
        results = []
        for place in places:
            name = place.get("displayName", {}).get("text", "Unknown")
            photo_url = None
            photos = place.get("photos", [])
            if photos:
                ref = photos[0].get("name", "")
                if ref:
                    photo_url = (
                        f"https://places.googleapis.com/v1/{ref}/media"
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

            loc = place.get("location", {})

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
                "latitude": loc.get("latitude"),
                "longitude": loc.get("longitude"),
            })
        return results
    except Exception as e:
        print(f"[restaurants] Places API error: {e}")
        return []


def _place_to_event(
    place: dict,
    tags: list[str],
    cuisine: Optional[str],
) -> ScrapedEvent:
    """Convert a Google Places result dict into a ScrapedEvent."""
    return ScrapedEvent(
        source="google_places",
        source_id=f"gp_{place['name'].replace(' ', '_').lower()}",
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
