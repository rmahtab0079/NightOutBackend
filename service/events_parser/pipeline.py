"""
EventsParser pipeline orchestrator.

Coordinates scraping from all sources, matches events to users,
and writes curated results to Firebase.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from .models import ScrapedEvent
from .ticketmaster_scraper import scrape_ticketmaster
from .eventbrite_scraper import scrape_eventbrite
from .amc_scraper import scrape_amc
from .partiful_scraper import scrape_partiful
from .restaurant_scraper import scrape_restaurants, tag_dietary_matches
from .matcher import match_events_to_user, group_events_by_category
from .firebase_writer import (
    get_all_users_with_preferences,
    get_user_swipe_categories,
    write_curated_events,
)

# Default location (New York City) when a user has no stored location
_DEFAULT_LAT = 40.7128
_DEFAULT_LON = -74.0060


def scrape_all_events(
    latitude: float,
    longitude: float,
    radius_miles: float = 25.0,
    days_ahead: int = 14,
) -> list[ScrapedEvent]:
    """Run all non-personalized scrapers for a given geographic center."""
    all_events: list[ScrapedEvent] = []

    print(f"\n--- Scraping events near ({latitude}, {longitude}), radius={radius_miles}mi ---")

    all_events.extend(scrape_ticketmaster(
        latitude, longitude,
        radius_miles=radius_miles,
        days_ahead=days_ahead,
        size=100,
    ))

    all_events.extend(scrape_eventbrite(
        latitude, longitude,
        radius_miles=radius_miles,
        days_ahead=days_ahead,
    ))

    all_events.extend(scrape_amc(
        latitude, longitude,
        radius_miles=radius_miles,
    ))

    all_events.extend(scrape_partiful())

    return all_events


def _scrape_restaurants_for_user(
    latitude: float,
    longitude: float,
    radius_miles: float,
    cuisines: list[str],
    dietary: list[str],
) -> list[ScrapedEvent]:
    """Scrape restaurants personalized to a specific user's cuisine rankings."""
    restaurants = scrape_restaurants(
        latitude, longitude,
        radius_miles=radius_miles,
        cuisines=cuisines if cuisines else None,
        max_per_cuisine=8,
    )
    restaurants = tag_dietary_matches(restaurants, dietary)
    return restaurants


def _dedup(events: list[ScrapedEvent]) -> list[ScrapedEvent]:
    seen: set[str] = set()
    deduped: list[ScrapedEvent] = []
    for ev in events:
        key = f"{ev.source}:{ev.source_id}"
        if key not in seen:
            seen.add(key)
            deduped.append(ev)
    return deduped


def run_pipeline(
    radius_miles: float = 25.0,
    days_ahead: int = 14,
    max_events_per_user: int = 50,
) -> dict:
    """
    Full pipeline: scrape events + restaurants, match to all users, write to Firebase.

    Steps:
      1. Load all users and their preferences from Firebase
      2. Group users by approximate location (to avoid redundant event scraping)
      3. For each location cluster, scrape events (shared across users in cluster)
      4. For each user, scrape restaurants personalized to their cuisine rankings
      5. Tag restaurants with dietary compatibility
      6. Score and rank all items by relevance
      7. Write curated lists to Firebase, grouped by category (including "dining")
    """
    run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    print(f"\n{'='*60}")
    print(f"EventsParser pipeline — run_id: {run_id}")
    print(f"{'='*60}")

    users = get_all_users_with_preferences()
    print(f"\nLoaded {len(users)} users from Firebase")

    if not users:
        print("No users found, exiting")
        return {"run_id": run_id, "users_processed": 0, "events_scraped": 0}

    location_clusters: dict[str, dict] = {}
    for user in users:
        lat = user.get("last_latitude")
        lon = user.get("last_longitude")
        if lat is None or lon is None:
            lat, lon = _DEFAULT_LAT, _DEFAULT_LON

        cluster_key = f"{round(lat, 1)}_{round(lon, 1)}"
        if cluster_key not in location_clusters:
            location_clusters[cluster_key] = {
                "latitude": lat,
                "longitude": lon,
                "users": [],
                "events": None,
            }
        location_clusters[cluster_key]["users"].append(user)

    print(f"Grouped into {len(location_clusters)} location cluster(s)")

    total_events = 0
    total_restaurants = 0
    total_matched = 0

    for cluster_key, cluster in location_clusters.items():
        shared_events = scrape_all_events(
            latitude=cluster["latitude"],
            longitude=cluster["longitude"],
            radius_miles=radius_miles,
            days_ahead=days_ahead,
        )
        cluster["events"] = shared_events
        total_events += len(shared_events)

        for user in cluster["users"]:
            email = user.get("email", "")
            if not email:
                continue

            interests = user.get("interests", [])
            cuisines = user.get("cuisines", [])
            dietary = user.get("dietaryPreferences", [])
            user_lat = user.get("last_latitude")
            user_lon = user.get("last_longitude")

            user_restaurants = _scrape_restaurants_for_user(
                latitude=cluster["latitude"],
                longitude=cluster["longitude"],
                radius_miles=radius_miles,
                cuisines=cuisines,
                dietary=dietary,
            )
            total_restaurants += len(user_restaurants)

            all_items = _dedup(shared_events + user_restaurants)

            liked_categories = get_user_swipe_categories(email)

            matched = match_events_to_user(
                events=all_items,
                user_interests=interests,
                user_cuisines=cuisines,
                user_dietary=dietary,
                user_lat=user_lat,
                user_lon=user_lon,
                max_radius_miles=radius_miles,
                liked_categories=liked_categories,
                max_events=max_events_per_user,
            )

            grouped = group_events_by_category(matched)
            write_curated_events(email, grouped, run_id)
            total_matched += len(matched)

            dining_count = len(grouped.get("dining", []))
            print(f"  {email}: {len(matched)} matched ({dining_count} restaurants) "
                  f"across {len(grouped)} categories")

    summary = {
        "run_id": run_id,
        "users_processed": len(users),
        "location_clusters": len(location_clusters),
        "events_scraped": total_events,
        "restaurants_scraped": total_restaurants,
        "events_matched": total_matched,
        "completed_at": datetime.utcnow().isoformat() + "Z",
    }

    print(f"\n{'='*60}")
    print(f"Pipeline complete: {summary}")
    print(f"{'='*60}\n")

    return summary
