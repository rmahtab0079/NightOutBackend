"""Eventbrite event scraper using their public search endpoint."""

from __future__ import annotations

import requests
from datetime import datetime, timedelta
from typing import Optional

from .models import ScrapedEvent, CLASSIFICATION_TO_CATEGORY

# Eventbrite category ID -> our canonical tag
_EB_CATEGORY_MAP: dict[str, tuple[str, str]] = {
    "103": ("music", "music"),           # Music
    "101": ("arts", "business"),         # Business
    "110": ("food", "food & drink"),     # Food & Drink
    "105": ("arts", "performing arts"),  # Performing & Visual Arts
    "108": ("sports", "sports"),         # Sports & Fitness
    "113": ("arts", "community"),        # Community
    "104": ("arts", "film & media"),     # Film, Media & Entertainment
    "109": ("outdoors", "travel"),       # Travel & Outdoor
    "107": ("arts", "charity"),          # Charity & Causes
    "102": ("arts", "science"),          # Science & Technology
}


def scrape_eventbrite(
    latitude: float,
    longitude: float,
    radius_miles: float = 25.0,
    days_ahead: int = 14,
    page_size: int = 50,
    keyword: Optional[str] = None,
) -> list[ScrapedEvent]:
    """
    Scrape public events from Eventbrite's search page API.

    Eventbrite's search page makes XHR requests to an internal endpoint.
    We replicate those requests to get public event data.
    """
    results: list[ScrapedEvent] = []

    now = datetime.utcnow()
    start = now.strftime("%Y-%m-%dT%H:%M:%S")
    end = (now + timedelta(days=days_ahead)).strftime("%Y-%m-%dT%H:%M:%S")

    search_url = "https://www.eventbrite.com/api/v3/destination/search/"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
        "Referer": "https://www.eventbrite.com/d/united-states/events/",
    }

    km_radius = radius_miles * 1.60934
    params = {
        "dates": "current_future",
        "date_range": f"{start}--{end}",
        "page_size": str(min(page_size, 50)),
        "latitude": str(latitude),
        "longitude": str(longitude),
        "within": f"{km_radius}km",
        "page": "1",
    }
    if keyword and keyword.strip():
        params["q"] = keyword.strip()

    try:
        resp = requests.get(search_url, params=params, headers=headers, timeout=15)

        if resp.status_code != 200:
            print(f"[eventbrite] Status {resp.status_code}, trying fallback")
            return _scrape_eventbrite_fallback(latitude, longitude, radius_miles, days_ahead)

        data = resp.json()
        events = data.get("events", {}).get("results", [])

        for ev in events:
            tags: list[str] = []
            category: Optional[str] = None

            cat_id = ev.get("primary_category", {}).get("id")
            if cat_id and str(cat_id) in _EB_CATEGORY_MAP:
                cat, tag = _EB_CATEGORY_MAP[str(cat_id)]
                category = cat
                tags.append(tag)

            for t in ev.get("tags", []):
                tag_text = t.get("display_name", "").lower()
                if tag_text:
                    tags.append(tag_text)

            venue = ev.get("primary_venue", {})
            venue_lat = venue.get("address", {}).get("latitude")
            venue_lon = venue.get("address", {}).get("longitude")

            start_info = ev.get("start_date", "")
            date_str = time_str = None
            if start_info:
                parts = start_info.split("T")
                date_str = parts[0] if parts else None
                time_str = parts[1] if len(parts) > 1 else None

            image_url = ev.get("image", {}).get("url")

            ticket_info = ev.get("ticket_availability", {})
            min_price = ticket_info.get("minimum_ticket_price", {}).get("major_value")
            max_price = ticket_info.get("maximum_ticket_price", {}).get("major_value")
            if min_price is not None:
                try:
                    min_price = float(min_price)
                except (ValueError, TypeError):
                    min_price = None
            if max_price is not None:
                try:
                    max_price = float(max_price)
                except (ValueError, TypeError):
                    max_price = None

            results.append(ScrapedEvent(
                source="eventbrite",
                source_id=str(ev.get("id", "")),
                name=ev.get("name", "Unknown"),
                date=date_str,
                time=time_str,
                venue_name=venue.get("name"),
                venue_address=venue.get("address", {}).get("localized_address_display"),
                latitude=float(venue_lat) if venue_lat else None,
                longitude=float(venue_lon) if venue_lon else None,
                image_url=image_url,
                description=ev.get("summary"),
                url=ev.get("url"),
                min_price=min_price,
                max_price=max_price,
                tags=tags,
                category=category,
            ))

        print(f"[eventbrite] Scraped {len(results)} events")

    except Exception as e:
        print(f"[eventbrite] Error: {e}")
        results = _scrape_eventbrite_fallback(latitude, longitude, radius_miles, days_ahead)

    return results


def _scrape_eventbrite_fallback(
    latitude: float,
    longitude: float,
    radius_miles: float = 25.0,
    days_ahead: int = 14,
) -> list[ScrapedEvent]:
    """Fallback: scrape Eventbrite's public HTML listing and extract JSON-LD data."""
    results: list[ScrapedEvent] = []
    try:
        city_url = f"https://www.eventbrite.com/d/nearby--{latitude},{longitude}/events/"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
        }
        resp = requests.get(city_url, headers=headers, timeout=15)
        if resp.status_code != 200:
            print(f"[eventbrite-fallback] Status {resp.status_code}")
            return results

        import json
        import re
        ld_blocks = re.findall(
            r'<script type="application/ld\+json">(.*?)</script>',
            resp.text,
            re.DOTALL,
        )
        for block in ld_blocks:
            try:
                data = json.loads(block)
                items = data if isinstance(data, list) else [data]
                for item in items:
                    if item.get("@type") != "Event":
                        continue
                    location = item.get("location", {})
                    address = location.get("address", {})
                    geo = location.get("geo", {})
                    start_date = item.get("startDate", "")
                    date_str = time_str = None
                    if "T" in start_date:
                        date_str, time_str = start_date.split("T", 1)
                    else:
                        date_str = start_date

                    results.append(ScrapedEvent(
                        source="eventbrite",
                        source_id=item.get("url", ""),
                        name=item.get("name", "Unknown"),
                        date=date_str,
                        time=time_str,
                        venue_name=location.get("name"),
                        venue_address=address.get("streetAddress"),
                        latitude=float(geo["latitude"]) if "latitude" in geo else None,
                        longitude=float(geo["longitude"]) if "longitude" in geo else None,
                        image_url=item.get("image"),
                        description=item.get("description"),
                        url=item.get("url"),
                        tags=[],
                        category=None,
                    ))
            except (json.JSONDecodeError, KeyError):
                continue

        print(f"[eventbrite-fallback] Scraped {len(results)} events")
    except Exception as e:
        print(f"[eventbrite-fallback] Error: {e}")

    return results
