"""Ticketmaster Discovery API scraper for the EventsParser pipeline."""

from __future__ import annotations

import os
import requests
from datetime import datetime, timedelta
from typing import Optional

from .models import ScrapedEvent, CLASSIFICATION_TO_CATEGORY


def _encode_geohash(lat: float, lon: float, precision: int = 9) -> str:
    """Encode lat/lon into a geohash string for the Ticketmaster API."""
    base32 = "0123456789bcdefghjkmnpqrstuvwxyz"
    lat_range, lon_range = [-90.0, 90.0], [-180.0, 180.0]
    bits = [16, 8, 4, 2, 1]
    bit, ch, is_lon = 0, 0, True
    result = []
    while len(result) < precision:
        if is_lon:
            mid = (lon_range[0] + lon_range[1]) / 2
            if lon > mid:
                ch |= bits[bit]
                lon_range[0] = mid
            else:
                lon_range[1] = mid
        else:
            mid = (lat_range[0] + lat_range[1]) / 2
            if lat > mid:
                ch |= bits[bit]
                lat_range[0] = mid
            else:
                lat_range[1] = mid
        is_lon = not is_lon
        if bit < 4:
            bit += 1
        else:
            result.append(base32[ch])
            bit, ch = 0, 0
    return "".join(result)


def scrape_ticketmaster(
    latitude: float,
    longitude: float,
    radius_miles: float = 25.0,
    days_ahead: int = 14,
    size: int = 50,
    classification: Optional[str] = None,
    keyword: Optional[str] = None,
) -> list[ScrapedEvent]:
    """Fetch upcoming events from the Ticketmaster Discovery API."""
    api_key = os.getenv("TICKETMASTER_API_KEY", "")
    if not api_key:
        print("[ticketmaster] No API key configured, skipping")
        return []

    geohash = _encode_geohash(latitude, longitude)
    now = datetime.utcnow()
    start = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    end = (now + timedelta(days=days_ahead)).strftime("%Y-%m-%dT%H:%M:%SZ")

    params: dict[str, str] = {
        "apikey": api_key,
        "geoPoint": geohash,
        "radius": str(int(max(radius_miles, 1))),
        "unit": "miles",
        "size": str(min(size, 200)),
        "sort": "date,asc",
        "startDateTime": start,
        "endDateTime": end,
    }
    if classification:
        params["classificationName"] = classification
    if keyword and keyword.strip():
        params["keyword"] = keyword.strip()

    try:
        resp = requests.get(
            "https://app.ticketmaster.com/discovery/v2/events.json",
            params=params,
            timeout=15,
        )
        if resp.status_code != 200:
            print(f"[ticketmaster] API returned {resp.status_code}")
            return []

        data = resp.json()
        raw_events = data.get("_embedded", {}).get("events", [])
        results: list[ScrapedEvent] = []

        for ev in raw_events:
            images = ev.get("images", [])
            image_url = None
            for img in sorted(images, key=lambda x: x.get("width", 0), reverse=True):
                if img.get("url"):
                    image_url = img["url"]
                    break

            venues = ev.get("_embedded", {}).get("venues", [])
            venue_name = venue_address = None
            venue_lat = venue_lon = None
            if venues:
                v = venues[0]
                venue_name = v.get("name")
                parts = [
                    v.get("address", {}).get("line1", ""),
                    v.get("city", {}).get("name", ""),
                    v.get("state", {}).get("stateCode", ""),
                ]
                venue_address = ", ".join(p for p in parts if p)
                loc = v.get("location", {})
                venue_lat = float(loc["latitude"]) if "latitude" in loc else None
                venue_lon = float(loc["longitude"]) if "longitude" in loc else None

            dates = ev.get("dates", {}).get("start", {})
            date_str = dates.get("localDate")
            time_str = dates.get("localTime")

            price_ranges = ev.get("priceRanges", [])
            min_price = max_price = None
            currency = "USD"
            if price_ranges:
                min_price = price_ranges[0].get("min")
                max_price = price_ranges[0].get("max")
                currency = price_ranges[0].get("currency", "USD")

            classifications = ev.get("classifications", [])
            segment = genre = None
            tags: list[str] = []
            if classifications:
                c = classifications[0]
                segment = c.get("segment", {}).get("name")
                genre = c.get("genre", {}).get("name")
                if segment:
                    tags.append(segment.lower())
                if genre:
                    tags.append(genre.lower())

            category = None
            for tag in tags:
                if tag in CLASSIFICATION_TO_CATEGORY:
                    category = CLASSIFICATION_TO_CATEGORY[tag]
                    break

            description = (
                ev.get("info") or ev.get("description") or ev.get("pleaseNote")
            )

            results.append(ScrapedEvent(
                source="ticketmaster",
                source_id=ev.get("id", ""),
                name=ev.get("name", "Unknown"),
                date=date_str,
                time=time_str,
                venue_name=venue_name,
                venue_address=venue_address,
                latitude=venue_lat,
                longitude=venue_lon,
                image_url=image_url,
                description=description,
                url=ev.get("url"),
                min_price=min_price,
                max_price=max_price,
                currency=currency,
                tags=tags,
                category=category,
                genre=genre,
            ))

        print(f"[ticketmaster] Scraped {len(results)} events")
        return results

    except Exception as e:
        print(f"[ticketmaster] Error: {e}")
        return []
