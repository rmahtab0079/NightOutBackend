"""Partiful discover page scraper for public events."""

from __future__ import annotations

import json
import re
import requests
from datetime import datetime
from typing import Optional

from .models import ScrapedEvent, CLASSIFICATION_TO_CATEGORY

_DISCOVER_URLS = [
    "https://partiful.com/discover",
    "https://partiful.com/discover/partilist",
]

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

_TAG_CATEGORY_MAP: dict[str, str] = {
    "pop-ups": "food",
    "food": "food",
    "drinks": "food",
    "brunch": "food",
    "dinner": "food",
    "music": "music",
    "live music": "music",
    "dj": "music",
    "party": "music",
    "art": "arts",
    "gallery": "arts",
    "comedy": "arts",
    "film": "arts",
    "sports": "sports",
    "fitness": "sports",
    "outdoor": "outdoors",
    "hiking": "outdoors",
    "meet new people": "food",
    "networking": "food",
}


def scrape_partiful(
    city: Optional[str] = None,
) -> list[ScrapedEvent]:
    """
    Scrape public events from Partiful's discover pages.

    Partiful primarily covers NYC and LA. We scrape their rendered HTML
    and extract event data from embedded JSON or structured markup.
    """
    results: list[ScrapedEvent] = []
    seen_ids: set[str] = set()

    for url in _DISCOVER_URLS:
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=15)
            if resp.status_code != 200:
                print(f"[partiful] {url} returned {resp.status_code}")
                continue

            events = _extract_from_html(resp.text)
            for ev in events:
                if ev.source_id not in seen_ids:
                    seen_ids.add(ev.source_id)
                    results.append(ev)

        except Exception as e:
            print(f"[partiful] Error scraping {url}: {e}")

    if city:
        city_url = f"https://partiful.com/discover?city={city.lower().replace(' ', '-')}"
        try:
            resp = requests.get(city_url, headers=_HEADERS, timeout=15)
            if resp.status_code == 200:
                for ev in _extract_from_html(resp.text):
                    if ev.source_id not in seen_ids:
                        seen_ids.add(ev.source_id)
                        results.append(ev)
        except Exception as e:
            print(f"[partiful] Error scraping city page: {e}")

    print(f"[partiful] Scraped {len(results)} events")
    return results


def _extract_from_html(html: str) -> list[ScrapedEvent]:
    """Extract event data from Partiful HTML — tries JSON-LD, then __NEXT_DATA__."""
    results: list[ScrapedEvent] = []

    ld_blocks = re.findall(
        r'<script type="application/ld\+json">(.*?)</script>',
        html,
        re.DOTALL,
    )
    for block in ld_blocks:
        try:
            data = json.loads(block)
            items = data if isinstance(data, list) else [data]
            for item in items:
                ev = _parse_ld_event(item)
                if ev:
                    results.append(ev)
        except json.JSONDecodeError:
            continue

    next_data = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
        html,
        re.DOTALL,
    )
    if next_data:
        try:
            nd = json.loads(next_data.group(1))
            page_props = nd.get("props", {}).get("pageProps", {})
            events = page_props.get("events", [])
            if not events:
                events = page_props.get("discoverEvents", [])
            for ev_data in events:
                ev = _parse_next_data_event(ev_data)
                if ev:
                    results.append(ev)
        except (json.JSONDecodeError, KeyError):
            pass

    return results


def _parse_ld_event(item: dict) -> Optional[ScrapedEvent]:
    """Parse a JSON-LD Event object."""
    if item.get("@type") != "Event":
        return None

    location = item.get("location", {})
    geo = location.get("geo", {})
    address = location.get("address", {})
    start_date = item.get("startDate", "")
    date_str = time_str = None
    if "T" in start_date:
        date_str, time_str = start_date.split("T", 1)
    elif start_date:
        date_str = start_date

    event_url = item.get("url", "")
    event_id = event_url.split("/")[-1] if event_url else item.get("name", "")

    tags = _infer_tags(item.get("name", ""), item.get("description", ""))
    category = _category_from_tags(tags)

    return ScrapedEvent(
        source="partiful",
        source_id=f"partiful_{event_id}",
        name=item.get("name", "Unknown"),
        date=date_str,
        time=time_str,
        venue_name=location.get("name"),
        venue_address=address.get("streetAddress") or address.get("addressLocality"),
        latitude=float(geo["latitude"]) if "latitude" in geo else None,
        longitude=float(geo["longitude"]) if "longitude" in geo else None,
        image_url=item.get("image"),
        description=item.get("description"),
        url=event_url,
        tags=tags,
        category=category,
    )


def _parse_next_data_event(ev: dict) -> Optional[ScrapedEvent]:
    """Parse an event from Partiful's __NEXT_DATA__ payload."""
    name = ev.get("title") or ev.get("name")
    if not name:
        return None

    event_id = ev.get("id") or ev.get("slug") or name
    location = ev.get("location", {}) or {}
    start = ev.get("startDate") or ev.get("start") or ""
    date_str = time_str = None
    if "T" in start:
        date_str, time_str = start.split("T", 1)
    elif start:
        date_str = start

    tags = _infer_tags(name, ev.get("description", ""))
    if ev.get("tags"):
        tags.extend([t.lower() for t in ev["tags"] if isinstance(t, str)])
    category = _category_from_tags(tags)

    return ScrapedEvent(
        source="partiful",
        source_id=f"partiful_{event_id}",
        name=name,
        date=date_str,
        time=time_str,
        venue_name=location.get("name"),
        venue_address=location.get("address"),
        latitude=location.get("lat"),
        longitude=location.get("lng"),
        image_url=ev.get("imageUrl") or ev.get("coverImage"),
        description=ev.get("description"),
        url=f"https://partiful.com/e/{event_id}" if event_id else None,
        tags=tags,
        category=category,
    )


def _infer_tags(name: str, description: str) -> list[str]:
    """Infer tags from event name and description text."""
    text = f"{name} {description}".lower()
    tags: list[str] = []
    keywords = [
        "music", "dj", "live", "concert", "party", "brunch", "dinner",
        "comedy", "art", "gallery", "film", "movie", "sports", "fitness",
        "yoga", "run", "hike", "outdoor", "food", "drinks", "wine",
        "pop-up", "networking",
    ]
    for kw in keywords:
        if kw in text:
            tags.append(kw)
    return tags


def _category_from_tags(tags: list[str]) -> Optional[str]:
    for tag in tags:
        if tag in _TAG_CATEGORY_MAP:
            return _TAG_CATEGORY_MAP[tag]
    return None
