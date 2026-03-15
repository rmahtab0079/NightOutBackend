"""Eventbrite event scraper via city-browse __SERVER_DATA__."""

from __future__ import annotations

import json
import re
import requests
from typing import Optional

from .models import ScrapedEvent

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

_EB_CATEGORY_MAP: dict[str, str] = {
    "103": "music",
    "101": "arts",
    "110": "food",
    "105": "arts",
    "108": "sports",
    "113": "other",
    "104": "arts",
    "109": "outdoors",
    "107": "other",
    "102": "arts",
}

_EB_DISPLAY_CATEGORY_MAP: dict[str, str] = {
    "music": "music",
    "food & drink": "food",
    "performing & visual arts": "arts",
    "sports & fitness": "sports",
    "film, media & entertainment": "arts",
    "travel & outdoor": "outdoors",
    "community & culture": "other",
    "charity & causes": "other",
    "business": "other",
    "science & technology": "arts",
    "health & wellness": "sports",
}


def scrape_eventbrite(
    latitude: float,
    longitude: float,
    radius_miles: float = 50.0,
    days_ahead: int = 14,
    page_size: int = 50,
    keyword: Optional[str] = None,
) -> list[ScrapedEvent]:
    """
    Scrape public events from Eventbrite's city-browse page.

    Fetches the browse HTML page and extracts event data from the
    embedded ``window.__SERVER_DATA__`` JSON blob.  The old XHR API
    endpoint no longer accepts unauthenticated requests.
    """
    km_radius = radius_miles * 1.60934
    browse_url = (
        f"https://www.eventbrite.com/d/united-states/events/"
        f"?location_within={km_radius:.0f}km"
        f"&location_latitude={latitude}"
        f"&location_longitude={longitude}"
    )

    results: list[ScrapedEvent] = []
    seen_ids: set[str] = set()

    try:
        resp = requests.get(browse_url, headers=_HEADERS, timeout=20)
        if resp.status_code != 200:
            print(f"[eventbrite] Browse page returned {resp.status_code}")
            return results

        events = _extract_server_data_events(resp.text)
        for ev in events:
            if ev.source_id not in seen_ids:
                seen_ids.add(ev.source_id)
                results.append(ev)

        if not results:
            results = _extract_json_ld_events(resp.text)

    except Exception as e:
        print(f"[eventbrite] Error: {e}")

    print(f"[eventbrite] Scraped {len(results)} events")
    return results


def _extract_server_data_events(html: str) -> list[ScrapedEvent]:
    """Parse events from window.__SERVER_DATA__ embedded in the HTML."""
    m = re.search(
        r"window\.__SERVER_DATA__\s*=\s*({.*?});\s*</script>",
        html,
        re.DOTALL,
    )
    if not m:
        m = re.search(r"window\.__SERVER_DATA__\s*=\s*({.*?});", html, re.DOTALL)
    if not m:
        return []

    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError:
        return []

    results: list[ScrapedEvent] = []
    buckets = data.get("buckets", [])
    for bucket in buckets:
        if not isinstance(bucket, dict):
            continue
        for ev in bucket.get("events", []):
            parsed = _parse_server_data_event(ev)
            if parsed:
                results.append(parsed)

    return results


def _parse_server_data_event(ev: dict) -> Optional[ScrapedEvent]:
    """Convert a single __SERVER_DATA__ event dict into a ScrapedEvent."""
    name = ev.get("name")
    if not name:
        return None

    if ev.get("is_online_event"):
        return None

    event_id = str(ev.get("id") or ev.get("eventbrite_event_id") or "")

    date_str = ev.get("start_date")
    time_str = ev.get("start_time")

    venue = ev.get("primary_venue") or {}
    addr = venue.get("address") or {}
    venue_lat = _safe_float(addr.get("latitude"))
    venue_lng = _safe_float(addr.get("longitude"))

    image_url = (ev.get("image") or {}).get("url")

    tags: list[str] = []
    category: Optional[str] = None
    for t in ev.get("tags", []):
        if not isinstance(t, dict):
            continue
        prefix = t.get("prefix", "")
        display = (t.get("display_name") or "").lower()
        tag_value = t.get("tag", "")

        if prefix == "EventbriteCategory":
            cat_id = tag_value.rsplit("/", 1)[-1] if "/" in tag_value else ""
            if cat_id in _EB_CATEGORY_MAP:
                category = category or _EB_CATEGORY_MAP[cat_id]
            if display in _EB_DISPLAY_CATEGORY_MAP:
                category = category or _EB_DISPLAY_CATEGORY_MAP[display]
        if display:
            tags.append(display)

    urgency = ev.get("urgency_signals") or {}
    urgency_msgs = urgency.get("messages") or []
    selling_fast = "fewTickets" in urgency_msgs or "salesEndSoon" in urgency_msgs

    return ScrapedEvent(
        source="eventbrite",
        source_id=event_id,
        name=name,
        date=date_str,
        time=time_str,
        venue_name=venue.get("name"),
        venue_address=addr.get("localized_address_display"),
        latitude=venue_lat,
        longitude=venue_lng,
        image_url=image_url,
        description=ev.get("summary"),
        url=ev.get("url"),
        tags=tags,
        category=category,
        is_selling_fast=selling_fast,
    )


def _extract_json_ld_events(html: str) -> list[ScrapedEvent]:
    """Fallback: extract events from JSON-LD script blocks."""
    results: list[ScrapedEvent] = []
    ld_blocks = re.findall(
        r'<script type="application/ld\+json">(.*?)</script>',
        html,
        re.DOTALL,
    )
    for block in ld_blocks:
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue

        items = data if isinstance(data, list) else [data]
        if isinstance(data, dict) and data.get("@type") == "ItemList":
            items = data.get("itemListElement", [])

        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("@type") == "ListItem":
                item = item.get("item", item)
            if item.get("@type") != "Event":
                continue

            location = item.get("location") or {}
            address = location.get("address") or {}
            geo = location.get("geo") or {}

            start_date = item.get("startDate", "")
            date_str = time_str = None
            if "T" in start_date:
                date_str, time_str = start_date.split("T", 1)
            else:
                date_str = start_date or None

            results.append(ScrapedEvent(
                source="eventbrite",
                source_id=item.get("url", ""),
                name=item.get("name", "Unknown"),
                date=date_str,
                time=time_str,
                venue_name=location.get("name"),
                venue_address=address.get("streetAddress"),
                latitude=_safe_float(geo.get("latitude")),
                longitude=_safe_float(geo.get("longitude")),
                image_url=item.get("image"),
                description=item.get("description"),
                url=item.get("url"),
                tags=[],
                category=None,
            ))

    return results


def _safe_float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None
