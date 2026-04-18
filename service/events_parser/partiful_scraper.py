"""Partiful discover page scraper for public events."""

from __future__ import annotations

import json
import re
import requests
from datetime import datetime, timezone
from typing import Optional

from .models import ScrapedEvent

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
    "meet new people": "other",
    "networking": "other",
}

_STRUCTURED_TAG_MAP: dict[str, str] = {
    "MUSIC": "music",
    "ARTS": "arts",
    "FILM": "arts",
    "FOOD": "food",
    "FOOD_AND_DRINK": "food",
    "NIGHTLIFE": "music",
    "SPORTS": "sports",
    "FITNESS": "sports",
    "OUTDOOR": "outdoors",
    "COMMUNITY": "other",
}

_GENERIC_STRUCTURED_TAGS = {"DISCOVER_HOME", "COMMUNITY"}
_LOCATION_TAG_PREFIXES = ("NYC_", "LA_", "SF_")


def scrape_partiful(
    city: Optional[str] = None,
) -> list[ScrapedEvent]:
    """
    Scrape public events from Partiful's discover pages.

    Partiful serves events for NYC, LA, and SF via __NEXT_DATA__
    embedded in HTML. The /discover page uses a `trendingSections` dict
    keyed by city; /discover/partilist uses a `sections` list.
    """
    results: list[ScrapedEvent] = []
    seen_ids: set[str] = set()

    for url in _DISCOVER_URLS:
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=30)
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
            resp = requests.get(city_url, headers=_HEADERS, timeout=30)
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
    """Extract event data from Partiful HTML via __NEXT_DATA__ or JSON-LD."""
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
    if not next_data:
        return results

    try:
        nd = json.loads(next_data.group(1))
    except json.JSONDecodeError:
        return results

    page_props = nd.get("props", {}).get("pageProps", {})

    # --- New format: trendingSections (dict keyed by city) ---
    trending = page_props.get("trendingSections")
    if isinstance(trending, dict):
        for _city, section in trending.items():
            if isinstance(section, dict):
                for item in section.get("items", []):
                    ev = _parse_section_item(item)
                    if ev:
                        results.append(ev)

    # --- New format: sections (list on /discover/partilist) ---
    sections = page_props.get("sections")
    if isinstance(sections, list):
        for section in sections:
            if isinstance(section, dict):
                for item in section.get("items", []):
                    ev = _parse_section_item(item)
                    if ev:
                        results.append(ev)

    # --- Legacy format: flat event lists (kept for backward compat) ---
    legacy_events = page_props.get("events") or page_props.get("discoverEvents") or []
    for ev_data in legacy_events:
        ev = _parse_legacy_event(ev_data)
        if ev:
            results.append(ev)

    return results


def _parse_section_item(wrapper: dict) -> Optional[ScrapedEvent]:
    """Parse a section item from Partiful's current __NEXT_DATA__ format.

    Each wrapper has shape: {id, type, descriptionTags, event: {...}, tags: [...]}
    """
    ev = wrapper.get("event")
    if not isinstance(ev, dict):
        return None

    name = ev.get("title") or ev.get("name")
    if not name:
        return None

    start_raw = ev.get("startDate", "")
    end_raw = ev.get("endDate", "")
    if not _is_future_or_present(start_raw, end_raw):
        return None

    event_id = ev.get("id", "")

    date_str, time_str = _split_iso_datetime(start_raw)

    loc_info = ev.get("locationInfo") or {}
    maps_info = loc_info.get("mapsInfo") or {}
    venue_name = maps_info.get("name") or loc_info.get("displayName")
    address_lines = loc_info.get("displayAddressLines") or maps_info.get("addressLines") or []
    venue_address = ", ".join(address_lines) if address_lines else None

    lat, lng = _extract_lat_lng_from_maps(maps_info)

    image_url = _extract_image_url(ev.get("image"))

    text_tags = _infer_tags(name, ev.get("description", ""))

    structured_tags = wrapper.get("tags") or []
    struct_category = None
    fallback_struct_category = None
    for st in structured_tags:
        if not isinstance(st, dict):
            continue
        tag_id = st.get("id", "")
        if tag_id.startswith(_LOCATION_TAG_PREFIXES) or tag_id == "DISCOVER_HOME":
            continue
        if tag_id in _STRUCTURED_TAG_MAP:
            mapped = _STRUCTURED_TAG_MAP[tag_id]
            if tag_id in _GENERIC_STRUCTURED_TAGS:
                fallback_struct_category = fallback_struct_category or mapped
            else:
                struct_category = struct_category or mapped
        label = (st.get("label") or "").lower()
        if label and label != "all":
            text_tags.append(label)

    category = struct_category or _category_from_tags(text_tags) or fallback_struct_category

    going = ev.get("goingGuestCount") or 0
    interested = ev.get("interestedGuestCount") or 0
    maybe = ev.get("maybeGuestCount") or 0

    return ScrapedEvent(
        source="partiful",
        source_id=f"partiful_{event_id}",
        name=name,
        date=date_str,
        time=time_str,
        venue_name=venue_name,
        venue_address=venue_address,
        latitude=lat,
        longitude=lng,
        image_url=image_url,
        description=ev.get("description"),
        url=f"https://partiful.com/e/{event_id}" if event_id else None,
        tags=text_tags,
        category=category,
        rsvp_count=going if going else None,
        interested_count=(interested + maybe) if (interested or maybe) else None,
    )


def _parse_ld_event(item: dict) -> Optional[ScrapedEvent]:
    """Parse a JSON-LD Event object."""
    if item.get("@type") != "Event":
        return None

    start_raw = item.get("startDate", "")
    end_raw = item.get("endDate", "")
    if not _is_future_or_present(start_raw, end_raw):
        return None

    location = item.get("location", {})
    geo = location.get("geo", {})
    address = location.get("address", {})

    date_str, time_str = _split_iso_datetime(start_raw)

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


def _parse_legacy_event(ev: dict) -> Optional[ScrapedEvent]:
    """Parse an event from the old flat events/discoverEvents list format."""
    name = ev.get("title") or ev.get("name")
    if not name:
        return None

    start_raw = ev.get("startDate") or ev.get("start") or ""
    end_raw = ev.get("endDate") or ev.get("end") or ""
    if not _is_future_or_present(start_raw, end_raw):
        return None

    event_id = ev.get("id") or ev.get("slug") or name
    location = ev.get("location", {}) or {}

    date_str, time_str = _split_iso_datetime(start_raw)

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_iso_datetime(raw: str) -> Optional[datetime]:
    """Parse an ISO-8601 datetime string into an aware UTC datetime."""
    if not raw or not isinstance(raw, str):
        return None
    s = raw.strip()
    # Normalize trailing 'Z' to '+00:00' so fromisoformat can parse it.
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        # Date-only values like "2024-10-01"
        try:
            dt = datetime.strptime(s[:10], "%Y-%m-%d")
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _is_future_or_present(start_raw: str, end_raw: str = "") -> bool:
    """Return True if the event has not ended yet.

    Uses endDate when available so events currently in progress are kept;
    otherwise falls back to startDate. Events missing both dates are kept
    (to avoid silently dropping items we can't classify).
    """
    now = datetime.now(timezone.utc)
    end_dt = _parse_iso_datetime(end_raw) if end_raw else None
    if end_dt is not None:
        return end_dt >= now
    start_dt = _parse_iso_datetime(start_raw) if start_raw else None
    if start_dt is not None:
        return start_dt >= now
    return True


def _split_iso_datetime(raw: str) -> tuple[Optional[str], Optional[str]]:
    """Split an ISO datetime string into (date, time) parts."""
    if not raw:
        return None, None
    if "T" in raw:
        parts = raw.split("T", 1)
        return parts[0], parts[1]
    return raw, None


def _extract_lat_lng_from_maps(maps_info: dict) -> tuple[Optional[float], Optional[float]]:
    """Extract latitude/longitude from Apple Maps or Google Maps URLs."""
    apple_url = maps_info.get("appleMapsUrl", "")
    m = re.search(r"sll=([-\d.]+),([-\d.]+)", apple_url)
    if m:
        try:
            return float(m.group(1)), float(m.group(2))
        except ValueError:
            pass

    google_url = maps_info.get("googleMapsUrl", "")
    m = re.search(r"query=([-\d.]+),([-\d.]+)", google_url)
    if m:
        try:
            return float(m.group(1)), float(m.group(2))
        except ValueError:
            pass

    return None, None


def _extract_image_url(image_data) -> Optional[str]:
    """Pull a usable image URL from Partiful's nested image object."""
    if image_data is None:
        return None
    if isinstance(image_data, str):
        return image_data
    if isinstance(image_data, dict):
        return (
            image_data.get("url")
            or (image_data.get("upload") or {}).get("url")
            or (image_data.get("gif") or {}).get("url")
        )
    return None


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
