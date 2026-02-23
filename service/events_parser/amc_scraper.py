"""AMC Theatres scraper — fetches now-playing movies and showtimes near a location."""

from __future__ import annotations

import requests
from datetime import datetime
from typing import Optional

from .models import ScrapedEvent

_AMC_BASE = "https://api.amctheatres.com"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "X-AMC-Vendor-Key": "AMCCE_1",
}


def scrape_amc(
    latitude: float,
    longitude: float,
    radius_miles: float = 25.0,
) -> list[ScrapedEvent]:
    """
    Fetch currently playing movies at nearby AMC theatres.

    Uses AMC's public-facing API endpoints that their website calls.
    """
    results: list[ScrapedEvent] = []

    theatres = _find_nearby_theatres(latitude, longitude, radius_miles)
    if not theatres:
        print("[amc] No nearby theatres found, trying web scrape")
        return _scrape_amc_web(latitude, longitude)

    today = datetime.utcnow().strftime("%Y-%m-%d")
    seen_movies: set[str] = set()

    for theatre in theatres[:3]:
        theatre_id = theatre.get("id")
        theatre_name = theatre.get("name", "AMC Theatre")
        theatre_address = theatre.get("address", "")
        theatre_lat = theatre.get("latitude")
        theatre_lon = theatre.get("longitude")

        showtimes = _get_showtimes(theatre_id, today)
        for show in showtimes:
            movie_name = show.get("movie_name", "")
            movie_key = f"{movie_name}_{theatre_id}"
            if movie_key in seen_movies:
                continue
            seen_movies.add(movie_key)

            results.append(ScrapedEvent(
                source="amc",
                source_id=f"amc_{theatre_id}_{show.get('movie_id', '')}",
                name=movie_name,
                date=today,
                time=show.get("first_showtime"),
                venue_name=theatre_name,
                venue_address=theatre_address,
                latitude=theatre_lat,
                longitude=theatre_lon,
                image_url=show.get("poster_url"),
                description=show.get("synopsis"),
                url=show.get("url"),
                tags=["movies", "film"],
                category="arts",
                genre=show.get("genre"),
            ))

    print(f"[amc] Scraped {len(results)} movies from {len(theatres)} theatres")
    return results


def _find_nearby_theatres(lat: float, lon: float, radius: float) -> list[dict]:
    """Find AMC theatres near the given coordinates."""
    try:
        url = f"{_AMC_BASE}/v2/theatres"
        params = {
            "latitude": str(lat),
            "longitude": str(lon),
            "radius": str(int(radius)),
            "page-size": "5",
        }
        resp = requests.get(url, params=params, headers=_HEADERS, timeout=10)
        if resp.status_code != 200:
            return []

        data = resp.json()
        theatres = []
        embedded = data.get("_embedded", {}).get("theatres", [])
        for t in embedded:
            location = t.get("location", {})
            address = location.get("address", {})
            theatres.append({
                "id": t.get("id"),
                "name": t.get("name"),
                "address": f"{address.get('line1', '')}, {address.get('city', '')}, {address.get('state', '')}",
                "latitude": location.get("geoCoordinates", {}).get("latitude"),
                "longitude": location.get("geoCoordinates", {}).get("longitude"),
            })
        return theatres
    except Exception as e:
        print(f"[amc] Theatre search error: {e}")
        return []


def _get_showtimes(theatre_id: int, date: str) -> list[dict]:
    """Get showtimes for a specific AMC theatre on a given date."""
    try:
        url = f"{_AMC_BASE}/v2/theatres/{theatre_id}/showtimes/{date}"
        resp = requests.get(url, headers=_HEADERS, timeout=10)
        if resp.status_code != 200:
            return []

        data = resp.json()
        showtimes_raw = data.get("_embedded", {}).get("showtimes", [])
        movies: dict[str, dict] = {}

        for s in showtimes_raw:
            movie = s.get("_embedded", {}).get("movie", {})
            movie_id = str(movie.get("id", ""))
            if movie_id not in movies:
                genres = movie.get("genres", [])
                genre_str = genres[0] if genres else None
                media = movie.get("media", {})
                poster = None
                for m in media.get("posterDynamic", []) or []:
                    poster = m.get("url")
                    break
                if not poster:
                    poster = media.get("posterLarge")

                movies[movie_id] = {
                    "movie_id": movie_id,
                    "movie_name": movie.get("name", "Unknown"),
                    "synopsis": movie.get("synopsis"),
                    "genre": genre_str,
                    "poster_url": poster,
                    "url": movie.get("websiteUrl"),
                    "first_showtime": s.get("showDateTimeLocal", "").split("T")[-1][:5] if s.get("showDateTimeLocal") else None,
                }
        return list(movies.values())
    except Exception as e:
        print(f"[amc] Showtime fetch error: {e}")
        return []


def _scrape_amc_web(lat: float, lon: float) -> list[ScrapedEvent]:
    """Fallback: scrape AMC's on-screen page for currently playing movies."""
    results: list[ScrapedEvent] = []
    try:
        url = "https://www.amctheatres.com/movies"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
            ),
        }
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code != 200:
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
                    if item.get("@type") != "Movie":
                        continue
                    results.append(ScrapedEvent(
                        source="amc",
                        source_id=f"amc_web_{item.get('name', '')}",
                        name=item.get("name", "Unknown"),
                        date=datetime.utcnow().strftime("%Y-%m-%d"),
                        image_url=item.get("image"),
                        description=item.get("description"),
                        url=item.get("url"),
                        tags=["movies", "film"],
                        category="arts",
                    ))
            except (json.JSONDecodeError, KeyError):
                continue

        print(f"[amc-web] Scraped {len(results)} movies")
    except Exception as e:
        print(f"[amc-web] Error: {e}")

    return results
