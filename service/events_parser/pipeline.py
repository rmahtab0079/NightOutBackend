"""
EventsParser pipeline orchestrator.

Coordinates scraping from all sources, matches events to users,
and writes curated results to Firebase.
"""

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional

import requests

from .models import ScrapedEvent
from .ticketmaster_scraper import resolve_attraction_id, scrape_ticketmaster
from .eventbrite_scraper import scrape_eventbrite
from .amc_scraper import scrape_amc
from .partiful_scraper import scrape_partiful
from .restaurant_scraper import scrape_restaurants, tag_dietary_matches
from .matcher import (
    extract_artist_picks,
    group_events_by_category,
    match_events_to_user,
)
from .firebase_writer import (
    get_all_users_with_preferences,
    get_user_swipe_categories,
    purge_catalog_outside_user_radius,
    write_curated_events,
)

# Per-cluster keyword budgets, split so favorite artists aren't starved when a
# cluster also has many sports fans.
MAX_ARTIST_KEYWORDS_PER_CLUSTER = 25
MAX_TEAM_KEYWORDS_PER_CLUSTER = 15
# Events to fetch per keyword per source
KEYWORD_SEARCH_SIZE = 25
# Artist/team scraping uses a far wider time + space window than the generic
# scrape: arena tours announce 3-6 months out and fans will travel an hour or
# two for a favourite act, so a 14-day / 50-mile window finds essentially
# nothing for users like the test account in NJ asking for Beyoncé.
KEYWORD_DAYS_AHEAD = 180
KEYWORD_RADIUS_MILES = 150


def scrape_all_events(
    latitude: float,
    longitude: float,
    radius_miles: float = 50.0,
    days_ahead: int = 14,
) -> list[ScrapedEvent]:
    """Run all non-personalized scrapers for a given geographic center."""
    all_events: list[ScrapedEvent] = []

    print(f"\n--- Scraping events near ({latitude}, {longitude}), radius={radius_miles}mi ---")

    all_events.extend(scrape_ticketmaster(
        latitude, longitude,
        radius_miles=radius_miles,
        days_ahead=days_ahead,
        size=150,
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

    return _filter_events_with_images(all_events)


def _scrape_restaurants_for_user(
    latitude: float,
    longitude: float,
    radius_miles: float,
    cuisines: list[str],
    dietary: list[str],
) -> list[ScrapedEvent]:
    """Scrape restaurants from Google Places (50-mile radius by default), then tag dietary."""
    restaurants = scrape_restaurants(
        latitude, longitude,
        radius_miles=radius_miles,
        cuisines=cuisines if cuisines else None,
        dietary=dietary if dietary else None,
        max_per_cuisine=14,
    )
    restaurants = tag_dietary_matches(restaurants, dietary)
    return _filter_events_with_images(restaurants)


def _dedup(events: list[ScrapedEvent]) -> list[ScrapedEvent]:
    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    deduped: list[ScrapedEvent] = []
    for ev in events:
        id_key = f"{ev.source}:{ev.source_id}"
        name_key = ev.name.strip().lower()
        if id_key in seen_ids or (name_key and name_key in seen_names):
            continue
        seen_ids.add(id_key)
        if name_key:
            seen_names.add(name_key)
        deduped.append(ev)
    return deduped


def _has_usable_image(ev: ScrapedEvent) -> bool:
    """
    Cheap shape check: the URL exists and is http(s). This is the *first* gate;
    a network HEAD/GET probe in `_url_is_reachable` is the second gate.
    """
    url = (ev.image_url or "").strip().lower()
    if not url or url in {"none", "null", "false"}:
        return False
    return url.startswith("http://") or url.startswith("https://")


# Browser-like UA — some hosts (Partiful's Firebase Storage, Ticketmaster CDN)
# 403 on plain Python clients but allow normal browser requests through.
_IMAGE_PROBE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/*;q=0.8,*/*;q=0.5",
}
_PROBE_TIMEOUT_SEC = 5.0


def _url_is_reachable(url: str) -> bool:
    """
    Returns True if `url` responds with a 2xx to either a HEAD or a tiny ranged
    GET. Many image CDNs/APIs reject HEAD with non-standard codes (e.g. Google
    Places photo media endpoints return 404 on HEAD even when GET succeeds), so
    if HEAD doesn't return 2xx we always fall through to a ranged GET.
    """
    try:
        r = requests.head(
            url,
            headers=_IMAGE_PROBE_HEADERS,
            timeout=_PROBE_TIMEOUT_SEC,
            allow_redirects=True,
        )
        if 200 <= r.status_code < 300:
            return True
        # HEAD support is unreliable across image hosts:
        #   - S3/Firebase Storage answer 403 on HEAD, 200 on GET
        #   - Google Places /photos/.../media answers 404 on HEAD, 200 on GET
        #   - Some CDNs answer 405 (method not allowed) on HEAD
        # Always retry once with a 1-byte ranged GET before declaring the URL
        # unreachable, otherwise we drop perfectly valid images.
        r2 = requests.get(
            url,
            headers={**_IMAGE_PROBE_HEADERS, "Range": "bytes=0-0"},
            timeout=_PROBE_TIMEOUT_SEC,
            allow_redirects=True,
            stream=True,
        )
        r2.close()
        return 200 <= r2.status_code < 300
    except requests.RequestException:
        return False


def _filter_events_with_images(
    events: list[ScrapedEvent],
    *,
    probe_network: bool = True,
    max_workers: int = 16,
) -> list[ScrapedEvent]:
    """
    Two-pass filter:
      1. Drop anything missing a syntactically valid http(s) image_url.
      2. (optional) Probe each remaining URL in parallel and drop any that
         don't return 2xx. This is what kills stale/forbidden Partiful URLs
         before they ever reach Firestore.
    """
    shape_pass = [ev for ev in events if _has_usable_image(ev)]
    shape_dropped = len(events) - len(shape_pass)

    if not probe_network or not shape_pass:
        if shape_dropped:
            print(f"  [filter] Dropped {shape_dropped} item(s) without image_url")
        return shape_pass

    reachable: list[ScrapedEvent] = []
    network_dropped = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_url_is_reachable, ev.image_url): ev
                   for ev in shape_pass}
        for fut in as_completed(futures):
            ev = futures[fut]
            try:
                ok = fut.result()
            except Exception:
                ok = False
            if ok:
                reachable.append(ev)
            else:
                network_dropped += 1

    total_dropped = shape_dropped + network_dropped
    if total_dropped:
        print(
            f"  [filter] Dropped {total_dropped} item(s) without usable image "
            f"({shape_dropped} bad URL, {network_dropped} unreachable)"
        )
    return reachable


def _scrape_keyword_events(
    latitude: float,
    longitude: float,
    artists: list[str],
    teams: list[str],
) -> list[ScrapedEvent]:
    """
    Run the per-cluster artist/team keyword scrapes.

    Split into its own function so the per-user fast path
    (`run_pipeline_for_user`) can reuse exactly the same scraping logic the
    full cron uses, instead of inlining a divergent copy.

    Artists are pre-tagged with `_matched_artist` at scrape time so even
    events whose name doesn't textually contain the artist (e.g.
    "An Evening With ...") still wind up in artist_picks.
    """
    artist_keyword_events: list[ScrapedEvent] = []
    for artist in artists:
        # Resolve to an attractionId first — TM's keyword search misses a
        # ton of accent / punctuation variations ("Beyoncé" vs "Beyonce",
        # "AC/DC" vs "ACDC") that the attraction lookup normalizes.
        attraction_id = resolve_attraction_id(artist)
        tm_events = scrape_ticketmaster(
            latitude,
            longitude,
            radius_miles=KEYWORD_RADIUS_MILES,
            days_ahead=KEYWORD_DAYS_AHEAD,
            size=KEYWORD_SEARCH_SIZE,
            keyword=artist if not attraction_id else None,
            attraction_id=attraction_id,
        )
        # Only pre-tag _matched_artist when we resolved a canonical
        # attractionId — those events are guaranteed to be the actual artist.
        # The free-text keyword fallback is fuzzy enough to surface unrelated
        # acts (it once tagged "Model/Actriz" as a Lady Gaga show), so we let
        # the matcher's `_first_keyword_hit` decide whether the event name
        # actually contains the artist's name before stamping it.
        if attraction_id:
            for ev in tm_events:
                if not getattr(ev, "_matched_artist", None):
                    ev._matched_artist = artist  # type: ignore[attr-defined]
        artist_keyword_events.extend(tm_events)

    team_keyword_events: list[ScrapedEvent] = []
    for team in teams:
        attraction_id = resolve_attraction_id(team)
        team_keyword_events.extend(
            scrape_ticketmaster(
                latitude,
                longitude,
                radius_miles=KEYWORD_RADIUS_MILES,
                days_ahead=KEYWORD_DAYS_AHEAD,
                size=KEYWORD_SEARCH_SIZE,
                keyword=team if not attraction_id else None,
                attraction_id=attraction_id,
            )
        )

    keyword_events = _filter_events_with_images(
        _dedup(artist_keyword_events + team_keyword_events)
    )
    if artists or teams:
        print(
            f"  Cluster: {len(artists)} artists + "
            f"{len(teams)} teams -> {len(keyword_events)} extra events"
        )
    return keyword_events


def _process_user_with_events(
    user: dict,
    shared_events: list[ScrapedEvent],
    keyword_events: list[ScrapedEvent],
    run_id: str,
    radius_miles: float,
    max_events_per_user: int,
) -> Optional[dict]:
    """
    Score & persist the per-user curated payload. Returns a small stats dict
    (`{"restaurants": int, "matched": int}`) or None if the user is unusable
    (missing email).
    """
    email = user.get("email", "")
    if not email:
        return None

    interests = user.get("interests", [])
    cuisines = user.get("cuisines", [])
    dietary = user.get("dietaryPreferences", [])
    favorite_teams = user.get("favoriteTeams") or []
    favorite_artists = user.get("favoriteArtists") or []
    user_lat = user.get("last_latitude")
    user_lon = user.get("last_longitude")

    # Restaurants are scoped to the *user's* exact coords (not the cluster
    # centroid) so a nomadic user gets fresh dining options the moment they
    # update their location.
    restaurant_lat = (
        float(user_lat) if user_lat is not None else None
    )
    restaurant_lon = (
        float(user_lon) if user_lon is not None else None
    )
    if restaurant_lat is None or restaurant_lon is None:
        user_restaurants: list[ScrapedEvent] = []
    else:
        user_restaurants = _scrape_restaurants_for_user(
            latitude=restaurant_lat,
            longitude=restaurant_lon,
            radius_miles=radius_miles,
            cuisines=cuisines,
            dietary=dietary,
        )

    all_items = _dedup(shared_events + keyword_events + user_restaurants)
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
        user_favorite_teams=favorite_teams,
        user_favorite_artists=favorite_artists,
        max_events=max_events_per_user,
    )

    grouped = group_events_by_category(matched)
    artist_picks = extract_artist_picks(matched, favorite_artists)
    write_curated_events(email, grouped, run_id, artist_picks=artist_picks)

    dining_count = len(grouped.get("dining", []))
    print(
        f"  {email}: {len(matched)} matched ({dining_count} restaurants, "
        f"{len(artist_picks)} artist picks) across {len(grouped)} categories"
    )
    return {"restaurants": len(user_restaurants), "matched": len(matched)}


def run_pipeline_for_user(
    email: str,
    radius_miles: float = 50.0,
    days_ahead: int = 14,
    max_events_per_user: int = 80,
) -> Optional[dict]:
    """
    Re-run the events parser for a single user. Used after the user's stored
    location changes meaningfully — re-running the entire cron just to refresh
    one user's coords would scrape every other user's region for nothing.

    Returns a small summary dict or None when the user is missing prefs /
    location.
    """
    from .firebase_writer import _get_db  # local import keeps import cheap

    db = _get_db()
    doc = db.collection("user_preferences").document(email).get()
    if not doc.exists:
        print(f"[per_user_pipeline] no prefs for {email}, skipping")
        return None
    prefs = doc.to_dict() or {}
    prefs["email"] = email

    lat = prefs.get("last_latitude")
    lon = prefs.get("last_longitude")
    if lat is None or lon is None:
        print(f"[per_user_pipeline] no location for {email}, skipping")
        return None
    try:
        lat_f = float(lat)
        lon_f = float(lon)
    except (TypeError, ValueError):
        print(f"[per_user_pipeline] bad location for {email}, skipping")
        return None

    run_id = (
        f"user_run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_"
        f"{uuid.uuid4().hex[:6]}"
    )
    print(f"\n[per_user_pipeline] {email} @ ({lat_f}, {lon_f}) — {run_id}")

    shared_events = scrape_all_events(
        latitude=lat_f,
        longitude=lon_f,
        radius_miles=radius_miles,
        days_ahead=days_ahead,
    )

    artist_keywords = [
        a.strip()
        for a in (prefs.get("favoriteArtists") or [])
        if isinstance(a, str) and a.strip()
    ][:MAX_ARTIST_KEYWORDS_PER_CLUSTER]
    team_keywords = [
        t.strip()
        for t in (prefs.get("favoriteTeams") or [])
        if isinstance(t, str) and t.strip()
    ][:MAX_TEAM_KEYWORDS_PER_CLUSTER]

    keyword_events = _scrape_keyword_events(
        latitude=lat_f,
        longitude=lon_f,
        artists=artist_keywords,
        teams=team_keywords,
    )

    stats = _process_user_with_events(
        user=prefs,
        shared_events=shared_events,
        keyword_events=keyword_events,
        run_id=run_id,
        radius_miles=radius_miles,
        max_events_per_user=max_events_per_user,
    )

    return {
        "run_id": run_id,
        "email": email,
        "events_scraped": len(shared_events) + len(keyword_events),
        "restaurants_scraped": (stats or {}).get("restaurants", 0),
        "events_matched": (stats or {}).get("matched", 0),
        "completed_at": datetime.utcnow().isoformat() + "Z",
    }


def run_pipeline(
    radius_miles: float = 50.0,
    days_ahead: int = 14,
    max_events_per_user: int = 80,
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
    skipped_no_location: list[str] = []
    for user in users:
        lat = user.get("last_latitude")
        lon = user.get("last_longitude")
        if lat is None or lon is None:
            # Refuse to silently fall back to a hard-coded city -- that's how
            # users in places like Orland Park ended up with NYC events.
            email = user.get("email", "<unknown>")
            skipped_no_location.append(email)
            continue

        try:
            lat = float(lat)
            lon = float(lon)
        except (TypeError, ValueError):
            skipped_no_location.append(user.get("email", "<unknown>"))
            continue

        cluster_key = f"{round(lat, 1)}_{round(lon, 1)}"
        if cluster_key not in location_clusters:
            location_clusters[cluster_key] = {
                "latitude": lat,
                "longitude": lon,
                "users": [],
                "events": None,
            }
        location_clusters[cluster_key]["users"].append(user)

    if skipped_no_location:
        print(
            f"Skipped {len(skipped_no_location)} user(s) with no stored "
            f"location: {skipped_no_location[:5]}"
            + (" ..." if len(skipped_no_location) > 5 else "")
        )
    print(f"Grouped into {len(location_clusters)} location cluster(s)")

    # Drop catalog entries that are no longer near any active user. This is
    # what keeps a one-off SF user's events from haunting other users' feeds
    # forever after that user churns or moves.
    try:
        purge_catalog_outside_user_radius(radius_miles=radius_miles)
    except Exception as e:
        print(f"  [pipeline] catalog purge failed (continuing): {e}")

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

        artists_set: set[str] = set()
        teams_set: set[str] = set()
        for user in cluster["users"]:
            for a in user.get("favoriteArtists") or []:
                if a and isinstance(a, str) and a.strip():
                    artists_set.add(a.strip())
            for t in user.get("favoriteTeams") or []:
                if t and isinstance(t, str) and t.strip():
                    teams_set.add(t.strip())

        keyword_events = _scrape_keyword_events(
            latitude=cluster["latitude"],
            longitude=cluster["longitude"],
            artists=list(artists_set)[:MAX_ARTIST_KEYWORDS_PER_CLUSTER],
            teams=list(teams_set)[:MAX_TEAM_KEYWORDS_PER_CLUSTER],
        )
        total_events += len(keyword_events)

        for user in cluster["users"]:
            stats = _process_user_with_events(
                user=user,
                shared_events=shared_events,
                keyword_events=keyword_events,
                run_id=run_id,
                radius_miles=radius_miles,
                max_events_per_user=max_events_per_user,
            )
            if stats is None:
                continue
            total_restaurants += stats["restaurants"]
            total_matched += stats["matched"]

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
