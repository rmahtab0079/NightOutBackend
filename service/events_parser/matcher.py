"""Match scraped events to users based on their interests, cuisines, and location.

Scoring incorporates:
  - Content relevance (interests, cuisines, dietary, keywords)
  - Proximity
  - Popularity signals (RSVP counts, ratings, selling-fast flags)
  - Temporal signals (event proximity in time, day-of-week fit, freshness)
  - Swipe history
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Optional

from .models import ScrapedEvent, INTEREST_TO_CATEGORY, INTEREST_TAGS


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in miles."""
    R = 3958.8
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Popularity scoring
# ---------------------------------------------------------------------------

# Thresholds for normalizing RSVP/interested counts (Partiful scale).
_RSVP_HIGH = 200
_INTERESTED_HIGH = 300
# Thresholds for restaurant ratings.
_RATING_FLOOR = 3.5
_RATING_CEILING = 5.0
_REVIEW_COUNT_HIGH = 500


def _popularity_bonus(event: ScrapedEvent) -> float:
    """Compute a 0–1 popularity score from available signals, then scale to bonus points."""
    MAX_BONUS = 1.0
    raw = 0.0
    signals = 0

    if event.popularity_score is not None:
        raw += event.popularity_score
        signals += 1

    if event.rsvp_count is not None and event.rsvp_count > 0:
        raw += min(1.0, event.rsvp_count / _RSVP_HIGH)
        signals += 1

    if event.interested_count is not None and event.interested_count > 0:
        raw += min(1.0, event.interested_count / _INTERESTED_HIGH)
        signals += 1

    if event.rating is not None:
        norm = max(0.0, event.rating - _RATING_FLOOR) / (_RATING_CEILING - _RATING_FLOOR)
        raw += min(1.0, norm)
        signals += 1

    if event.review_count is not None and event.review_count > 0:
        raw += min(1.0, event.review_count / _REVIEW_COUNT_HIGH)
        signals += 1

    if event.is_selling_fast:
        raw += 0.7
        signals += 1

    if signals == 0:
        return 0.0

    return (raw / signals) * MAX_BONUS


# ---------------------------------------------------------------------------
# Temporal scoring
# ---------------------------------------------------------------------------

_WEEKEND_CATEGORIES = frozenset({"music", "sports", "food", "dining"})
_EVENING_CATEGORIES = frozenset({"music", "food", "dining"})
_MORNING_TAGS = frozenset({"brunch", "breakfast", "yoga", "run", "hike", "fitness"})


def _temporal_bonus(event: ScrapedEvent, now: Optional[datetime] = None) -> float:
    """Compute a temporal relevance bonus (0–1.5 range).

    Components:
      - Event proximity: events happening sooner get a boost
      - Weekend fit: party/music/sports events on Fri/Sat get a boost
      - Freshness: recently scraped events get a small boost
    """
    if now is None:
        now = datetime.utcnow()

    bonus = 0.0

    event_dt = _parse_event_datetime(event)

    if event_dt is not None:
        days_away = (event_dt - now).total_seconds() / 86400.0
        if days_away < 0:
            days_away = 0.0

        # Sooner events rank higher: full bonus at 0 days, decays over 14 days
        if days_away <= 14:
            bonus += 0.6 * max(0.0, 1.0 - days_away / 14.0)
        elif days_away <= 30:
            bonus += 0.2 * max(0.0, 1.0 - (days_away - 14) / 16.0)

        event_weekday = event_dt.weekday()  # 0=Mon, 4=Fri, 5=Sat, 6=Sun
        cat = (event.category or "").lower()
        tags_lower = {t.lower() for t in event.tags}

        # Weekend boost for nightlife/sports/dining events on Fri-Sun
        if event_weekday >= 4 and cat in _WEEKEND_CATEGORIES:
            bonus += 0.3

        # Evening events in evening-appropriate categories
        if event_dt.hour >= 17 and cat in _EVENING_CATEGORIES:
            bonus += 0.15

        # Morning/daytime events with morning tags
        if event_dt.hour < 14 and tags_lower & _MORNING_TAGS:
            bonus += 0.15

    # Freshness: small bonus for items scraped recently (within the last hour)
    try:
        scraped = datetime.fromisoformat(event.scraped_at.replace("Z", "+00:00")).replace(tzinfo=None)
        hours_old = (now - scraped).total_seconds() / 3600.0
        if hours_old < 1:
            bonus += 0.2
        elif hours_old < 6:
            bonus += 0.1
    except (ValueError, AttributeError):
        pass

    return bonus


def _parse_event_datetime(event: ScrapedEvent) -> Optional[datetime]:
    """Best-effort parse of event date + time into a datetime."""
    if not event.date:
        return None
    try:
        date_str = event.date.strip()
        time_str = (event.time or "").strip()
        # Strip timezone suffixes for naive comparison
        for suffix in ("Z", "+00:00", ".000Z"):
            time_str = time_str.replace(suffix, "")
            date_str = date_str.replace(suffix, "")
        if time_str:
            return datetime.strptime(f"{date_str} {time_str[:8]}", "%Y-%m-%d %H:%M:%S")
        return datetime.strptime(date_str[:10], "%Y-%m-%d")
    except (ValueError, IndexError):
        return None


def _score_restaurant(
    event: ScrapedEvent,
    user_cuisines: list[str],
    user_dietary: list[str],
    user_lat: Optional[float],
    user_lon: Optional[float],
    max_radius_miles: float,
    liked_categories: Optional[set[str]],
) -> float:
    """
    Dedicated scoring for restaurants / dining items.

    Ranking factors:
      1. Cuisine rank bonus — #1 cuisine gets the biggest boost, diminishing for lower ranks
      2. Dietary compatibility
      3. Proximity
      4. Popularity (Google rating + review count)
      5. Swipe history boost
    """
    score = 0.0

    if user_lat is not None and user_lon is not None and event.latitude and event.longitude:
        dist = _haversine_miles(user_lat, user_lon, event.latitude, event.longitude)
        if dist > max_radius_miles:
            return 0.0
        score += max(0, 1.0 - dist / max_radius_miles) * 0.5

    event_tags_lower = {t.lower() for t in event.tags}
    event_text = f"{event.name} {event.description or ''} {event.genre or ''}".lower()

    for rank, cuisine in enumerate(user_cuisines):
        cuisine_lower = cuisine.lower()
        if cuisine_lower in event_tags_lower or cuisine_lower in event_text:
            score += max(0.5, 3.0 - rank * 0.3)
            break

    if user_dietary:
        dietary_lower = {d.lower() for d in user_dietary}
        diet_matches = event_tags_lower & dietary_lower

        is_halal_user = "Halal" in user_dietary
        if is_halal_user and "halal" in diet_matches:
            score += 2.0
            diet_matches = diet_matches - {"halal"}
        score += len(diet_matches) * 1.0

        if "Vegetarian" in user_dietary or "Vegan" in user_dietary:
            meat_signals = {"steak_house", "steakhouse", "bbq", "barbecue", "wings"}
            if event_tags_lower & meat_signals or any(s in event_text for s in meat_signals):
                score -= 1.5

        if is_halal_user:
            hard_non_halal = {"pork", "bacon", "ham"}
            if any(s in event_text for s in hard_non_halal):
                score -= 1.5
            soft_non_halal = {"wine bar", "beer garden", "brewery", "brewpub", "hookah bar"}
            if event_tags_lower & soft_non_halal or any(s in event_text for s in soft_non_halal):
                score -= 0.5

    if liked_categories and "dining" in liked_categories:
        score += 0.5
    if liked_categories and "food" in liked_categories:
        score += 0.3

    if user_cuisines:
        score += 0.3

    # Popularity: Google rating & review count
    score += _popularity_bonus(event) * 0.8

    return score


def _keyword_match_bonus(
    event: ScrapedEvent,
    keywords: list[str],
    bonus_per_match: float = 1.8,
) -> float:
    """Bonus when event name/description contains any of the given keywords (e.g. team or artist)."""
    if not keywords:
        return 0.0
    text = f"{event.name} {event.description or ''}".lower()
    score = 0.0
    for kw in keywords:
        if kw and kw.lower() in text:
            score += bonus_per_match
    return score


def score_event_for_user(
    event: ScrapedEvent,
    user_interests: list[str],
    user_cuisines: list[str],
    user_dietary: list[str],
    user_lat: Optional[float] = None,
    user_lon: Optional[float] = None,
    max_radius_miles: float = 30.0,
    liked_categories: Optional[set[str]] = None,
    user_favorite_teams: Optional[list[str]] = None,
    user_favorite_artists: Optional[list[str]] = None,
    now: Optional[datetime] = None,
) -> float:
    """
    Score an event's relevance for a user (0.0 = no match, higher = better).

    Components:
      - Content relevance (interests, cuisines, keywords)  ~0–6 pts
      - Proximity                                          ~0–0.3 pts
      - Popularity bonus                                   ~0–1.0 pts
      - Temporal bonus                                     ~0–1.5 pts
      - Swipe history                                      ~0–0.8 pts
    """
    if event.category == "dining":
        return _score_restaurant(
            event, user_cuisines, user_dietary,
            user_lat, user_lon, max_radius_miles, liked_categories,
        )

    score = 0.0

    # --- Proximity ---
    if user_lat is not None and user_lon is not None and event.latitude and event.longitude:
        dist = _haversine_miles(user_lat, user_lon, event.latitude, event.longitude)
        if dist > max_radius_miles:
            return 0.0
        score += max(0, 1.0 - dist / max_radius_miles) * 0.3

    # --- Content relevance ---
    user_categories: set[str] = set()
    for interest in user_interests:
        cat = INTEREST_TO_CATEGORY.get(interest.lower())
        if cat:
            user_categories.add(cat)

    if event.category and event.category in user_categories:
        score += 2.0

    event_tags_lower = {t.lower() for t in event.tags}
    user_interests_lower = {i.lower() for i in user_interests}
    tag_overlap = event_tags_lower & user_interests_lower
    score += len(tag_overlap) * 1.5

    score += _keyword_match_bonus(event, user_favorite_teams or [], bonus_per_match=1.8)
    score += _keyword_match_bonus(event, user_favorite_artists or [], bonus_per_match=1.8)

    food_tags = {"food", "restaurant", "brunch", "dinner", "drinks", "wine", "coffee", "brewery"}
    if event_tags_lower & food_tags:
        for i, cuisine in enumerate(user_cuisines):
            event_text = f"{event.name} {event.description or ''}".lower()
            if cuisine.lower() in event_text:
                rank_bonus = max(0.5, 1.5 - i * 0.15)
                score += rank_bonus

    # --- Swipe history ---
    if liked_categories and event.category and event.category in liked_categories:
        score += 0.8

    # --- Popularity bonus ---
    score += _popularity_bonus(event)

    # --- Temporal bonus ---
    score += _temporal_bonus(event, now=now)

    # --- Fallback for users with no preferences ---
    if not user_interests and not user_cuisines:
        score += 0.5

    return score


def match_events_to_user(
    events: list[ScrapedEvent],
    user_interests: list[str],
    user_cuisines: list[str],
    user_dietary: list[str],
    user_lat: Optional[float] = None,
    user_lon: Optional[float] = None,
    max_radius_miles: float = 30.0,
    liked_categories: Optional[set[str]] = None,
    user_favorite_teams: Optional[list[str]] = None,
    user_favorite_artists: Optional[list[str]] = None,
    min_score: float = 0.3,
    max_events: int = 50,
) -> list[tuple[ScrapedEvent, float]]:
    """
    Score and rank all events for a user, returning top matches.

    Returns a list of (event, score) tuples sorted by descending score.
    """
    scored: list[tuple[ScrapedEvent, float]] = []
    now = datetime.utcnow()

    for event in events:
        s = score_event_for_user(
            event,
            user_interests,
            user_cuisines,
            user_dietary,
            user_lat,
            user_lon,
            max_radius_miles,
            liked_categories,
            user_favorite_teams=user_favorite_teams,
            user_favorite_artists=user_favorite_artists,
            now=now,
        )
        if s >= min_score:
            scored.append((event, s))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:max_events]


# Only these sources are real dining/places; others must not go in "food".
_DINING_SOURCES = frozenset({"google_places"})


def group_events_by_category(
    matched: list[tuple[ScrapedEvent, float]],
) -> dict[str, list[dict]]:
    """
    Group matched events by their category tag.

    The "food" key is reserved for real dining/places only (Google Places).
    Events from Ticketmaster/Eventbrite/Partiful that were tagged "food" are
    grouped under "other" so the Food tab never shows theatre/entertainment.
    """
    groups: dict[str, list[dict]] = {}
    for event, score in matched:
        cat = event.category or "other"
        # Prevent entertainment events from appearing under "food"
        if cat == "food" and event.source not in _DINING_SOURCES:
            cat = "other"
        entry = event.to_dict()
        entry["relevance_score"] = round(score, 2)
        groups.setdefault(cat, []).append(entry)
    return groups
