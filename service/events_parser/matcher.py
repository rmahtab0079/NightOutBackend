"""Match scraped events to users based on their interests, cuisines, and location."""

from __future__ import annotations

import math
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
      2. Dietary compatibility — bonus if the restaurant matches a dietary preference
      3. Proximity
      4. Rating quality signal
      5. Swipe history boost
    """
    score = 0.0

    # Proximity
    if user_lat is not None and user_lon is not None and event.latitude and event.longitude:
        dist = _haversine_miles(user_lat, user_lon, event.latitude, event.longitude)
        if dist > max_radius_miles:
            return 0.0
        score += max(0, 1.0 - dist / max_radius_miles) * 0.5

    # Cuisine rank match (strongest signal for dining)
    event_tags_lower = {t.lower() for t in event.tags}
    event_text = f"{event.name} {event.description or ''} {event.genre or ''}".lower()

    for rank, cuisine in enumerate(user_cuisines):
        cuisine_lower = cuisine.lower()
        if cuisine_lower in event_tags_lower or cuisine_lower in event_text:
            # #1 ranked cuisine gets 3.0, #2 gets 2.7, etc.
            score += max(0.5, 3.0 - rank * 0.3)
            break

    # Dietary preference match — if the restaurant signals compatibility
    if user_dietary:
        dietary_lower = {d.lower() for d in user_dietary}
        diet_matches = event_tags_lower & dietary_lower
        score += len(diet_matches) * 1.0

        # Penalty for restaurants that conflict with dietary prefs
        if "Vegetarian" in user_dietary or "Vegan" in user_dietary:
            meat_signals = {"steak_house", "steakhouse", "bbq", "barbecue", "wings"}
            if event_tags_lower & meat_signals or any(s in event_text for s in meat_signals):
                score -= 1.5

        if "Halal" in user_dietary:
            non_halal = {"pork", "bacon", "ham"}
            if any(s in event_text for s in non_halal):
                score -= 1.5

    # Swipe history boost
    if liked_categories and "dining" in liked_categories:
        score += 0.5
    if liked_categories and "food" in liked_categories:
        score += 0.3

    # All restaurants get a baseline if user has cuisines set
    if user_cuisines:
        score += 0.3

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
) -> float:
    """
    Score an event's relevance for a user (0.0 = no match, higher = better).

    Dispatches to restaurant-specific scoring for dining items.
    """
    if event.category == "dining":
        return _score_restaurant(
            event, user_cuisines, user_dietary,
            user_lat, user_lon, max_radius_miles, liked_categories,
        )

    score = 0.0

    if user_lat is not None and user_lon is not None and event.latitude and event.longitude:
        dist = _haversine_miles(user_lat, user_lon, event.latitude, event.longitude)
        if dist > max_radius_miles:
            return 0.0
        score += max(0, 1.0 - dist / max_radius_miles) * 0.3

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

    food_tags = {"food", "restaurant", "brunch", "dinner", "drinks", "wine", "coffee", "brewery"}
    if event_tags_lower & food_tags:
        for i, cuisine in enumerate(user_cuisines):
            event_text = f"{event.name} {event.description or ''}".lower()
            if cuisine.lower() in event_text:
                rank_bonus = max(0.5, 1.5 - i * 0.15)
                score += rank_bonus

    if liked_categories and event.category and event.category in liked_categories:
        score += 0.8

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
    min_score: float = 0.3,
    max_events: int = 50,
) -> list[tuple[ScrapedEvent, float]]:
    """
    Score and rank all events for a user, returning top matches.

    Returns a list of (event, score) tuples sorted by descending score.
    """
    scored: list[tuple[ScrapedEvent, float]] = []

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
        )
        if s >= min_score:
            scored.append((event, s))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:max_events]


def group_events_by_category(
    matched: list[tuple[ScrapedEvent, float]],
) -> dict[str, list[dict]]:
    """
    Group matched events by their category tag.

    Returns a dict like:
      {
        "sports": [...],
        "music": [...],
        "arts": [...],
        "food": [...],
        "dining": [...],
        "outdoors": [...],
        "other": [...],
      }
    """
    groups: dict[str, list[dict]] = {}
    for event, score in matched:
        cat = event.category or "other"
        entry = event.to_dict()
        entry["relevance_score"] = round(score, 2)
        groups.setdefault(cat, []).append(entry)
    return groups
