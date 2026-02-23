"""Shared data models for the EventsParser pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional


@dataclass
class ScrapedEvent:
    source: str  # ticketmaster | eventbrite | amc | partiful
    source_id: str
    name: str
    date: Optional[str] = None
    time: Optional[str] = None
    venue_name: Optional[str] = None
    venue_address: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    image_url: Optional[str] = None
    description: Optional[str] = None
    url: Optional[str] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    currency: str = "USD"
    tags: list[str] = field(default_factory=list)
    category: Optional[str] = None
    genre: Optional[str] = None
    scraped_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


# Maps interest keywords to the canonical tag categories we write to Firebase.
# These align with the interest categories from user preferences.
INTEREST_TAGS: dict[str, list[str]] = {
    "sports": [
        "Soccer", "Basketball", "Volleyball", "Tennis", "Golf", "Swimming",
        "Running", "Cycling", "Boxing", "Bowling", "Rock Climbing",
        "Kayaking", "Fishing",
    ],
    "music": ["Live Music", "Dance Clubs", "Karaoke"],
    "arts": ["Comedy Shows", "Theater", "Art Galleries", "Museums", "Film Festivals"],
    "food": ["Wine Tasting", "Cooking Classes", "Brewery Tours", "Coffee Culture"],
    "outdoors": ["Hiking", "Camping"],
}

# Reverse lookup: interest -> tag category
INTEREST_TO_CATEGORY: dict[str, str] = {}
for _cat, _interests in INTEREST_TAGS.items():
    for _interest in _interests:
        INTEREST_TO_CATEGORY[_interest.lower()] = _cat

# Classification / keyword -> tag category (for matching scraped events)
CLASSIFICATION_TO_CATEGORY: dict[str, str] = {
    "sports": "sports",
    "music": "music",
    "arts": "arts",
    "arts & theatre": "arts",
    "film": "arts",
    "miscellaneous": "food",
    "comedy": "arts",
    "theatre": "arts",
    "dance": "music",
}
