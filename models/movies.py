from typing import List
from models.asset_helper import get_assets


def get_movies(start_year: int = 1970, end_year: int = 2026, genres: List[int] = None):
    if genres is None:
        genres = []
    return get_assets("movie", start_year=start_year, end_year=end_year, genres=genres)