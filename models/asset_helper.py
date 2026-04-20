import pandas as pd
import requests
import os
import threading
from queue import Queue
from typing import List


# file_path = "./models/movies_dataset.csv"

# # Load the dataset
# df = pd.read_csv(file_path)

# # Filter movies with more than 1000 ratings
# filtered_df = df[df["vote_count"] > 1000]

# # Group by original language
# grouped_by_language = filtered_df.groupby("original_language")

# # Create a DataFrame for English movies and sort by vote average
# english_movies_sorted = (
#     grouped_by_language.get_group("en")
#     .sort_values(by="vote_average", ascending=False)
# )

BASE_IMAGE_URL = "https://image.tmdb.org/t/p/"
# TMDB poster sizes: w92, w154, w185, w342, w500, w780, original.
# w200 (the historical default) is a 200-pixel-wide JPEG, which looks soft
# when stretched across a modern 3x phone hero. w780 is roughly 4x the pixel
# count and the largest dedicated poster size short of `original`.
DEFAULT_IMAGE_SIZE = "w780"
# TMDB backdrop sizes: w300, w780, w1280, original. Hero images render
# landscape, so use the widest dedicated size.
DEFAULT_BACKDROP_SIZE = "w1280"


def _extract_year(date_str: str) -> int | None:
    """Extract year from a date string like '2024-05-15' or '2024'."""
    if not date_str:
        return None
    try:
        return int(date_str.split("-")[0])
    except (ValueError, IndexError):
        return None


def _filter_assets(
    assets: List[dict],
    asset_type: str,
    start_year: int,
    end_year: int,
    genres: List[int],
) -> List[dict]:
    """
    Filter assets by genre IDs and release year range.
    - asset_type: 'movie' uses 'release_date', 'tv' uses 'first_air_date'
    - genres: if non-empty, asset must have at least one matching genre_id
    - start_year/end_year: inclusive range for the release/air year
    """
    date_field = "release_date" if asset_type == "movie" else "first_air_date"
    filtered = []

    for asset in assets:
        # --- Genre filter ---
        if genres:
            asset_genres = asset.get("genre_ids", [])
            if not any(g in genres for g in asset_genres):
                continue

        # --- Year filter ---
        date_str = asset.get(date_field, "")
        year = _extract_year(date_str)
        if year is not None:
            if year < start_year or year > end_year:
                continue

        filtered.append(asset)

    return filtered


def get_assets(
    asset_type: str,
    start_year: int = 1970,
    end_year: int = 2026,
    genres: List[int] = None,
):
    if genres is None:
        genres = []

    tmdb_api_key = os.getenv("TMDB_API_KEY")
    results_queue = Queue()

    threads = []
    for i in range(1, 100):
        t = threading.Thread(target=get_assets_page, args=(tmdb_api_key, i, results_queue, asset_type))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    responses = []
    while not results_queue.empty():
        responses.append(results_queue.get())

    responses.sort(key=lambda x: x["page"])

    # Flatten pages and build a TMDB-like envelope expected by the client
    merged_results = []
    for r in responses:
        merged_results.extend(r.get("results", []))

    # Apply filters
    filtered_results = _filter_assets(
        merged_results,
        asset_type=asset_type,
        start_year=start_year,
        end_year=end_year,
        genres=genres,
    )

    return {
        "page": 1,
        "total_pages": 1,
        "total_results": len(filtered_results),
        "results": filtered_results,
    }


def get_assets_page(api_key, page, results_queue, asset_type: str):
    base_image_url = os.getenv("BASE_IMAGE_URL") or BASE_IMAGE_URL
    poster_size = os.getenv("DEFAULT_IMAGE_SIZE") or DEFAULT_IMAGE_SIZE
    backdrop_size = os.getenv("DEFAULT_BACKDROP_SIZE") or DEFAULT_BACKDROP_SIZE
    url = f"https://api.themoviedb.org/3/{asset_type}/top_rated?language=en-US&page={page}&api_key={api_key}"
    #url = f"https://api.themoviedb.org/3/discover/movie?include_adult=false&include_video=false&language=en-US&page=1&sort_by=popularity.desc&api_key={tmdb_api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        # Parse JSON response
        data = response.json()
        movies = []
        for movie in data.get("results", []):
            poster_path = movie.get("poster_path")
            if poster_path:
                movie["poster_url"] = f"{base_image_url}{poster_size}{poster_path}"
                # Also expose the original-resolution URL so very high-DPI
                # devices can opt into it without another roundtrip.
                movie["poster_url_original"] = f"{base_image_url}original{poster_path}"
            backdrop_path = movie.get("backdrop_path")
            if backdrop_path:
                # Backdrops are landscape and a much better hero source than
                # the squarish poster on a phone in portrait orientation.
                movie["backdrop_url"] = f"{base_image_url}{backdrop_size}{backdrop_path}"
                movie["backdrop_url_original"] = f"{base_image_url}original{backdrop_path}"
            if movie["original_language"] == "en":
                movies.append(movie)
        # Add the modified response to the queue
        data["results"] = movies
        results_queue.put(data)

    else:
        print(f"Request failed with status code {response.status_code}")


def get_movies_diff():
    """
    Makes a GET request to the specified URL with optional parameters and headers.

    Args:
        url (str): The endpoint URL.
        params (dict): Query parameters to send with the request.
        headers (dict): Headers to send with the request.

    Returns:
        Response object: The response from the server.
    """
    api_key = "126d8071"
    url = f"http://www.omdbapi.com/?apikey={api_key}&s=horror"
    response = requests.get(url)
    if response.status_code == 200:
        print("Response: ", response.json())
    else:
        print(f"Request failed with status code {response.status_code}")


# Print the sorted DataFrame for English movies
#print(english_movies_sorted)


