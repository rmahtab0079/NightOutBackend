import requests
import os
from queue import Queue
import pandas as pd
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import time

# Load environment from mounted secret path if provided, else from local .env
dotenv_path = os.getenv("DOTENV_PATH")
if dotenv_path and os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
else:
    load_dotenv()

BASE_IMAGE_URL = "https://image.tmdb.org/t/p/"
DEFAULT_IMAGE_SIZE = "w200"  # Customize based on required size (e.g., "original", "w300", etc.)


def write_tv_df():
    tmdb_api_key = os.getenv("TMDB_API_KEY")
    results_queue = Queue()

    # Use a thread pool with a max of 10 concurrent workers
    with ThreadPoolExecutor(max_workers=10) as executor:
        for i in range(1, 10000):
            executor.submit(get_movies_page, tmdb_api_key, i, results_queue)

    responses = []
    while not results_queue.empty():
        responses.append(results_queue.get())

    # Flatten pages and build a TMDB-like envelope expected by the client
    merged_results = []
    for r in responses:
        merged_results.extend(r.get("results", []))

    # Create a DataFrame from merged results and write to Parquet
    df = pd.DataFrame(merged_results)
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets", "tv_dataset.parquet")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False, engine="pyarrow")


def write_movies_df():
    tmdb_api_key = os.getenv("TMDB_API_KEY")
    results_queue = Queue()

    # Use a thread pool with a max of 10 concurrent workers
    with ThreadPoolExecutor(max_workers=10) as executor:
        for i in range(1, 10000):
            executor.submit(get_movies_page, tmdb_api_key, i, results_queue)

    responses = []
    while not results_queue.empty():
        responses.append(results_queue.get())

    # Flatten pages and build a TMDB-like envelope expected by the client
    merged_results = []
    for r in responses:
        merged_results.extend(r.get("results", []))

    # Create a DataFrame from merged results and write to Parquet
    df = pd.DataFrame(merged_results)
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets", "movies_dataset.parquet")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False, engine="pyarrow")


def read_movies_df():
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets", "movies_dataset.parquet")
    df = pd.read_parquet(output_path, engine="pyarrow")
    return print(len(df))


def get_tv_page(api_key, page, results_queue):
    # Small delay to avoid hitting API rate limits
    time.sleep(0.4)
    url = f"https://api.themoviedb.org/3/tv/top_rated?language=en-US&page={page}&api_key={api_key}"
    #url = f"https://api.themoviedb.org/3/discover/movie?include_adult=false&include_video=false&language=en-US&page=1&sort_by=popularity.desc&api_key={tmdb_api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        # Parse JSON response
        data = response.json()
        for movie in data.get("results", []):
            backdrop_path = movie.get("poster_path")
            if backdrop_path:
                # Construct the full backdrop image URL
                movie["poster_url"] = f"{BASE_IMAGE_URL}{DEFAULT_IMAGE_SIZE}{backdrop_path}"
        # Add the modified response to the queue
        results_queue.put(data)

    else:
        print(f"Request failed with status code {response.status_code}")


def get_movies_page(api_key, page, results_queue):
    # Small delay to avoid hitting API rate limits
    time.sleep(0.4)
    url = f"https://api.themoviedb.org/3/movie/top_rated?language=en-US&page={page}&api_key={api_key}"
    #url = f"https://api.themoviedb.org/3/discover/movie?include_adult=false&include_video=false&language=en-US&page=1&sort_by=popularity.desc&api_key={tmdb_api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        # Parse JSON response
        data = response.json()
        for movie in data.get("results", []):
            backdrop_path = movie.get("poster_path")
            if backdrop_path:
                # Construct the full backdrop image URL
                movie["poster_url"] = f"{BASE_IMAGE_URL}{DEFAULT_IMAGE_SIZE}{backdrop_path}"
        # Add the modified response to the queue
        results_queue.put(data)

    else:
        print(f"Request failed with status code {response.status_code}")


def get_movie_genres():
    genre_map = {}
    api_key = os.getenv("TMDB_API_KEY")
    url = f"https://api.themoviedb.org/3/genre/movie/list?language=en&api_key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        for genre in data.get("genres", []):
            genre_map[genre["id"]] = genre["name"]
    print(genre_map)
    return genre_map


def get_tv_genres():
    genre_map = {}
    api_key = os.getenv("TMDB_API_KEY")
    url = f"https://api.themoviedb.org/3/genre/tv/list?language=en&api_key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        for genre in data.get("genres", []):
            genre_map[genre["id"]] = genre["name"]
    print(genre_map)
    return genre_map


def movies_dataframe():
    """
    This function reads the Parquet File and returns a Pandas DataFrame.
    :return:
    """
    file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets", "movies_dataset.parquet")
    df = pd.read_parquet(file_path, engine="pyarrow")
    print(df.head(10))

def get_movies_


if __name__ == "__main__":
    movies_dataframe()