import pandas as pd
import requests
import os
import threading
from queue import Queue


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
DEFAULT_IMAGE_SIZE = "w200"  # Customize based on required size (e.g., "original", "w300", etc.)


def get_movies():
    tmdb_api_key = os.getenv("TMDB_API_KEY")
    results_queue = Queue()

    threads = []
    for i in range(1, 100):
        t = threading.Thread(target=get_movies_page, args=(tmdb_api_key, i, results_queue))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    responses = []
    while not results_queue.empty():
        responses.append(results_queue.get())

    # Flatten pages and build a TMDB-like envelope expected by the client
    page = 1
    total_pages = len(responses)
    total_results = sum(len(r.get("results", [])) for r in responses)
    merged_results = []
    for r in responses:
        merged_results.extend(r.get("results", []))

    return {
        "page": page,
        "total_pages": total_pages,
        "total_results": total_results,
        "results": merged_results,
    }


def get_movies_page(api_key, page, results_queue):
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


