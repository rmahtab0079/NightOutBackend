import pandas as pd
import requests
import os

file_path = "./models/movies_dataset.csv"

# Load the dataset
df = pd.read_csv(file_path)

# Filter movies with more than 1000 ratings
filtered_df = df[df["vote_count"] > 1000]

# Group by original language
grouped_by_language = filtered_df.groupby("original_language")

# Create a DataFrame for English movies and sort by vote average
english_movies_sorted = (
    grouped_by_language.get_group("en")
    .sort_values(by="vote_average", ascending=False)
)

def get_movies_tmdb():
    tmdb_api_key = os.getenv("TMDB_API_KEY")
    url = f"https://api.themoviedb.org/3/discover/movie?include_adult=false&include_video=false&language=en-US&page=1&sort_by=popularity.desc&api_key={tmdb_api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        print("Response: ", response.json())
    else:
        print(f"Request failed with status code {response.status_code}")


def get_movies():
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

