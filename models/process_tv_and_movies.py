import os
import json
import requests

import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Firebase Admin SDK imports for Storage access
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage


tmdb_api_key = os.getenv("TMDB_API_KEY")

# Force bucket name as requested; can still be overridden by FIREBASE_STORAGE_BUCKET
DEFAULT_STORAGE_BUCKET = "nightoutclient-7931e.firebasestorage.app"
movies_path = "movies_dataset.parquet"
tv_path = "tv_dataset.parquet"


def _load_service_account_project_id(path: str | None) -> str | None:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("project_id")
    except Exception:
        return None


def _infer_bucket_name() -> str:
    # Use explicit env if provided; otherwise use the fixed bucket name
    return os.getenv("FIREBASE_STORAGE_BUCKET", DEFAULT_STORAGE_BUCKET)


def _initialize_firebase_if_needed() -> None:
    if firebase_admin._apps:  # type: ignore[attr-defined]
        return
    service_account_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH")
    # Prefer explicit FIREBASE_PROJECT_ID, else fall back to GOOGLE_CLOUD_PROJECT (ADC) or SA json
    project_id = os.getenv("FIREBASE_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT") or _load_service_account_project_id(service_account_path)
    storage_bucket = os.getenv("FIREBASE_STORAGE_BUCKET", DEFAULT_STORAGE_BUCKET)

    options: dict[str, str] = {}
    if project_id:
        options["projectId"] = project_id
    if storage_bucket:
        options["storageBucket"] = storage_bucket

    if service_account_path and os.path.exists(service_account_path):
        cred = credentials.Certificate(service_account_path)
        firebase_admin.initialize_app(cred, options or None)
    else:
        # Application Default Credentials (Cloud Run recommended)
        firebase_admin.initialize_app(options=options or None)


def get_asset_df_from_storage(blob_path: str, asset_type: str) -> pd.DataFrame:
    # Ensure Firebase is initialized so storage has project/bucket context
    _initialize_firebase_if_needed()
    # Use the fixed bucket name (or env override)
    bucket_name = _infer_bucket_name()
    bucket = storage.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    if asset_type == 'movie':
        title = "original_title"
    else:
        title = "original_name"

    if not blob.exists():
        raise FileNotFoundError(f"Blob not found in bucket '{bucket.name}': {blob_path}")

    # Download to a temporary file and load with pandas
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as tmp_file:
        temp_path = tmp_file.name
    try:
        blob.download_to_filename(temp_path)
        movie_df = pd.read_parquet(temp_path, engine="pyarrow")
        movie_df = movie_df.drop_duplicates(subset=title, keep="first")
        return movie_df
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass


def get_asset_df(asset_type : str) -> pd.DataFrame:
    if asset_type == 'movies':
        file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets", "movies_dataset.parquet")
        title = "original_title"
    else:
        file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets", "tv_dataset.parquet")
        title = "original_name"
    asset_df = pd.read_parquet(file_path, engine="pyarrow")
    asset_df = asset_df.drop_duplicates(subset=title, keep='first')

    return asset_df


def genre_similarity(movies: set[int], asset_type: str) -> pd.DataFrame:
    """
    This function reads the Parquet File, processes genre_ids, and returns a DataFrame with crosstab operation.
    """
    movie_genre_df = get_asset_df(asset_type)

    # Explode the 'genre_ids' column to split lists into individual rows
    movie_genre_df = movie_genre_df.explode('genre_ids')

    # Convert genre_ids to integers if necessary (may currently be strings)
    movie_genre_df['genre_ids'] = movie_genre_df['genre_ids'].astype(float)

    # Create cross-tabulation between original_title and genre_ids
    movie_cross_table = pd.crosstab(movie_genre_df['id'], movie_genre_df['genre_ids'])

    # Select rows for movies

    jaccard_distances = pdist(movie_cross_table, metric='jaccard')
    square_jaccard_distances = squareform(jaccard_distances)
    jaccard_similarity_array = 1 - square_jaccard_distances
    distance_df = pd.DataFrame(jaccard_similarity_array, index=movie_cross_table.index, columns=movie_cross_table.index)

    # Get similarities for each movie in the set and average them
    valid_movies = [m for m in movies if m in distance_df.columns]

    if not valid_movies:
        raise ValueError("None of the provided movies found in dataset")

        # Average similarity across all input movies
    avg_similarity = distance_df[valid_movies].mean(axis=1)
    sorted_movie_df = avg_similarity.sort_values(ascending=False)

    return sorted_movie_df


def overview_similarity(movies: set[int], asset_type) -> pd.DataFrame:
    # Create vectorized data from the number of word occurences
    movies_df = get_asset_df(asset_type)
    vectorizer = TfidfVectorizer(min_df=2, max_df=0.7)
    vectorized_data = vectorizer.fit_transform(movies_df['overview'])

    movies_overview_df = pd.DataFrame(vectorized_data.toarray(), columns=vectorizer.get_feature_names_out())

    movies_overview_df.index = movies_df['id']

    # Calculate the cosine similarity
    cosine_similarity_array = cosine_similarity(movies_overview_df)

    cosine_similarity_df = pd.DataFrame(cosine_similarity_array, index=movies_overview_df.index,
                                        columns=movies_overview_df.index)

    # Get similarities for each movie in the set and average them
    valid_movies = [m for m in movies if m in cosine_similarity_df.columns]

    if not valid_movies:
        raise ValueError("None of the provided movies found in dataset")

    # Average similarity across all input movies
    avg_similarity = cosine_similarity_df[valid_movies].mean(axis=1)

    ordered_similarities = avg_similarity.sort_values(ascending=False)

    return ordered_similarities


def get_movie_detail(movie_id: int, asset_type: str):
    url = f"https://api.themoviedb.org/3/{asset_type}/{movie_id}?language=en-US&api_key={tmdb_api_key}"
    #url = f"https://api.themoviedb.org/3/{asset_type}/{movie_id}?language=en-US&api_key=2ed2b9d2e44bf6e9a70b687c134ed8f9"
    response = requests.get(url)
    if response.status_code == 200:
        # Parse JSON response
        data = response.json()
        return data
    else:
        print(f"Request failed with status code {response.json()}")


def find_similar_asset(assets: set[int], asset_type: str):
    if asset_type == 'movie':
        asset_df =  get_asset_df_from_storage(movies_path, asset_type)
        title = "original_title"
    else:
        asset_df = get_asset_df_from_storage(tv_path, asset_type)
        title = "original_name"
    asset_df.set_index('id', inplace=True)
    genre_similarity_df = genre_similarity(assets, asset_type)
    overview_similarity_df = overview_similarity(assets, asset_type)

    # Combine both similarities into one DataFrame
    combined_df = pd.DataFrame({
        'genre_sim': genre_similarity_df,
        'overview_sim': overview_similarity_df
    })

    # Calculate combined score (you can adjust weights)
    combined_df['combined_score'] = (combined_df['genre_sim'] * 0.5 +
                                     combined_df['overview_sim'] * 0.5)

    # Sort by combined score
    combined_df = combined_df.sort_values('combined_score', ascending=False)

    # Find first non-sequel/prequel movie
    for curr_asset_id in combined_df.index:
        # Skip if it's one of the input movies
        if curr_asset_id in assets:
            continue

        # Ensure the candidate exists in the DataFrame and language column is present
        if curr_asset_id not in asset_df.index:
            continue
        if 'original_language' in asset_df.columns:
            if asset_df.at[curr_asset_id, 'original_language'] != "en":
                continue

        # Check if it's a sequel/prequel of any of the input movies
        curr_asset_title = asset_df.loc[curr_asset_id, title]

        is_related = False
        for input_asset in assets:
            # ensure both index and column exist before accessing
            if input_asset in asset_df.index and title in asset_df.columns:
                asset_title = asset_df.at[input_asset, title]
                if asset_title and str(asset_title).strip():
                    if is_sequel_or_prequel(str(asset_title), curr_asset_title):
                        is_related = True
                        break

        if not is_related:
            print(f"Found similar movie series ID: {curr_asset_id}")
            print(f"Found similar movie: {curr_asset_title}")
            print(f"Genre similarity: {combined_df.loc[curr_asset_id, 'genre_sim']:.3f}")
            print(f"Overview similarity: {combined_df.loc[curr_asset_id, 'overview_sim']:.3f}")
            print(f"Combined score: {combined_df.loc[curr_asset_id, 'combined_score']:.3f}")

            asset_detail = get_movie_detail(curr_asset_id, asset_type)
            print(asset_detail)

            return asset_detail

    return None


def is_sequel_or_prequel(original_movie: str, candidate_movie: str) -> bool:
    """
    Check if candidate_movie is likely a sequel or prequel of original_movie
    """
    original_lower = original_movie.lower()
    candidate_lower = candidate_movie.lower()

    # Remove common sequel/prequel indicators to get base title
    sequel_prequel_indicators = [
        ' 2', ' 3', ' 4', ' 5', ' 6', ' 7', ' 8', ' 9',
        ' ii', ' iii', ' iv', ' v', ' vi', ' vii', ' viii',
        ' part', ' chapter',
        ': the', ' returns', ' reloaded', ' revolutions', ' rises',
        ' begins', ' origins', ' first', ' zero', ' episode',
        ' prequel', ' sequel', ' saga', ' chronicles'
    ]

    original_base = original_lower
    candidate_base = candidate_lower

    for indicator in sequel_prequel_indicators:
        original_base = original_base.split(indicator)[0]
        candidate_base = candidate_base.split(indicator)[0]

    # If base titles are very similar, likely a sequel or prequel
    return original_base in candidate_base or candidate_base in original_base


if __name__ == "__main__":
    #movies = {278, 238, 240, 424}
    shows = {94605, 1396, 246, 85077, 60625}
    find_similar_asset(shows, "tv")

