import os
import json
import requests
import wikipedia

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
movies_plots_path = "movies_plots.parquet"
tv_plots_path = "tv_plots.parquet"

# Cache for loaded plot DataFrames to avoid repeated downloads
_plots_cache: dict[str, pd.DataFrame] = {}


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


def get_plots_df_from_storage(blob_path: str) -> pd.DataFrame:
    """
    Load pre-computed Wikipedia plots from Firebase Storage.
    Results are cached in memory to avoid repeated downloads.
    
    Args:
        blob_path: Path to the plots parquet file in Firebase Storage
        
    Returns:
        DataFrame with columns: id, name, plot
    """
    global _plots_cache
    
    # Check cache first
    if blob_path in _plots_cache:
        return _plots_cache[blob_path]
    
    _initialize_firebase_if_needed()
    bucket_name = _infer_bucket_name()
    bucket = storage.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    
    if not blob.exists():
        raise FileNotFoundError(f"Plots file not found in bucket '{bucket.name}': {blob_path}")
    
    import tempfile
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as tmp_file:
        temp_path = tmp_file.name
    
    try:
        blob.download_to_filename(temp_path)
        plots_df = pd.read_parquet(temp_path, engine="pyarrow")
        
        # Cache for future use
        _plots_cache[blob_path] = plots_df
        
        print(f"Loaded {len(plots_df)} plots from {blob_path}")
        return plots_df
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass


def get_plots_df_local(asset_type: str) -> pd.DataFrame:
    """
    Load pre-computed Wikipedia plots from local datasets folder.
    Results are cached in memory.
    
    Args:
        asset_type: 'movies' or 'tv'
        
    Returns:
        DataFrame with columns: id, name, plot
    """
    global _plots_cache
    
    if asset_type == 'movies':
        file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets", "movies_plots.parquet")
    else:
        file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets", "tv_plots.parquet")
    
    # Check cache first
    if file_path in _plots_cache:
        return _plots_cache[file_path]
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Plots file not found: {file_path}")
    
    plots_df = pd.read_parquet(file_path, engine="pyarrow")
    
    # Cache for future use
    _plots_cache[file_path] = plots_df
    
    print(f"Loaded {len(plots_df)} plots from {file_path}")
    return plots_df


def get_plots_df(asset_type: str, read_from_local: bool = False) -> pd.DataFrame:
    """
    Get the plots DataFrame from either local storage or Firebase.
    
    Args:
        asset_type: 'movie' or 'tv' (will be normalized internally)
        read_from_local: If True, read from local datasets folder
        
    Returns:
        DataFrame with columns: id, name, plot
    """
    # Normalize asset_type
    asset_type_normalized = "movies" if asset_type == "movie" else asset_type
    
    if read_from_local:
        return get_plots_df_local(asset_type_normalized)
    else:
        blob_path = movies_plots_path if asset_type_normalized == "movies" else tv_plots_path
        return get_plots_df_from_storage(blob_path)


def clear_plots_cache() -> None:
    """Clear the in-memory plots cache."""
    global _plots_cache
    _plots_cache.clear()
    print("Plots cache cleared")


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

def get_movie_plot(movie_name: str) -> str:
    try:
        page = wikipedia.page(f"{movie_name} (film)")
        plot = page.section("Plot")
        return plot if plot else "Plot section not found"
    except wikipedia.exceptions.DisambiguationError as e:
        # Handle multiple matches - try the first one
        page = wikipedia.page(e.options[0])
        return page.section("Plot")
    except wikipedia.exceptions.PageError:
        return "Page not found"


def get_movie_ids_from_names(names: list[str]) -> set[int]:
    asset_df = get_asset_df("movies")
    filtered_df = asset_df[asset_df['original_title'].isin(names)]
    print(filtered_df)
    return set(filtered_df['id'].tolist())

def get_tv_ids_from_names(names: list[str]) -> set[int]:
    asset_df = get_asset_df("tv")
    filtered_df = asset_df[asset_df['original_name'].isin(names)]
    print(filtered_df)
    return set(filtered_df['id'].tolist())

def get_movie_names_from_ids(ids: set[int]) -> list[str]:
    asset_df = get_asset_df("movies")
    filtered_df = asset_df[asset_df['id'].isin(ids)]
    return filtered_df['original_title'].tolist()

def get_tv_names_from_ids(ids: set[int]) -> list[str]:
    asset_df = get_asset_df("tv")
    filtered_df = asset_df[asset_df['id'].isin(ids)]
    return filtered_df['original_name'].tolist()

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


def get_movie_detail(movie_id: int, asset_type: str, tmdb_api_key="2ed2b9d2e44bf6e9a70b687c134ed8f9"):
    url = f"https://api.themoviedb.org/3/{asset_type}/{movie_id}?language=en-US&api_key={tmdb_api_key}"
    #url = f"https://api.themoviedb.org/3/{asset_type}/{movie_id}?language=en-US&api_key=2ed2b9d2e44bf6e9a70b687c134ed8f9"
    response = requests.get(url)
    if response.status_code == 200:
        # Parse JSON response
        data = response.json()
        return data
    else:
        print(f"Request failed with status code {response.json()}")


def find_similar_asset(assets: set[int], asset_type: str, read_from_local: bool = False):
    if read_from_local:
        if asset_type == 'movie':
            asset_df = get_asset_df("movies")
            title = "original_title"
        else:
            asset_df = get_asset_df("tv")
            title = "original_name"
    else:
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
        # if curr_asset_id in assets:
        #     continue

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


def find_similar_asset_v2(assets: set[int], asset_type: str, read_from_local: bool = False, top_n: int = 5):
    """
    Find similar movies using genre similarity + pre-computed Wikipedia plot TF-IDF cosine similarity.
    
    This optimized version uses pre-computed Wikipedia plots from parquet files
    instead of fetching plots dynamically, making inference much faster.
    
    1. Get top 100 genre-similar movies
    2. Load pre-computed Wikipedia plots from parquet
    3. Create TF-IDF vectors from plots
    4. Compute cosine similarity to input movies
    5. Return top N most similar movies
    """
    # Get asset DataFrame
    if read_from_local:
        if asset_type == 'movie':
            asset_df = get_asset_df("movies")
            title = "original_title"
        else:
            asset_df = get_asset_df("tv")
            title = "original_name"
    else:
        if asset_type == 'movie':
            asset_df = get_asset_df_from_storage(movies_path, asset_type)
            title = "original_title"
        else:
            asset_df = get_asset_df_from_storage(tv_path, asset_type)
            title = "original_name"
    
    asset_df.set_index('id', inplace=True)
    
    # Normalize asset_type for functions that expect "movies"/"tv"
    asset_type_plural = "movies" if asset_type == "movie" else "tv"
    
    # Step 1: Get top 100 genre-similar movies
    genre_sim_df = genre_similarity(assets, asset_type_plural)
    top_100_candidates = genre_sim_df.head(100).index.tolist()
    
    # Remove input movies from candidates
    top_100_candidates = [m for m in top_100_candidates if m not in assets]
    
    print(f"Found {len(top_100_candidates)} genre-similar candidates")
    
    # Step 2: Load pre-computed Wikipedia plots
    print("Loading pre-computed Wikipedia plots...")
    try:
        plots_df = get_plots_df(asset_type, read_from_local)
    except FileNotFoundError as e:
        print(f"Warning: Pre-computed plots not found ({e}). Falling back to overview similarity.")
        # Fallback to overview-based similarity if plots not available
        return _find_similar_asset_fallback(assets, asset_df, asset_type, title, genre_sim_df, top_n)
    
    # Create a lookup dict for fast access: id -> plot
    plots_lookup = dict(zip(plots_df['id'], plots_df['plot']))
    print(f"Loaded {len(plots_lookup)} pre-computed plots")
    
    # Step 3: Get plots for input movies and candidates
    input_plots = {}
    for movie_id in assets:
        if movie_id in plots_lookup:
            input_plots[movie_id] = plots_lookup[movie_id]
    
    print(f"Found plots for {len(input_plots)}/{len(assets)} input movies")
    
    if not input_plots:
        print("Warning: No plots found for input movies. Falling back to overview similarity.")
        return _find_similar_asset_fallback(assets, asset_df, asset_type, title, genre_sim_df, top_n)
    
    candidate_plots = {}
    for movie_id in top_100_candidates:
        if movie_id in plots_lookup:
            candidate_plots[movie_id] = plots_lookup[movie_id]
    
    print(f"Found plots for {len(candidate_plots)}/{len(top_100_candidates)} candidates")
    
    if not candidate_plots:
        print("Warning: No plots found for candidates. Falling back to overview similarity.")
        return _find_similar_asset_fallback(assets, asset_df, asset_type, title, genre_sim_df, top_n)
    
    # Step 4: Create TF-IDF vectors
    all_movie_ids = list(input_plots.keys()) + list(candidate_plots.keys())
    all_plots = list(input_plots.values()) + list(candidate_plots.values())
    
    vectorizer = TfidfVectorizer(stop_words='english', min_df=1, max_df=0.95)
    tfidf_matrix = vectorizer.fit_transform(all_plots)
    
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        index=all_movie_ids,
        columns=vectorizer.get_feature_names_out()
    )
    
    # Step 5: Compute cosine similarity
    # Create user profile by averaging input movie TF-IDF vectors
    input_movie_ids = list(input_plots.keys())
    user_profile = tfidf_df.loc[input_movie_ids].mean(axis=0).values.reshape(1, -1)
    
    # Get candidate vectors
    candidate_ids = list(candidate_plots.keys())
    candidate_vectors = tfidf_df.loc[candidate_ids].values
    
    # Compute cosine similarity between user profile and all candidates
    similarities = cosine_similarity(user_profile, candidate_vectors)[0]
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'movie_id': candidate_ids,
        'similarity': similarities
    })
    results_df = results_df.sort_values('similarity', ascending=False)
    
    # Step 6: Filter out sequels/prequels and return top N
    results = []
    for _, row in results_df.iterrows():
        movie_id = row['movie_id']
        
        if movie_id not in asset_df.index:
            continue
            
        # Check language
        if 'original_language' in asset_df.columns:
            if asset_df.at[movie_id, 'original_language'] != "en":
                continue
        
        movie_title = asset_df.loc[movie_id, title]
        
        # Check if sequel/prequel of any input movie
        is_related = False
        for input_id in assets:
            if input_id in asset_df.index:
                input_title = asset_df.at[input_id, title]
                if input_title and str(input_title).strip():
                    if is_sequel_or_prequel(str(input_title), movie_title):
                        is_related = True
                        break
        
        if not is_related:
            movie_detail = get_movie_detail(movie_id, asset_type)
            results.append({
                'movie_id': movie_id,
                'title': movie_title,
                'similarity': row['similarity'],
                'detail': movie_detail
            })
            print(f"Found: {movie_title} (similarity: {row['similarity']:.3f})")
            
            if len(results) >= top_n:
                break

    return results


def _find_similar_asset_fallback(
    assets: set[int],
    asset_df: pd.DataFrame,
    asset_type: str,
    title: str,
    genre_sim_df: pd.Series,
    top_n: int
) -> list[dict]:
    """
    Fallback method using overview similarity when Wikipedia plots are not available.
    Uses the existing overview column from the main dataset instead of Wikipedia plots.
    """
    print("Using fallback: overview-based similarity")
    
    asset_type_plural = "movies" if asset_type == "movie" else "tv"
    overview_sim_df = overview_similarity(assets, asset_type_plural)
    
    # Combine genre and overview similarity
    combined_df = pd.DataFrame({
        'genre_sim': genre_sim_df,
        'overview_sim': overview_sim_df
    })
    combined_df['combined_score'] = (combined_df['genre_sim'] * 0.5 + 
                                      combined_df['overview_sim'] * 0.5)
    combined_df = combined_df.sort_values('combined_score', ascending=False)
    
    results = []
    for movie_id in combined_df.index:
        if movie_id in assets:
            continue
            
        if movie_id not in asset_df.index:
            continue
        
        if 'original_language' in asset_df.columns:
            if asset_df.at[movie_id, 'original_language'] != "en":
                continue
        
        movie_title = asset_df.loc[movie_id, title]
        
        is_related = False
        for input_id in assets:
            if input_id in asset_df.index:
                input_title = asset_df.at[input_id, title]
                if input_title and str(input_title).strip():
                    if is_sequel_or_prequel(str(input_title), movie_title):
                        is_related = True
                        break
        
        if not is_related:
            movie_detail = get_movie_detail(movie_id, asset_type)
            results.append({
                'movie_id': movie_id,
                'title': movie_title,
                'similarity': combined_df.loc[movie_id, 'combined_score'],
                'detail': movie_detail
            })
            print(f"Found (fallback): {movie_title} (score: {combined_df.loc[movie_id, 'combined_score']:.3f})")
            
            if len(results) >= top_n:
                break
    
    return results


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

