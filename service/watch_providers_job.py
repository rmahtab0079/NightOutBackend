"""
Fetch watch provider data from TMDB and create parquet files for movies and TV shows.

Uses TMDB's Watch Providers API:
- Movies: https://api.themoviedb.org/3/movie/{movie_id}/watch/providers
- TV: https://api.themoviedb.org/3/tv/{series_id}/watch/providers
"""

import os
import time
import json
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

from firebase_admin import storage

from models.process_tv_and_movies import (
    _initialize_firebase_if_needed,
    _infer_bucket_name,
    get_asset_df_from_storage,
    movies_path,
    tv_path,
)


# Rate limiting
REQUESTS_PER_SECOND = float(os.getenv("TMDB_REQUESTS_PER_SECOND", "40"))
REQUEST_DELAY = 1.0 / max(REQUESTS_PER_SECOND, 0.1)

# TMDB image base URL for provider logos
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/original"

# Map common provider names to standardized names (for matching with user preferences)
PROVIDER_NAME_MAPPING = {
    "Netflix": "Netflix",
    "Amazon Prime Video": "Amazon Prime Video",
    "Amazon Video": "Amazon Prime Video",
    "Disney Plus": "Disney+",
    "Disney+": "Disney+",
    "Hulu": "Hulu",
    "Max": "HBO Max",
    "HBO Max": "HBO Max",
    "Apple TV Plus": "Apple TV+",
    "Apple TV+": "Apple TV+",
    "Paramount Plus": "Paramount+",
    "Paramount+": "Paramount+",
    "Peacock": "Peacock",
    "Peacock Premium": "Peacock",
    "Crunchyroll": "Crunchyroll",
    "YouTube Premium": "YouTube Premium",
    "YouTube": "YouTube Premium",
}


def get_tmdb_api_key() -> str:
    """Get TMDB API key from environment."""
    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        raise ValueError("TMDB_API_KEY environment variable not set")
    return api_key


def fetch_watch_providers(
    asset_id: int,
    asset_type: str,  # 'movie' or 'tv'
    api_key: str,
    region: str = "US",
) -> Optional[Dict[str, Any]]:
    """
    Fetch watch providers for a movie or TV show from TMDB.
    
    Returns a dict with provider info or None if not available.
    """
    try:
        url = f"https://api.themoviedb.org/3/{asset_type}/{asset_id}/watch/providers"
        params = {"api_key": api_key}
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            return None
        
        data = response.json()
        results = data.get("results", {})
        
        # Get US providers (or specified region)
        region_data = results.get(region, {})
        
        providers = []
        
        # Combine flatrate (subscription), rent, and buy options
        for provider_type in ["flatrate", "rent", "buy", "free"]:
            provider_list = region_data.get(provider_type, [])
            for p in provider_list:
                provider_name = p.get("provider_name", "")
                # Normalize provider name
                normalized_name = PROVIDER_NAME_MAPPING.get(provider_name, provider_name)
                
                provider_info = {
                    "provider_id": p.get("provider_id"),
                    "provider_name": normalized_name,
                    "original_name": provider_name,
                    "logo_path": p.get("logo_path"),
                    "display_priority": p.get("display_priority", 999),
                    "provider_type": provider_type,
                }
                
                # Avoid duplicates
                if not any(
                    existing["provider_id"] == provider_info["provider_id"]
                    for existing in providers
                ):
                    providers.append(provider_info)
        
        # Sort by display priority
        providers.sort(key=lambda x: x["display_priority"])
        
        return {
            "id": asset_id,
            "providers": providers,
            "tmdb_link": region_data.get("link"),
        }
        
    except Exception as e:
        print(f"Error fetching providers for {asset_type} {asset_id}: {e}")
        return None


def fetch_provider_with_delay(
    asset_id: int,
    asset_type: str,
    api_key: str,
) -> Optional[Dict[str, Any]]:
    """Fetch provider with rate limiting delay."""
    time.sleep(REQUEST_DELAY)
    return fetch_watch_providers(asset_id, asset_type, api_key)


def generate_providers_dataframe(
    asset_type: str,  # 'movies' or 'tv'
    max_items: Optional[int] = None,
    read_from_storage: bool = True,
    max_workers: int = 10,
) -> pd.DataFrame:
    """
    Generate a DataFrame with watch provider information for all assets.
    """
    is_movie = asset_type == "movies"
    tmdb_asset_type = "movie" if is_movie else "tv"
    
    api_key = get_tmdb_api_key()
    
    # Load asset data
    if read_from_storage:
        blob_path = movies_path if is_movie else tv_path
        df = get_asset_df_from_storage(blob_path, "movie" if is_movie else "tv")
    else:
        from models.process_tv_and_movies import get_asset_df
        df = get_asset_df("movies" if is_movie else "tv")
    
    # Filter English content
    if "original_language" in df.columns:
        df = df[df["original_language"] == "en"]
    
    if max_items:
        df = df.head(max_items)
    
    asset_ids = df["id"].tolist()
    total = len(asset_ids)
    print(f"Fetching watch providers for {total} {asset_type}...")
    
    results: List[Dict[str, Any]] = []
    
    # Use thread pool for concurrent fetching
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_id = {
            executor.submit(
                fetch_watch_providers,
                asset_id,
                tmdb_asset_type,
                api_key,
            ): asset_id
            for asset_id in asset_ids
        }
        
        completed = 0
        for future in as_completed(future_to_id):
            asset_id = future_to_id[future]
            completed += 1
            
            try:
                result = future.result()
                if result and result.get("providers"):
                    results.append(result)
            except Exception as e:
                print(f"Error processing {asset_id}: {e}")
            
            if completed % 100 == 0 or completed == total:
                print(f"Progress: {completed}/{total} ({len(results)} with providers)")
    
    # Convert to DataFrame
    # Store providers as JSON string for parquet compatibility
    records = []
    for r in results:
        records.append({
            "id": r["id"],
            "providers_json": json.dumps(r["providers"]),
            "tmdb_link": r.get("tmdb_link"),
            "provider_count": len(r["providers"]),
        })
    
    providers_df = pd.DataFrame(records)
    print(f"Generated provider data for {len(providers_df)} {asset_type}")
    
    return providers_df


def upload_to_firebase(local_path: str, blob_name: str) -> None:
    """Upload a file to Firebase Storage."""
    _initialize_firebase_if_needed()
    bucket_name = _infer_bucket_name()
    bucket = storage.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    print(f"Uploaded {local_path} to gs://{bucket_name}/{blob_name}")


def generate_and_upload_providers(
    asset_type: str,  # 'movies' or 'tv'
    output_dir: str,
    max_items: Optional[int] = None,
    read_from_storage: bool = True,
    max_workers: int = 10,
) -> Dict[str, Any]:
    """
    Generate watch provider parquet and upload to Firebase Storage.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    providers_df = generate_providers_dataframe(
        asset_type=asset_type,
        max_items=max_items,
        read_from_storage=read_from_storage,
        max_workers=max_workers,
    )
    
    file_name = f"{asset_type}_providers.parquet"
    local_path = os.path.join(output_dir, file_name)
    providers_df.to_parquet(local_path, engine="pyarrow", index=False)
    
    upload_to_firebase(local_path, file_name)
    
    return {
        "asset_type": asset_type,
        "count": len(providers_df),
        "local_path": local_path,
        "blob_name": file_name,
    }


# Cache for loaded provider DataFrames
_providers_cache: Dict[str, pd.DataFrame] = {}


def get_providers_df(
    asset_type: str,  # 'movie' or 'tv'
    read_from_local: bool = False,
) -> pd.DataFrame:
    """
    Load providers DataFrame from Firebase Storage or local file.
    """
    cache_key = f"{asset_type}_providers"
    if cache_key in _providers_cache:
        return _providers_cache[cache_key]
    
    file_name = f"{'movies' if asset_type == 'movie' else 'tv'}_providers.parquet"
    
    if read_from_local:
        local_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "datasets",
            file_name,
        )
        df = pd.read_parquet(local_path, engine="pyarrow")
    else:
        _initialize_firebase_if_needed()
        bucket_name = _infer_bucket_name()
        bucket = storage.bucket(bucket_name)
        blob = bucket.blob(file_name)
        
        if not blob.exists():
            print(f"Provider parquet not found: {file_name}")
            return pd.DataFrame()
        
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as tmp:
            temp_path = tmp.name
        
        try:
            blob.download_to_filename(temp_path)
            df = pd.read_parquet(temp_path, engine="pyarrow")
        finally:
            try:
                os.remove(temp_path)
            except Exception:
                pass
    
    _providers_cache[cache_key] = df
    return df


def get_providers_for_asset(
    asset_id: int,
    asset_type: str,  # 'movie' or 'tv'
    read_from_local: bool = False,
) -> List[Dict[str, Any]]:
    """
    Get watch providers for a specific asset.
    """
    try:
        df = get_providers_df(asset_type, read_from_local)
        if df.empty:
            return []
        
        row = df[df["id"] == asset_id]
        if row.empty:
            return []
        
        providers_json = row.iloc[0]["providers_json"]
        return json.loads(providers_json)
    except Exception as e:
        print(f"Error getting providers for {asset_type} {asset_id}: {e}")
        return []
