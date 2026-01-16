"""
Cache and Database Layer for Similar Asset Recommendations.

This module implements a multi-layer caching strategy:
1. In-memory cache with TTL (1 hour default)
2. Firestore database for persistent storage
3. API fallback via find_similar_asset_v2

Flow:
- Check in-memory cache first
- If not in cache, check Firestore database
- If not in database, fetch from API (find_similar_asset_v2)
- Store results in both cache and database
"""

import hashlib
import json
import time
from datetime import datetime
from typing import Optional

import firebase_admin
from firebase_admin import credentials, firestore

from models.process_tv_and_movies import find_similar_asset_v2


# Cache TTL in seconds (1 hour)
CACHE_TTL_SECONDS = 3600

# In-memory cache: key -> {"data": ..., "timestamp": ...}
_memory_cache: dict[str, dict] = {}

# Firestore client (initialized lazily)
_firestore_db = None

# Collection name for storing recommendations
RECOMMENDATIONS_COLLECTION = "similar_asset_recommendations"


def _get_firestore_client():
    """Get or initialize Firestore client."""
    global _firestore_db
    if _firestore_db is not None:
        return _firestore_db

    # Check if Firebase is already initialized
    if firebase_admin._apps:
        _firestore_db = firestore.client()
        return _firestore_db

    # Try to initialize Firebase
    import os
    service_account_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH", "serviceAccount.json")
    project_id = os.getenv("FIREBASE_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")

    try:
        if service_account_path and os.path.exists(service_account_path):
            cred = credentials.Certificate(service_account_path)
            options = {"projectId": project_id} if project_id else None
            firebase_admin.initialize_app(cred, options)
        else:
            # Use Application Default Credentials
            options = {"projectId": project_id} if project_id else None
            firebase_admin.initialize_app(options=options)
        
        _firestore_db = firestore.client()
        return _firestore_db
    except Exception as e:
        print(f"Failed to initialize Firestore: {e}")
        return None


def _generate_cache_key(
    asset_ids: set[int],
    asset_type: str,
    genres: list[int] | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    top_n: int = 5,
    excluded_ids: set[int] | None = None
) -> str:
    """
    Generate a unique cache key based on input parameters.
    Uses a hash of sorted asset IDs and filter parameters.
    """
    sorted_ids = sorted(asset_ids)
    sorted_genres = sorted(genres) if genres else []
    sorted_excluded = sorted(excluded_ids) if excluded_ids else []
    
    key_data = {
        "asset_ids": sorted_ids,
        "asset_type": asset_type,
        "genres": sorted_genres,
        "start_year": start_year,
        "end_year": end_year,
        "top_n": top_n,
        "excluded_ids": sorted_excluded
    }
    
    key_string = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_string.encode()).hexdigest()


def _is_cache_valid(cache_entry: dict) -> bool:
    """Check if a cache entry is still valid based on TTL."""
    if not cache_entry or "timestamp" not in cache_entry:
        return False
    
    age = time.time() - cache_entry["timestamp"]
    return age < CACHE_TTL_SECONDS


def _get_from_memory_cache(cache_key: str) -> Optional[list[dict]]:
    """Retrieve data from in-memory cache if valid."""
    entry = _memory_cache.get(cache_key)
    if entry and _is_cache_valid(entry):
        print(f"Cache hit (memory): {cache_key[:16]}...")
        return entry["data"]
    return None


def _set_memory_cache(cache_key: str, data: list[dict]) -> None:
    """Store data in in-memory cache with current timestamp."""
    _memory_cache[cache_key] = {
        "data": data,
        "timestamp": time.time()
    }
    print(f"Cached in memory: {cache_key[:16]}...")


def _get_from_firestore(cache_key: str) -> Optional[list[dict]]:
    """Retrieve data from Firestore if valid."""
    db = _get_firestore_client()
    if db is None:
        return None
    
    try:
        doc_ref = db.collection(RECOMMENDATIONS_COLLECTION).document(cache_key)
        doc = doc_ref.get()
        
        if not doc.exists:
            return None
        
        doc_data = doc.to_dict()
        
        # Check if the cached data is still valid
        cached_at = doc_data.get("cached_at")
        if cached_at:
            # Convert Firestore timestamp to epoch seconds
            if hasattr(cached_at, "timestamp"):
                cached_timestamp = cached_at.timestamp()
            else:
                cached_timestamp = cached_at
            
            if time.time() - cached_timestamp < CACHE_TTL_SECONDS:
                print(f"Cache hit (Firestore): {cache_key[:16]}...")
                return doc_data.get("results", [])
        
        return None
    except Exception as e:
        print(f"Firestore read error: {e}")
        return None


def _save_to_firestore(
    cache_key: str,
    results: list[dict],
    asset_ids: set[int],
    asset_type: str,
    genres: list[int] | None,
    start_year: int | None,
    end_year: int | None
) -> None:
    """Save results to Firestore for persistent caching."""
    db = _get_firestore_client()
    if db is None:
        return
    
    try:
        doc_ref = db.collection(RECOMMENDATIONS_COLLECTION).document(cache_key)
        doc_data = {
            "results": results,
            "asset_ids": list(asset_ids),
            "asset_type": asset_type,
            "genres": genres or [],
            "start_year": start_year,
            "end_year": end_year,
            "cached_at": firestore.SERVER_TIMESTAMP,
            "created_at_iso": datetime.utcnow().isoformat() + "Z"
        }
        doc_ref.set(doc_data)
        print(f"Saved to Firestore: {cache_key[:16]}...")
    except Exception as e:
        print(f"Firestore write error: {e}")


def _filter_results_by_params(
    results: list[dict],
    genres: list[int] | None,
    start_year: int | None,
    end_year: int | None,
    asset_type: str,
    excluded_ids: set[int] | None = None
) -> list[dict]:
    """
    Filter API results by genre IDs, release year range, and excluded IDs.
    
    Args:
        results: List of recommendation results with 'detail' containing TMDB data
        genres: List of genre IDs to filter by (empty = all genres)
        start_year: Minimum release year (inclusive)
        end_year: Maximum release year (inclusive)
        asset_type: 'movie' or 'tv'
        excluded_ids: Set of asset IDs to exclude (e.g., disliked items)
    """
    if not genres and not start_year and not end_year and not excluded_ids:
        return results
    
    date_field = "release_date" if asset_type == "movie" else "first_air_date"
    filtered = []
    
    for result in results:
        # Exclude by ID (disliked items)
        movie_id = result.get("movie_id")
        if excluded_ids and movie_id in excluded_ids:
            continue
        
        detail = result.get("detail", {})
        
        # Genre filter
        if genres:
            result_genres = detail.get("genres", [])
            result_genre_ids = [g.get("id") for g in result_genres if isinstance(g, dict)]
            if not any(g in genres for g in result_genre_ids):
                continue
        
        # Year filter
        if start_year or end_year:
            date_str = detail.get(date_field, "")
            if date_str:
                try:
                    year = int(date_str.split("-")[0])
                    if start_year and year < start_year:
                        continue
                    if end_year and year > end_year:
                        continue
                except (ValueError, IndexError):
                    pass
        
        filtered.append(result)
    
    return filtered


def get_similar_assets(
    asset_ids: set[int],
    asset_type: str,
    genres: list[int] | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    top_n: int = 5,
    read_from_local: bool = False,
    bypass_cache: bool = False,
    excluded_ids: set[int] | None = None
) -> list[dict]:
    """
    Get similar assets with caching support.
    
    Implements the caching flow:
    1. Check in-memory cache first
    2. If not in cache, check Firestore database
    3. If not in database, fetch from API (find_similar_asset_v2)
    4. Store results in both cache and database
    
    Args:
        asset_ids: Set of asset IDs (movie or TV show IDs)
        asset_type: 'movie' or 'tv'
        genres: List of genre IDs to filter by (None = all genres)
        start_year: Minimum release year (inclusive)
        end_year: Maximum release year (inclusive)
        top_n: Number of recommendations to return
        read_from_local: If True, read from local parquet files instead of Firebase Storage
        bypass_cache: If True, skip cache and fetch fresh results
        excluded_ids: Set of asset IDs to exclude from results (e.g., disliked items)
    
    Returns:
        List of similar asset recommendations with details
    """
    # Generate cache key (includes excluded_ids for proper cache isolation)
    cache_key = _generate_cache_key(
        asset_ids=asset_ids,
        asset_type=asset_type,
        genres=genres,
        start_year=start_year,
        end_year=end_year,
        top_n=top_n,
        excluded_ids=excluded_ids
    )
    
    if not bypass_cache:
        # Step 1: Check in-memory cache
        cached_results = _get_from_memory_cache(cache_key)
        if cached_results is not None:
            return cached_results
        
        # Step 2: Check Firestore database
        db_results = _get_from_firestore(cache_key)
        if db_results is not None:
            # Store in memory cache for faster subsequent access
            _set_memory_cache(cache_key, db_results)
            return db_results
    
    # Step 3: Fetch from API (find_similar_asset_v2)
    print("Cache miss - fetching from API...")
    
    # Request more results than needed to account for filtering
    # Increase multiplier when excluded_ids is present to ensure we have enough after filtering
    has_filters = genres or start_year or end_year or excluded_ids
    fetch_count = max(top_n * 5, 20) if has_filters else max(top_n * 2, 10)
    
    try:
        api_results = find_similar_asset_v2(
            assets=asset_ids,
            asset_type=asset_type,
            read_from_local=read_from_local,
            top_n=fetch_count
        )
    except Exception as e:
        print(f"API fetch error: {e}")
        raise
    
    # Apply genre, year, and exclusion filters
    filtered_results = _filter_results_by_params(
        results=api_results,
        genres=genres,
        start_year=start_year,
        end_year=end_year,
        asset_type=asset_type,
        excluded_ids=excluded_ids
    )
    
    # Limit to requested top_n
    final_results = filtered_results[:top_n]
    
    # Step 4: Store in both caches
    _set_memory_cache(cache_key, final_results)
    _save_to_firestore(
        cache_key=cache_key,
        results=final_results,
        asset_ids=asset_ids,
        asset_type=asset_type,
        genres=genres,
        start_year=start_year,
        end_year=end_year
    )
    
    return final_results


def clear_memory_cache() -> int:
    """Clear all entries from in-memory cache. Returns number of entries cleared."""
    global _memory_cache
    count = len(_memory_cache)
    _memory_cache = {}
    print(f"Cleared {count} entries from memory cache")
    return count


def clear_expired_firestore_cache() -> int:
    """
    Remove expired entries from Firestore cache.
    Returns number of documents deleted.
    
    Note: This should be called periodically (e.g., via a scheduled job)
    to clean up stale cache entries.
    """
    db = _get_firestore_client()
    if db is None:
        return 0
    
    deleted_count = 0
    cutoff_time = time.time() - CACHE_TTL_SECONDS
    
    try:
        collection_ref = db.collection(RECOMMENDATIONS_COLLECTION)
        docs = collection_ref.stream()
        
        for doc in docs:
            doc_data = doc.to_dict()
            cached_at = doc_data.get("cached_at")
            
            if cached_at:
                if hasattr(cached_at, "timestamp"):
                    cached_timestamp = cached_at.timestamp()
                else:
                    cached_timestamp = cached_at
                
                if cached_timestamp < cutoff_time:
                    doc.reference.delete()
                    deleted_count += 1
        
        print(f"Deleted {deleted_count} expired entries from Firestore")
        return deleted_count
    except Exception as e:
        print(f"Error clearing Firestore cache: {e}")
        return deleted_count


def get_cache_stats() -> dict:
    """Get statistics about the cache state."""
    memory_count = len(_memory_cache)
    memory_valid = sum(1 for entry in _memory_cache.values() if _is_cache_valid(entry))
    
    db = _get_firestore_client()
    firestore_count = 0
    
    if db is not None:
        try:
            collection_ref = db.collection(RECOMMENDATIONS_COLLECTION)
            firestore_count = len(list(collection_ref.stream()))
        except Exception:
            pass
    
    return {
        "memory_cache_total": memory_count,
        "memory_cache_valid": memory_valid,
        "firestore_cache_total": firestore_count,
        "cache_ttl_seconds": CACHE_TTL_SECONDS
    }
