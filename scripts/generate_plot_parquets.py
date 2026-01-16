"""
Script to generate Wikipedia plot parquet files for movies and TV shows.

This script:
1. Reads existing movies/TV datasets
2. Fetches Wikipedia plots for each item (with rate limiting and error handling)
3. Saves parquet files with columns: id, name, plot
4. Optionally uploads to Firebase Storage

Usage:
    python scripts/generate_plot_parquets.py --type movies
    python scripts/generate_plot_parquets.py --type tv
    python scripts/generate_plot_parquets.py --type all --upload
"""

import os
import sys
import time
import argparse
import pandas as pd
import wikipedia
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.process_tv_and_movies import (
    get_asset_df,
    _initialize_firebase_if_needed,
    _infer_bucket_name,
    DEFAULT_STORAGE_BUCKET
)
from firebase_admin import storage


# Rate limiting settings
REQUESTS_PER_SECOND = 2
REQUEST_DELAY = 1.0 / REQUESTS_PER_SECOND

# Output paths
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets")
MOVIES_PLOTS_PATH = os.path.join(OUTPUT_DIR, "movies_plots.parquet")
TV_PLOTS_PATH = os.path.join(OUTPUT_DIR, "tv_plots.parquet")


def get_wikipedia_plot(name: str, is_movie: bool = True) -> str | None:
    """
    Fetch Wikipedia plot for a movie or TV show.
    Returns the plot text or None if not found.
    """
    try:
        # Try with (film) or (TV series) suffix first
        suffix = "(film)" if is_movie else "(TV series)"
        try:
            page = wikipedia.page(f"{name} {suffix}", auto_suggest=False)
        except (wikipedia.exceptions.PageError, wikipedia.exceptions.DisambiguationError):
            # Try without suffix
            try:
                page = wikipedia.page(name, auto_suggest=False)
            except wikipedia.exceptions.DisambiguationError as e:
                # Try first option from disambiguation
                if e.options:
                    try:
                        page = wikipedia.page(e.options[0], auto_suggest=False)
                    except Exception:
                        return None
                else:
                    return None
            except wikipedia.exceptions.PageError:
                return None
        
        # Try to get the Plot section
        plot = page.section("Plot")
        if plot and len(plot.strip()) > 50:  # Ensure plot has meaningful content
            return plot.strip()
        
        # Fallback: try Synopsis or Summary
        for section_name in ["Synopsis", "Summary", "Plot summary", "Premise"]:
            section = page.section(section_name)
            if section and len(section.strip()) > 50:
                return section.strip()
        
        # Last resort: use the summary (first paragraph)
        if page.summary and len(page.summary.strip()) > 100:
            return page.summary.strip()
        
        return None
        
    except Exception as e:
        print(f"  Error fetching plot for '{name}': {e}")
        return None


def fetch_plot_with_delay(item: dict, is_movie: bool) -> dict:
    """Fetch plot with rate limiting."""
    time.sleep(REQUEST_DELAY)
    plot = get_wikipedia_plot(item['name'], is_movie)
    return {
        'id': item['id'],
        'name': item['name'],
        'plot': plot
    }


def generate_plots_dataframe(asset_type: str, max_items: int | None = None) -> pd.DataFrame:
    """
    Generate a DataFrame with id, name, plot for all items of the given type.
    
    Args:
        asset_type: 'movies' or 'tv'
        max_items: Maximum number of items to process (for testing)
    
    Returns:
        DataFrame with columns: id, name, plot
    """
    print(f"\n{'='*60}")
    print(f"Generating plots for: {asset_type}")
    print(f"{'='*60}")
    
    # Load the dataset
    is_movie = asset_type == 'movies'
    title_col = 'original_title' if is_movie else 'original_name'
    
    df = get_asset_df(asset_type)
    print(f"Loaded {len(df)} items from {asset_type} dataset")
    
    # Filter to English only for better Wikipedia coverage
    if 'original_language' in df.columns:
        df = df[df['original_language'] == 'en']
        print(f"Filtered to {len(df)} English items")
    
    # Limit items if specified
    if max_items:
        df = df.head(max_items)
        print(f"Limited to {len(df)} items for testing")
    
    # Prepare items list
    items = [
        {'id': row['id'], 'name': row[title_col]}
        for _, row in df.iterrows()
        if pd.notna(row[title_col]) and str(row[title_col]).strip()
    ]
    print(f"Processing {len(items)} items with valid names")
    
    # Fetch plots with progress bar
    results = []
    successful = 0
    failed = 0
    
    print("\nFetching Wikipedia plots (this may take a while)...")
    
    for item in tqdm(items, desc=f"Fetching {asset_type} plots"):
        result = fetch_plot_with_delay(item, is_movie)
        results.append(result)
        
        if result['plot']:
            successful += 1
        else:
            failed += 1
    
    print(f"\nResults: {successful} successful, {failed} failed")
    
    # Create DataFrame
    plots_df = pd.DataFrame(results)
    
    # Filter out items without plots
    plots_df = plots_df[plots_df['plot'].notna()]
    print(f"Final dataset: {len(plots_df)} items with plots")
    
    return plots_df


def save_parquet(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to parquet file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, engine='pyarrow', index=False)
    print(f"Saved to: {path}")
    print(f"File size: {os.path.getsize(path) / 1024 / 1024:.2f} MB")


def upload_to_firebase(local_path: str, blob_name: str) -> None:
    """Upload a file to Firebase Storage."""
    _initialize_firebase_if_needed()
    bucket_name = _infer_bucket_name()
    bucket = storage.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    print(f"Uploading {local_path} to gs://{bucket_name}/{blob_name}...")
    blob.upload_from_filename(local_path)
    print(f"Upload complete!")


def main():
    parser = argparse.ArgumentParser(description='Generate Wikipedia plot parquet files')
    parser.add_argument('--type', choices=['movies', 'tv', 'all'], default='all',
                        help='Type of content to process')
    parser.add_argument('--max-items', type=int, default=None,
                        help='Maximum items to process (for testing)')
    parser.add_argument('--upload', action='store_true',
                        help='Upload to Firebase Storage after generation')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR,
                        help='Output directory for parquet files')
    
    args = parser.parse_args()
    
    types_to_process = ['movies', 'tv'] if args.type == 'all' else [args.type]
    
    for asset_type in types_to_process:
        # Generate plots DataFrame
        plots_df = generate_plots_dataframe(asset_type, args.max_items)
        
        # Save locally
        if asset_type == 'movies':
            local_path = os.path.join(args.output_dir, "movies_plots.parquet")
            blob_name = "movies_plots.parquet"
        else:
            local_path = os.path.join(args.output_dir, "tv_plots.parquet")
            blob_name = "tv_plots.parquet"
        
        save_parquet(plots_df, local_path)
        
        # Upload to Firebase if requested
        if args.upload:
            upload_to_firebase(local_path, blob_name)
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == "__main__":
    main()
