#!/usr/bin/env python3
"""
Generate watch provider parquet files and upload to Firebase Storage.

This script fetches streaming provider information from TMDB for all movies
and TV shows, then stores them in parquet files for quick retrieval.

Usage:
    python scripts/generate_and_upload_watch_providers.py --type movies
    python scripts/generate_and_upload_watch_providers.py --type tv
    python scripts/generate_and_upload_watch_providers.py --type all
"""

import os
import sys
import argparse

# Add repo root to path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)

# Load environment variables from .env file
from dotenv import load_dotenv
env_path = os.path.join(repo_root, ".env")
load_dotenv(env_path)

# Ensure service account path is absolute if it's relative
sa_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH")
if sa_path and not os.path.isabs(sa_path):
    os.environ["FIREBASE_SERVICE_ACCOUNT_PATH"] = os.path.join(repo_root, sa_path)

from service.watch_providers_job import generate_and_upload_providers


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and upload watch provider parquets"
    )
    parser.add_argument(
        "--type",
        choices=["movies", "tv", "all"],
        default="all",
        help="Type of assets to process",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Maximum number of items to process (for testing)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets"),
        help="Directory to save local parquet files",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Read datasets from local parquet files instead of Firebase Storage",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of concurrent workers for API requests",
    )
    args = parser.parse_args()

    types = ["movies", "tv"] if args.type == "all" else [args.type]
    
    for asset_type in types:
        print(f"\n{'='*60}")
        print(f"Processing {asset_type}...")
        print(f"{'='*60}")
        
        result = generate_and_upload_providers(
            asset_type=asset_type,
            output_dir=args.output_dir,
            max_items=args.max_items,
            read_from_storage=not args.local,
            max_workers=args.workers,
        )
        
        print(f"\nCompleted: {result['count']} {asset_type} with providers")
        print(f"Uploaded to: {result['blob_name']}")


if __name__ == "__main__":
    main()
