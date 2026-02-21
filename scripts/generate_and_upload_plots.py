#!/usr/bin/env python3
"""
Generate Wikipedia plot parquet files and upload to Firebase Storage.

Usage:
    python scripts/generate_and_upload_plots.py --type movies
    python scripts/generate_and_upload_plots.py --type tv
    python scripts/generate_and_upload_plots.py --type all
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from service.wiki_plot_job import generate_and_upload_plots


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and upload plot parquets")
    parser.add_argument("--type", choices=["movies", "tv", "all"], default="all")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets"),
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Read datasets from local parquet files instead of Firebase Storage",
    )
    args = parser.parse_args()

    types = ["movies", "tv"] if args.type == "all" else [args.type]
    for asset_type in types:
        result = generate_and_upload_plots(
            asset_type=asset_type,
            output_dir=args.output_dir,
            max_items=args.max_items,
            read_from_storage=not args.local,
        )
        print(f"Uploaded {result['count']} {asset_type} plots to {result['blob_name']}")


if __name__ == "__main__":
    main()
