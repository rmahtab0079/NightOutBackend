#!/usr/bin/env python3
"""
Merge Kaggle Wikipedia movie plots with TMDB dataset and upload parquet.

Produces a parquet with columns: id, name, plot
"""

import os
import re
import sys
import argparse
from typing import Optional

import pandas as pd

from firebase_admin import storage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.process_tv_and_movies import _initialize_firebase_if_needed, _infer_bucket_name


def _normalize_title(title: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
    return re.sub(r"\s+", " ", cleaned)


def _parse_year(value: Optional[str]) -> Optional[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return int(str(value).split("-", 1)[0])
    except Exception:
        return None


def _upload_to_firebase(local_path: str, blob_name: str) -> None:
    _initialize_firebase_if_needed()
    bucket_name = _infer_bucket_name()
    bucket = storage.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)


def merge_and_upload(
    kaggle_csv: str,
    tmdb_parquet: str,
    output_path: str,
    upload: bool,
) -> dict:
    kaggle_df = pd.read_csv(kaggle_csv)
    kaggle_df = kaggle_df.rename(columns={"Title": "title", "Plot": "plot", "Release Year": "release_year"})
    kaggle_df["release_year"] = kaggle_df["release_year"].apply(_parse_year)
    kaggle_df["title_norm"] = kaggle_df["title"].astype(str).apply(_normalize_title)
    kaggle_df = kaggle_df[kaggle_df["plot"].notna()]

    tmdb_df = pd.read_parquet(tmdb_parquet, engine="pyarrow")
    tmdb_df["release_year"] = tmdb_df["release_date"].apply(_parse_year)
    tmdb_df["title_norm"] = tmdb_df["original_title"].fillna(tmdb_df["title"]).astype(str).apply(_normalize_title)

    merged = pd.merge(
        tmdb_df,
        kaggle_df,
        on=["title_norm", "release_year"],
        how="inner",
        suffixes=("_tmdb", "_kaggle"),
    )

    if merged.empty:
        return {"count": 0, "output_path": output_path, "uploaded": False}

    result_df = merged[["id", "original_title", "plot"]].rename(
        columns={"original_title": "name"}
    )
    result_df = result_df.dropna(subset=["plot"]).drop_duplicates(subset=["id"])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_parquet(output_path, engine="pyarrow", index=False)

    uploaded = False
    if upload:
        _upload_to_firebase(output_path, os.path.basename(output_path))
        uploaded = True

    return {"count": len(result_df), "output_path": output_path, "uploaded": uploaded}


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Kaggle plots into TMDB IDs")
    parser.add_argument(
        "--kaggle-csv",
        default=os.path.join("datasets", "wiki_movie_plots_deduped.csv"),
    )
    parser.add_argument(
        "--tmdb-parquet",
        default=os.path.join("datasets", "movies_dataset.parquet"),
    )
    parser.add_argument(
        "--output",
        default=os.path.join("datasets", "movies_plots.parquet"),
    )
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    result = merge_and_upload(
        kaggle_csv=args.kaggle_csv,
        tmdb_parquet=args.tmdb_parquet,
        output_path=args.output,
        upload=args.upload,
    )
    print(f"Merged {result['count']} plots -> {result['output_path']}")
    if result["uploaded"]:
        print(f"Uploaded to gs://{_infer_bucket_name()}/{os.path.basename(args.output)}")


if __name__ == "__main__":
    main()
