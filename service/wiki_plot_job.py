import os
import time
from typing import Dict, List, Optional

import pandas as pd
import wikipedia

from firebase_admin import storage

from models.process_tv_and_movies import (
    _initialize_firebase_if_needed,
    _infer_bucket_name,
    get_asset_df,
    get_asset_df_from_storage,
    movies_path,
    tv_path,
)


REQUESTS_PER_SECOND = float(os.getenv("WIKI_REQUESTS_PER_SECOND", "1.5"))
REQUEST_DELAY = 1.0 / max(REQUESTS_PER_SECOND, 0.1)


def get_wikipedia_plot(name: str, is_movie: bool = True) -> Optional[str]:
    try:
        suffix = "(film)" if is_movie else "(TV series)"
        try:
            page = wikipedia.page(f"{name} {suffix}", auto_suggest=False)
        except (wikipedia.exceptions.PageError, wikipedia.exceptions.DisambiguationError):
            try:
                page = wikipedia.page(name, auto_suggest=False)
            except wikipedia.exceptions.DisambiguationError as e:
                if e.options:
                    try:
                        page = wikipedia.page(e.options[0], auto_suggest=False)
                    except Exception:
                        return None
                else:
                    return None
            except wikipedia.exceptions.PageError:
                return None

        plot = page.section("Plot")
        if plot and len(plot.strip()) > 50:
            return plot.strip()

        for section_name in ["Synopsis", "Summary", "Plot summary", "Premise"]:
            section = page.section(section_name)
            if section and len(section.strip()) > 50:
                return section.strip()

        if page.summary and len(page.summary.strip()) > 100:
            return page.summary.strip()

        return None
    except Exception as e:
        print(f"Error fetching plot for '{name}': {e}")
        return None


def _fetch_plot_with_delay(item: Dict[str, object], is_movie: bool) -> Dict[str, object]:
    time.sleep(REQUEST_DELAY)
    plot = get_wikipedia_plot(str(item["name"]), is_movie)
    return {
        "id": int(item["id"]),
        "name": str(item["name"]),
        "plot": plot,
    }


def generate_plots_dataframe(
    asset_type: str,
    max_items: Optional[int] = None,
    read_from_storage: bool = True,
) -> pd.DataFrame:
    is_movie = asset_type == "movies"
    title_col = "original_title" if is_movie else "original_name"

    if read_from_storage:
        blob_path = movies_path if is_movie else tv_path
        df = get_asset_df_from_storage(blob_path, "movie" if is_movie else "tv")
    else:
        df = get_asset_df("movies" if is_movie else "tv")

    if "original_language" in df.columns:
        df = df[df["original_language"] == "en"]

    if max_items:
        df = df.head(max_items)

    items = [
        {"id": row["id"], "name": row[title_col]}
        for _, row in df.iterrows()
        if pd.notna(row[title_col]) and str(row[title_col]).strip()
    ]

    results: List[Dict[str, object]] = []
    for item in items:
        results.append(_fetch_plot_with_delay(item, is_movie))

    plots_df = pd.DataFrame(results)
    plots_df = plots_df[plots_df["plot"].notna()]
    return plots_df


def upload_to_firebase(local_path: str, blob_name: str) -> None:
    _initialize_firebase_if_needed()
    bucket_name = _infer_bucket_name()
    bucket = storage.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)


def generate_and_upload_plots(
    asset_type: str,
    output_dir: str,
    max_items: Optional[int] = None,
    read_from_storage: bool = True,
) -> Dict[str, object]:
    os.makedirs(output_dir, exist_ok=True)

    plots_df = generate_plots_dataframe(
        asset_type=asset_type,
        max_items=max_items,
        read_from_storage=read_from_storage,
    )

    file_name = "movies_plots.parquet" if asset_type == "movies" else "tv_plots.parquet"
    local_path = os.path.join(output_dir, file_name)
    plots_df.to_parquet(local_path, engine="pyarrow", index=False)
    upload_to_firebase(local_path, file_name)

    return {
        "asset_type": asset_type,
        "count": len(plots_df),
        "local_path": local_path,
        "blob_name": file_name,
    }
