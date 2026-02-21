import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import firebase_admin
from firebase_admin import credentials, firestore

from models.process_tv_and_movies import (
    get_asset_df_from_storage,
    get_plots_df,
    movies_path,
    tv_path,
)


MOVIE_RECS_COLLECTION = "user_genre_recommendations_movies"
TV_RECS_COLLECTION = "user_genre_recommendations_tv"
_scheduler_started = False
_GENRE_TFIDF_CACHE: dict[str, Dict[int, tuple[TfidfVectorizer, List[int], Any]]] = {}


def _get_firestore_client() -> firestore.Client | None:
    if firebase_admin._apps:  # type: ignore[attr-defined]
        return firestore.client()

    service_account_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH", "serviceAccount.json")
    project_id = os.getenv("FIREBASE_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")

    try:
        if service_account_path and os.path.exists(service_account_path):
            cred = credentials.Certificate(service_account_path)
            options = {"projectId": project_id} if project_id else None
            firebase_admin.initialize_app(cred, options)
        else:
            options = {"projectId": project_id} if project_id else None
            firebase_admin.initialize_app(options=options)
        return firestore.client()
    except Exception as e:
        print(f"Failed to initialize Firestore: {e}")
        return None


def _build_genre_index(
    asset_df: pd.DataFrame,
    id_to_plot: Dict[int, str],
) -> Dict[int, List[int]]:
    genre_to_ids: Dict[int, List[int]] = {}

    for _, row in asset_df.iterrows():
        asset_id = int(row["id"])
        if asset_id not in id_to_plot:
            continue
        genre_ids = row.get("genre_ids")
        if genre_ids is None:
            continue
        if isinstance(genre_ids, float):
            continue
        for genre_id in genre_ids:
            try:
                gid = int(genre_id)
            except Exception:
                continue
            genre_to_ids.setdefault(gid, []).append(asset_id)

    return genre_to_ids


def _compute_top_similar_ids(
    user_plot_texts: List[str],
    candidate_ids: List[int],
    id_to_plot: Dict[int, str],
    limit: int = 100,
) -> List[int]:
    candidate_texts = [id_to_plot[i] for i in candidate_ids if i in id_to_plot]
    if not candidate_texts:
        return []

    all_texts = user_plot_texts + candidate_texts
    vectorizer = TfidfVectorizer(stop_words="english", min_df=1, max_df=0.95)
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    user_vectors = tfidf_matrix[: len(user_plot_texts)]
    candidate_vectors = tfidf_matrix[len(user_plot_texts) :]

    user_profile = user_vectors.mean(axis=0)
    similarities = cosine_similarity(user_profile, candidate_vectors)[0]

    ranked = sorted(zip(candidate_ids, similarities), key=lambda x: x[1], reverse=True)
    return [asset_id for asset_id, _ in ranked[:limit]]


def _rank_and_filter_ids(
    candidate_ids: List[int],
    similarities: List[float],
    excluded_ids: Set[int],
    limit: int,
) -> List[int]:
    ranked = sorted(zip(candidate_ids, similarities), key=lambda x: x[1], reverse=True)
    top_ids: List[int] = []
    for asset_id, _ in ranked:
        if asset_id in excluded_ids:
            continue
        top_ids.append(asset_id)
        if len(top_ids) >= limit:
            break
    return top_ids


def _load_assets_and_plots(asset_type: str) -> Tuple[pd.DataFrame, Dict[int, str], Dict[int, List[int]]]:
    if asset_type == "movie":
        asset_df = get_asset_df_from_storage(movies_path, "movie")
    else:
        asset_df = get_asset_df_from_storage(tv_path, "tv")

    if "original_language" in asset_df.columns:
        asset_df = asset_df[asset_df["original_language"] == "en"]

    plots_df = get_plots_df(asset_type, read_from_local=False)
    id_to_plot = {int(row["id"]): row["plot"] for _, row in plots_df.iterrows() if isinstance(row["plot"], str)}

    genre_to_ids = _build_genre_index(asset_df, id_to_plot)
    return asset_df, id_to_plot, genre_to_ids


def _build_genre_tfidf_cache(
    asset_type: str,
    genre_to_ids: Dict[int, List[int]],
    id_to_plot: Dict[int, str],
) -> Dict[int, tuple[TfidfVectorizer, List[int], Any]]:
    cache_key = f"{asset_type}"
    if cache_key in _GENRE_TFIDF_CACHE:
        return _GENRE_TFIDF_CACHE[cache_key]

    genre_cache: Dict[int, tuple[TfidfVectorizer, List[int], Any]] = {}
    for genre_id, candidate_ids in genre_to_ids.items():
        filtered_ids = [cid for cid in candidate_ids if cid in id_to_plot]
        if not filtered_ids:
            continue
        texts = [id_to_plot[cid] for cid in filtered_ids]
        vectorizer = TfidfVectorizer(stop_words="english", min_df=1, max_df=0.95)
        candidate_matrix = vectorizer.fit_transform(texts)
        genre_cache[genre_id] = (vectorizer, filtered_ids, candidate_matrix)

    _GENRE_TFIDF_CACHE[cache_key] = genre_cache
    return genre_cache


def _compute_user_genre_recommendations(
    user_doc: dict,
    asset_type: str,
    id_to_plot: Dict[int, str],
    genre_to_ids: Dict[int, List[int]],
    limit: int = 100,
    genre_tfidf_cache: Optional[Dict[int, tuple[TfidfVectorizer, List[int], Any]]] = None,
) -> Dict[str, List[int]]:
    if asset_type == "movie":
        watched_ids = user_doc.get("movieIds", [])
        disliked_ids = set(user_doc.get("dislikedMovieIds", []))
    else:
        watched_ids = user_doc.get("tvShowIds", [])
        disliked_ids = set(user_doc.get("dislikedTvShowIds", []))

    watched_ids = [int(i) for i in watched_ids]
    user_plot_texts = [id_to_plot[i] for i in watched_ids if i in id_to_plot]
    if not user_plot_texts:
        return {}

    excluded_ids: Set[int] = set(watched_ids) | set(disliked_ids)
    genre_recs: Dict[str, List[int]] = {}
    for genre_id, candidate_ids in genre_to_ids.items():
        if genre_tfidf_cache and genre_id in genre_tfidf_cache:
            vectorizer, cached_candidate_ids, candidate_matrix = genre_tfidf_cache[genre_id]
            user_vectors = vectorizer.transform(user_plot_texts)
            user_profile = np.asarray(user_vectors.mean(axis=0))
            similarities = cosine_similarity(user_profile, candidate_matrix)[0].tolist()
            top_ids = _rank_and_filter_ids(
                candidate_ids=cached_candidate_ids,
                similarities=similarities,
                excluded_ids=excluded_ids,
                limit=limit,
            )
        else:
            filtered_candidates = [cid for cid in candidate_ids if cid not in excluded_ids]
            if not filtered_candidates:
                continue
            top_ids = _compute_top_similar_ids(
                user_plot_texts=user_plot_texts,
                candidate_ids=filtered_candidates,
                id_to_plot=id_to_plot,
                limit=limit,
            )
        if top_ids:
            genre_recs[str(genre_id)] = top_ids

    return genre_recs


def _prepare_recommendation_context() -> dict[str, object]:
    movie_asset_df, movie_plot_map, movie_genre_to_ids = _load_assets_and_plots("movie")
    try:
        tv_asset_df, tv_plot_map, tv_genre_to_ids = _load_assets_and_plots("tv")
    except FileNotFoundError:
        print("TV plots not found. Skipping TV precompute.")
        tv_plot_map = {}
        tv_genre_to_ids = {}
        tv_asset_df = None
    del movie_asset_df
    if tv_asset_df is not None:
        del tv_asset_df

    movie_genre_cache = _build_genre_tfidf_cache("movie", movie_genre_to_ids, movie_plot_map)
    tv_genre_cache = _build_genre_tfidf_cache("tv", tv_genre_to_ids, tv_plot_map) if tv_plot_map else {}

    return {
        "movie_plot_map": movie_plot_map,
        "movie_genre_to_ids": movie_genre_to_ids,
        "movie_genre_cache": movie_genre_cache,
        "tv_plot_map": tv_plot_map,
        "tv_genre_to_ids": tv_genre_to_ids,
        "tv_genre_cache": tv_genre_cache,
    }


def _set_recommendations(
    db: firestore.Client,
    email: str,
    movie_recs: Optional[Dict[str, List[int]]],
    tv_recs: Optional[Dict[str, List[int]]],
    batch: Optional[firestore.WriteBatch] = None,
) -> None:
    if movie_recs is not None:
        payload = {"genres": movie_recs, "updatedAt": datetime.utcnow().isoformat() + "Z"}
        movie_ref = db.collection(MOVIE_RECS_COLLECTION).document(email)
        if batch:
            batch.set(movie_ref, payload)
        else:
            movie_ref.set(payload)
    if tv_recs is not None:
        payload = {"genres": tv_recs, "updatedAt": datetime.utcnow().isoformat() + "Z"}
        tv_ref = db.collection(TV_RECS_COLLECTION).document(email)
        if batch:
            batch.set(tv_ref, payload)
        else:
            tv_ref.set(payload)


def precompute_user_genre_recommendations_for_email(
    email: str,
    limit_per_genre: int = 100,
    asset_type: Optional[str] = None,
) -> None:
    db = _get_firestore_client()
    if db is None:
        print("Firestore not configured. Skipping precompute.")
        return

    user_doc = db.collection("user_preferences").document(email).get()
    if not user_doc.exists:
        print(f"User not found: {email}")
        return

    context = _prepare_recommendation_context()
    user_data = user_doc.to_dict() or {}

    movie_recs = None
    tv_recs = None
    if asset_type in (None, "movie"):
        movie_recs = _compute_user_genre_recommendations(
            user_doc=user_data,
            asset_type="movie",
            id_to_plot=context["movie_plot_map"],
            genre_to_ids=context["movie_genre_to_ids"],
            limit=limit_per_genre,
            genre_tfidf_cache=context["movie_genre_cache"],
        )
    if asset_type in (None, "tv"):
        tv_recs = _compute_user_genre_recommendations(
            user_doc=user_data,
            asset_type="tv",
            id_to_plot=context["tv_plot_map"],
            genre_to_ids=context["tv_genre_to_ids"],
            limit=limit_per_genre,
            genre_tfidf_cache=context["tv_genre_cache"],
        )

    _set_recommendations(db, email, movie_recs if movie_recs else {}, tv_recs if tv_recs else {})
    print(f"Updated genre recommendations for {email}")


def precompute_all_user_genre_recommendations(limit_per_genre: int = 100) -> None:
    db = _get_firestore_client()
    if db is None:
        print("Firestore not configured. Skipping precompute.")
        return

    context = _prepare_recommendation_context()

    users_ref = db.collection("user_preferences")
    users = list(users_ref.stream())
    total_users = len(users)
    print(f"Starting genre precompute for {total_users} users")
    start_time = time.time()

    batch = db.batch()
    batch_count = 0
    batch_limit = 400

    for idx, user in enumerate(users, start=1):
        user_doc = user.to_dict()
        email = user_doc.get("email") or user.id
        if not email:
            continue

        movie_recs = _compute_user_genre_recommendations(
            user_doc=user_doc,
            asset_type="movie",
            id_to_plot=context["movie_plot_map"],
            genre_to_ids=context["movie_genre_to_ids"],
            limit=limit_per_genre,
            genre_tfidf_cache=context["movie_genre_cache"],
        )
        tv_recs = _compute_user_genre_recommendations(
            user_doc=user_doc,
            asset_type="tv",
            id_to_plot=context["tv_plot_map"],
            genre_to_ids=context["tv_genre_to_ids"],
            limit=limit_per_genre,
            genre_tfidf_cache=context["tv_genre_cache"],
        )

        if movie_recs:
            movie_ref = db.collection(MOVIE_RECS_COLLECTION).document(email)
            batch.set(movie_ref, {"genres": movie_recs, "updatedAt": datetime.utcnow().isoformat() + "Z"})
            batch_count += 1
        if tv_recs:
            tv_ref = db.collection(TV_RECS_COLLECTION).document(email)
            batch.set(tv_ref, {"genres": tv_recs, "updatedAt": datetime.utcnow().isoformat() + "Z"})
            batch_count += 1

        if batch_count >= batch_limit:
            batch.commit()
            batch = db.batch()
            batch_count = 0
        if idx % 10 == 0 or idx == total_users:
            elapsed = time.time() - start_time
            print(f"Progress: {idx}/{total_users} users ({elapsed:.1f}s)")

    if batch_count:
        batch.commit()
    total_elapsed = time.time() - start_time
    print(f"Completed genre precompute for {total_users} users in {total_elapsed:.1f}s")


def start_genre_recommendation_scheduler(interval_seconds: int = 300) -> None:
    global _scheduler_started
    if _scheduler_started:
        return
    _scheduler_started = True

    def _loop() -> None:
        while True:
            try:
                precompute_all_user_genre_recommendations()
            except Exception as e:
                print(f"Genre recommendation job error: {e}")
            time.sleep(interval_seconds)

    import threading

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()
