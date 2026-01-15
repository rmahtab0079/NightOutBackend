import unittest
from models.process_tv_and_movies import get_asset_df, get_movie_ids_from_names, get_tv_ids_from_names, overview_similarity, get_movie_names_from_ids, find_similar_asset_v2, get_tfidf_from_movies, get_movie_detail
import wikipedia


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

class ProcessTVAndMoviesTest(unittest.TestCase):
    def test_find_similar_asset(self):
        assets = ['The Dark Knight', 'The Avengers']
        movie_ids = get_movie_ids_from_names(assets)
        print(movie_ids)
        asset_type = "movie"
        get_tfidf_from_movies(movie_ids, asset_type)

    def test_find_similar_asset_v2(self):
        assets = ['The Matrix', 'The Matrix Reloaded', 'The Dark Knight', 'Oppenheimer', 'Inception', 'Interstellar', 'Avengers: Endgame']
        movie_ids = get_movie_ids_from_names(assets)
        print(movie_ids)
        asset_type = "movie"
        find_similar_asset_v2(movie_ids, asset_type, read_from_local=True, top_n=5)

    def test_get_movie_detail(self):
        movie_id = "35"
        asset_type = "movie"
        print(get_movie_detail(movie_id, asset_type))
