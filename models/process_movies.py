import os
import pandas as pd
from scipy.spatial.distance import pdist, squareform


def movies_dataframe(movie: str):
    """
    This function reads the Parquet File, processes genre_ids, and returns a DataFrame with crosstab operation.
    """
    file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets", "movies_dataset.parquet")
    movie_genre_df = pd.read_parquet(file_path, engine="pyarrow")

    # Explode the 'genre_ids' column to split lists into individual rows
    movie_genre_df = movie_genre_df.explode('genre_ids')

    # Convert genre_ids to integers if necessary (may currently be strings)
    movie_genre_df['genre_ids'] = movie_genre_df['genre_ids'].astype(float)

    # Create cross-tabulation between original_title and genre_ids
    movie_cross_table = pd.crosstab(movie_genre_df['original_title'], movie_genre_df['genre_ids'])

    # Select rows for movies

    jaccard_distances = pdist(movie_cross_table, metric='jaccard')
    square_jaccard_distances = squareform(jaccard_distances)
    jaccard_similarity_array = 1 - square_jaccard_distances
    distance_df = pd.DataFrame(jaccard_similarity_array, index=movie_cross_table.index, columns=movie_cross_table.index)
    print(distance_df[movie].sort_values(ascending=False))


if __name__ == "__main__":
    movies_dataframe("Cars")