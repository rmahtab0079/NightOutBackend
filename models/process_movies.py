import os
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_movies_df() -> pd.DataFrame:
    file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets", "movies_dataset.parquet")
    movie_df = pd.read_parquet(file_path, engine="pyarrow")

    movie_df = movie_df.drop_duplicates(subset='original_title', keep='first')

    return movie_df

def genre_similarity(movies: set[str]) -> pd.DataFrame:
    """
    This function reads the Parquet File, processes genre_ids, and returns a DataFrame with crosstab operation.
    """
    movie_genre_df = get_movies_df()

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

    # Get similarities for each movie in the set and average them
    valid_movies = [m for m in movies if m in distance_df.columns]

    if not valid_movies:
        raise ValueError("None of the provided movies found in dataset")

        # Average similarity across all input movies
    avg_similarity = distance_df[valid_movies].mean(axis=1)
    sorted_movie_df = avg_similarity.sort_values(ascending=False)

    return sorted_movie_df

def overview_similarity(movies: set[str]) -> pd.DataFrame:
    # Create vectorized data from the number of word occurences
    movies_df = get_movies_df()
    vectorizer = TfidfVectorizer(min_df=2, max_df=0.7)
    vectorized_data = vectorizer.fit_transform(movies_df['overview'])

    movies_overview_df = pd.DataFrame(vectorized_data.toarray(), columns=vectorizer.get_feature_names_out())

    movies_overview_df.index = movies_df['original_title']

    # Calculate the cosine similarity
    cosine_similarity_array = cosine_similarity(movies_overview_df)

    cosine_similarity_df = pd.DataFrame(cosine_similarity_array, index=movies_overview_df.index,
                                        columns=movies_overview_df.index)

    # Get similarities for each movie in the set and average them
    valid_movies = [m for m in movies if m in cosine_similarity_df.columns]

    if not valid_movies:
        raise ValueError("None of the provided movies found in dataset")

    # Average similarity across all input movies
    avg_similarity = cosine_similarity_df[valid_movies].mean(axis=1)

    ordered_similarities = avg_similarity.sort_values(ascending=False)

    return ordered_similarities


def find_similar_movie(movies: set[str]):
    movies_df = get_movies_df()
    movies_df.set_index('original_title', inplace=True)
    genre_similarity_df = genre_similarity(movies)
    overview_similarity_df = overview_similarity(movies)

    # Combine both similarities into one DataFrame
    combined_df = pd.DataFrame({
        'genre_sim': genre_similarity_df,
        'overview_sim': overview_similarity_df
    })

    # Calculate combined score (you can adjust weights)
    combined_df['combined_score'] = (combined_df['genre_sim'] * 0.5 +
                                     combined_df['overview_sim'] * 0.5)

    # Sort by combined score
    combined_df = combined_df.sort_values('combined_score', ascending=False)

    # Find first non-sequel/prequel movie
    for curr_movie_title in combined_df.index:
        # Skip if it's one of the input movies
        if curr_movie_title in movies:
            continue

        if movies_df.loc[curr_movie_title, 'original_language'] != "en":
            continue

        # Check if it's a sequel/prequel of any of the input movies
        is_related = False
        for input_movie in movies:
            if is_sequel_or_prequel(input_movie, curr_movie_title):
                is_related = True
                break

        if not is_related:
            print(f"Found similar movie: {curr_movie_title}")
            print(f"Genre similarity: {combined_df.loc[curr_movie_title, 'genre_sim']:.3f}")
            print(f"Overview similarity: {combined_df.loc[curr_movie_title, 'overview_sim']:.3f}")
            print(f"Combined score: {combined_df.loc[curr_movie_title, 'combined_score']:.3f}")
            return curr_movie_title

    return None


def is_sequel_or_prequel(original_movie: str, candidate_movie: str) -> bool:
    """
    Check if candidate_movie is likely a sequel or prequel of original_movie
    """
    original_lower = original_movie.lower()
    candidate_lower = candidate_movie.lower()

    # Remove common sequel/prequel indicators to get base title
    sequel_prequel_indicators = [
        ' 2', ' 3', ' 4', ' 5', ' 6', ' 7', ' 8', ' 9',
        ' ii', ' iii', ' iv', ' v', ' vi', ' vii', ' viii',
        ' part', ' chapter',
        ': the', ' returns', ' reloaded', ' revolutions', ' rises',
        ' begins', ' origins', ' first', ' zero', ' episode',
        ' prequel', ' sequel', ' saga', ' chronicles'
    ]

    original_base = original_lower
    candidate_base = candidate_lower

    for indicator in sequel_prequel_indicators:
        original_base = original_base.split(indicator)[0]
        candidate_base = candidate_base.split(indicator)[0]

    # If base titles are very similar, likely a sequel or prequel
    return original_base in candidate_base or candidate_base in original_base


if __name__ == "__main__":
    movies = set(["The Godfather", "The Matrix", "The Dark Knight", "Oppenheimer", "Dune"])
    find_similar_movie(movies)
