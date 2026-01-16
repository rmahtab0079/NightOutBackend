"""
Tests for cache_database_layer.py

This test file covers:
- Cache key generation
- In-memory caching with TTL
- Firestore integration (with mocking)
- Full caching flow
- Parameter filtering
"""

import unittest
from unittest.mock import patch, MagicMock
import time
import hashlib
import json

from service.cache_database_layer import (
    _generate_cache_key,
    _is_cache_valid,
    _get_from_memory_cache,
    _set_memory_cache,
    _filter_results_by_params,
    get_similar_assets,
    clear_memory_cache,
    get_cache_stats,
    CACHE_TTL_SECONDS,
    _memory_cache
)


class TestCacheKeyGeneration(unittest.TestCase):
    """Test cache key generation logic."""

    def test_generate_cache_key_basic(self):
        """Test basic cache key generation."""
        key = _generate_cache_key(
            asset_ids={1, 2, 3},
            asset_type="movie"
        )
        self.assertIsInstance(key, str)
        self.assertEqual(len(key), 64)  # SHA256 hex digest length

    def test_generate_cache_key_deterministic(self):
        """Test that same inputs produce same key."""
        key1 = _generate_cache_key(
            asset_ids={1, 2, 3},
            asset_type="movie",
            genres=[28, 12],
            start_year=2000,
            end_year=2020
        )
        key2 = _generate_cache_key(
            asset_ids={3, 1, 2},  # Different order
            asset_type="movie",
            genres=[12, 28],  # Different order
            start_year=2000,
            end_year=2020
        )
        self.assertEqual(key1, key2)

    def test_generate_cache_key_different_for_different_inputs(self):
        """Test that different inputs produce different keys."""
        key1 = _generate_cache_key(
            asset_ids={1, 2, 3},
            asset_type="movie"
        )
        key2 = _generate_cache_key(
            asset_ids={1, 2, 3},
            asset_type="tv"
        )
        self.assertNotEqual(key1, key2)

    def test_generate_cache_key_with_all_params(self):
        """Test key generation with all parameters."""
        key = _generate_cache_key(
            asset_ids={278, 238, 240},
            asset_type="movie",
            genres=[28, 12, 878],
            start_year=1990,
            end_year=2025,
            top_n=10
        )
        self.assertIsInstance(key, str)
        self.assertEqual(len(key), 64)


class TestCacheValidation(unittest.TestCase):
    """Test cache entry validation."""

    def test_is_cache_valid_fresh_entry(self):
        """Test that fresh entries are valid."""
        entry = {"timestamp": time.time(), "data": []}
        self.assertTrue(_is_cache_valid(entry))

    def test_is_cache_valid_expired_entry(self):
        """Test that expired entries are invalid."""
        entry = {"timestamp": time.time() - CACHE_TTL_SECONDS - 1, "data": []}
        self.assertFalse(_is_cache_valid(entry))

    def test_is_cache_valid_missing_timestamp(self):
        """Test that entries without timestamp are invalid."""
        entry = {"data": []}
        self.assertFalse(_is_cache_valid(entry))

    def test_is_cache_valid_none_entry(self):
        """Test that None entries are invalid."""
        self.assertFalse(_is_cache_valid(None))

    def test_is_cache_valid_empty_entry(self):
        """Test that empty entries are invalid."""
        self.assertFalse(_is_cache_valid({}))


class TestMemoryCache(unittest.TestCase):
    """Test in-memory cache operations."""

    def setUp(self):
        """Clear cache before each test."""
        clear_memory_cache()

    def tearDown(self):
        """Clear cache after each test."""
        clear_memory_cache()

    def test_set_and_get_memory_cache(self):
        """Test basic set and get operations."""
        test_key = "test_key_123"
        test_data = [{"id": 1, "title": "Test Movie"}]
        
        _set_memory_cache(test_key, test_data)
        result = _get_from_memory_cache(test_key)
        
        self.assertEqual(result, test_data)

    def test_get_nonexistent_key(self):
        """Test getting a key that doesn't exist."""
        result = _get_from_memory_cache("nonexistent_key")
        self.assertIsNone(result)

    def test_clear_memory_cache(self):
        """Test clearing the memory cache."""
        _set_memory_cache("key1", [{"data": 1}])
        _set_memory_cache("key2", [{"data": 2}])
        
        count = clear_memory_cache()
        
        self.assertEqual(count, 2)
        self.assertIsNone(_get_from_memory_cache("key1"))
        self.assertIsNone(_get_from_memory_cache("key2"))


class TestFilterResults(unittest.TestCase):
    """Test result filtering logic."""

    def setUp(self):
        """Set up test data."""
        self.test_results = [
            {
                "movie_id": 1,
                "title": "Action Movie",
                "detail": {
                    "genres": [{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}],
                    "release_date": "2020-05-15"
                }
            },
            {
                "movie_id": 2,
                "title": "Comedy Movie",
                "detail": {
                    "genres": [{"id": 35, "name": "Comedy"}],
                    "release_date": "2015-03-20"
                }
            },
            {
                "movie_id": 3,
                "title": "Sci-Fi Movie",
                "detail": {
                    "genres": [{"id": 878, "name": "Science Fiction"}],
                    "release_date": "2022-11-01"
                }
            }
        ]

    def test_filter_no_params(self):
        """Test filtering with no parameters returns all results."""
        filtered = _filter_results_by_params(
            results=self.test_results,
            genres=None,
            start_year=None,
            end_year=None,
            asset_type="movie"
        )
        self.assertEqual(len(filtered), 3)

    def test_filter_by_genre(self):
        """Test filtering by genre ID."""
        filtered = _filter_results_by_params(
            results=self.test_results,
            genres=[28],  # Action
            start_year=None,
            end_year=None,
            asset_type="movie"
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["movie_id"], 1)

    def test_filter_by_multiple_genres(self):
        """Test filtering by multiple genre IDs."""
        filtered = _filter_results_by_params(
            results=self.test_results,
            genres=[28, 35],  # Action or Comedy
            start_year=None,
            end_year=None,
            asset_type="movie"
        )
        self.assertEqual(len(filtered), 2)

    def test_filter_by_year_range(self):
        """Test filtering by year range."""
        filtered = _filter_results_by_params(
            results=self.test_results,
            genres=None,
            start_year=2018,
            end_year=2023,
            asset_type="movie"
        )
        self.assertEqual(len(filtered), 2)  # 2020 and 2022 movies

    def test_filter_by_start_year_only(self):
        """Test filtering with only start year."""
        filtered = _filter_results_by_params(
            results=self.test_results,
            genres=None,
            start_year=2020,
            end_year=None,
            asset_type="movie"
        )
        self.assertEqual(len(filtered), 2)  # 2020 and 2022 movies

    def test_filter_by_end_year_only(self):
        """Test filtering with only end year."""
        filtered = _filter_results_by_params(
            results=self.test_results,
            genres=None,
            start_year=None,
            end_year=2016,
            asset_type="movie"
        )
        self.assertEqual(len(filtered), 1)  # 2015 movie

    def test_filter_combined_genre_and_year(self):
        """Test filtering with both genre and year."""
        filtered = _filter_results_by_params(
            results=self.test_results,
            genres=[28, 878],  # Action or Sci-Fi
            start_year=2021,
            end_year=2025,
            asset_type="movie"
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["movie_id"], 3)  # Only Sci-Fi Movie from 2022


class TestFilterResultsTV(unittest.TestCase):
    """Test result filtering for TV shows."""

    def setUp(self):
        """Set up test data for TV shows."""
        self.test_results = [
            {
                "movie_id": 1,
                "title": "Drama Show",
                "detail": {
                    "genres": [{"id": 18, "name": "Drama"}],
                    "first_air_date": "2019-01-15"
                }
            },
            {
                "movie_id": 2,
                "title": "Comedy Show",
                "detail": {
                    "genres": [{"id": 35, "name": "Comedy"}],
                    "first_air_date": "2021-06-20"
                }
            }
        ]

    def test_filter_tv_by_first_air_date(self):
        """Test filtering TV shows by first air date."""
        filtered = _filter_results_by_params(
            results=self.test_results,
            genres=None,
            start_year=2020,
            end_year=2025,
            asset_type="tv"
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["movie_id"], 2)


class TestGetSimilarAssetsWithMocking(unittest.TestCase):
    """Test get_similar_assets with mocked dependencies."""

    def setUp(self):
        """Clear cache before each test."""
        clear_memory_cache()

    def tearDown(self):
        """Clear cache after each test."""
        clear_memory_cache()

    @patch('service.cache_database_layer.find_similar_asset_v2')
    @patch('service.cache_database_layer._get_firestore_client')
    def test_get_similar_assets_cache_miss_calls_api(self, mock_firestore, mock_api):
        """Test that cache miss triggers API call."""
        mock_firestore.return_value = None  # No Firestore
        mock_api.return_value = [
            {"movie_id": 1, "title": "Test", "similarity": 0.9, "detail": {}}
        ]

        result = get_similar_assets(
            asset_ids={278, 238},
            asset_type="movie",
            bypass_cache=True
        )

        mock_api.assert_called_once()
        self.assertEqual(len(result), 1)

    @patch('service.cache_database_layer.find_similar_asset_v2')
    @patch('service.cache_database_layer._get_firestore_client')
    def test_get_similar_assets_cache_hit_skips_api(self, mock_firestore, mock_api):
        """Test that cache hit skips API call."""
        mock_firestore.return_value = None  # No Firestore
        test_data = [{"movie_id": 1, "title": "Cached", "similarity": 0.9, "detail": {}}]
        
        # First call to populate cache
        mock_api.return_value = test_data
        get_similar_assets(
            asset_ids={278, 238},
            asset_type="movie"
        )
        
        # Reset mock
        mock_api.reset_mock()
        
        # Second call should hit cache
        result = get_similar_assets(
            asset_ids={278, 238},
            asset_type="movie"
        )

        mock_api.assert_not_called()
        self.assertEqual(result, test_data)

    @patch('service.cache_database_layer.find_similar_asset_v2')
    @patch('service.cache_database_layer._get_firestore_client')
    def test_get_similar_assets_bypass_cache(self, mock_firestore, mock_api):
        """Test bypass_cache forces API call."""
        mock_firestore.return_value = None
        mock_api.return_value = [
            {"movie_id": 1, "title": "Fresh", "similarity": 0.9, "detail": {}}
        ]
        
        # Pre-populate cache
        _set_memory_cache(
            _generate_cache_key({278, 238}, "movie"),
            [{"movie_id": 1, "title": "Stale", "similarity": 0.9, "detail": {}}]
        )
        
        # Call with bypass
        result = get_similar_assets(
            asset_ids={278, 238},
            asset_type="movie",
            bypass_cache=True
        )

        mock_api.assert_called_once()
        self.assertEqual(result[0]["title"], "Fresh")


class TestCacheStats(unittest.TestCase):
    """Test cache statistics."""

    def setUp(self):
        """Clear cache before each test."""
        clear_memory_cache()

    def tearDown(self):
        """Clear cache after each test."""
        clear_memory_cache()

    @patch('service.cache_database_layer._get_firestore_client')
    def test_get_cache_stats_empty(self, mock_firestore):
        """Test stats with empty cache."""
        mock_firestore.return_value = None
        
        stats = get_cache_stats()
        
        self.assertEqual(stats["memory_cache_total"], 0)
        self.assertEqual(stats["memory_cache_valid"], 0)
        self.assertEqual(stats["cache_ttl_seconds"], CACHE_TTL_SECONDS)

    @patch('service.cache_database_layer._get_firestore_client')
    def test_get_cache_stats_with_entries(self, mock_firestore):
        """Test stats with cache entries."""
        mock_firestore.return_value = None
        
        _set_memory_cache("key1", [{"data": 1}])
        _set_memory_cache("key2", [{"data": 2}])
        
        stats = get_cache_stats()
        
        self.assertEqual(stats["memory_cache_total"], 2)
        self.assertEqual(stats["memory_cache_valid"], 2)


class TestIntegration(unittest.TestCase):
    """
    Integration tests that test the full flow.
    These tests require local parquet files to be available.
    """

    def setUp(self):
        """Clear cache before each test."""
        clear_memory_cache()

    def tearDown(self):
        """Clear cache after each test."""
        clear_memory_cache()

    @unittest.skip("Requires local parquet files and may take a while")
    def test_full_flow_with_local_data(self):
        """
        Test the full caching flow with local data.
        Skip by default as it requires data files.
        """
        from models.process_tv_and_movies import get_movie_ids_from_names
        
        # Get movie IDs
        movie_names = ['The Matrix', 'Inception']
        movie_ids = get_movie_ids_from_names(movie_names)
        
        # First call - should fetch from API
        result1 = get_similar_assets(
            asset_ids=movie_ids,
            asset_type="movie",
            read_from_local=True,
            top_n=3
        )
        
        self.assertIsInstance(result1, list)
        self.assertGreater(len(result1), 0)
        
        # Second call - should hit cache
        result2 = get_similar_assets(
            asset_ids=movie_ids,
            asset_type="movie",
            read_from_local=True,
            top_n=3
        )
        
        self.assertEqual(result1, result2)


if __name__ == "__main__":
    unittest.main()
