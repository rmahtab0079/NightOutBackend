"""
Run a one-off job to precompute per-user genre recommendations.

Usage:
    python scripts/precompute_user_genre_recommendations.py
"""

import os
import sys

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from service.user_genre_recommendations import precompute_all_user_genre_recommendations


if __name__ == "__main__":
    precompute_all_user_genre_recommendations()
