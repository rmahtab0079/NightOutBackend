#!/usr/bin/env python3
"""
Clear user_curated_events in Firebase and optionally run the EventsParser pipeline.

Use this after fixing pipeline bugs (e.g. wrong events in the food category)
so that the next run repopulates with correct data.

Usage:
  # Clear only (you run the cron separately)
  python scripts/clear_curated_events_and_rerun.py

  # Clear and run the pipeline locally
  python scripts/clear_curated_events_and_rerun.py --rerun

  # Clear and run with custom radius/days
  python scripts/clear_curated_events_and_rerun.py --rerun --radius 50 --days 14
"""
from __future__ import annotations

import argparse
import os
import sys

# Ensure backend root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load env for Firebase
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clear user_curated_events and optionally rerun the EventsParser pipeline.",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="After clearing, run the pipeline (same as cron job).",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=50.0,
        help="Search radius in miles for pipeline (default 50).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=14,
        help="Days ahead to search (default 14).",
    )
    args = parser.parse_args()

    print("Clearing user_curated_events...")
    from service.events_parser.firebase_writer import clear_user_curated_events
    deleted = clear_user_curated_events()
    print(f"Cleared {deleted} user curated event document(s).")

    if args.rerun:
        print("\nRunning EventsParser pipeline...")
        from service.events_parser.pipeline import run_pipeline
        summary = run_pipeline(
            radius_miles=args.radius,
            days_ahead=args.days,
            max_events_per_user=80,
        )
        print("\nPipeline summary:", summary)
    else:
        print("\nSkipping pipeline (use --rerun to run it, or trigger the cron job).")


if __name__ == "__main__":
    main()
