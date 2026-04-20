#!/usr/bin/env python3
"""
Clear stale event data in Firebase and optionally run the EventsParser pipeline.

By default this will:
  1. Purge past and imageless entries from `parsed_events_catalog`.
  2. Clear every document in `user_curated_events` so the next pipeline run
     rebuilds user-facing lists from a clean catalog.

Use this after fixing pipeline bugs (e.g. wrong events in the food category,
or past events lingering from pre-filter runs) so that the next run
repopulates with correct data.

Usage:
  # Clean up only (cron will refresh lists on its next tick)
  python scripts/clear_curated_events_and_rerun.py

  # Skip the catalog purge, just wipe per-user curated lists
  python scripts/clear_curated_events_and_rerun.py --skip-catalog-purge

  # Clean up and run the pipeline locally
  python scripts/clear_curated_events_and_rerun.py --rerun

  # Clean up and run with custom radius/days
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
    parser.add_argument(
        "--skip-catalog-purge",
        action="store_true",
        help="Do not purge past/imageless entries from parsed_events_catalog.",
    )
    parser.add_argument(
        "--skip-image-probe",
        action="store_true",
        help=(
            "Skip the per-URL HTTP reachability probe during the catalog "
            "purge. The probe is what removes URLs that look valid but 403 "
            "(e.g. Partiful Firebase Storage links). Skip only if you need a "
            "fast purge."
        ),
    )
    args = parser.parse_args()

    from service.events_parser.firebase_writer import (
        clear_user_curated_events,
        purge_catalog_outside_user_radius,
        purge_stale_parsed_events,
    )

    if not args.skip_catalog_purge:
        probe = not args.skip_image_probe
        print(
            "Purging parsed_events_catalog (past + imageless"
            + (" + unreachable" if probe else "")
            + " events)..."
        )
        stats = purge_stale_parsed_events(probe_network=probe)
        print(
            f"Catalog purge: kept={stats['kept']}, "
            f"past_removed={stats['deleted_past']}, "
            f"imageless_removed={stats['deleted_no_image']}, "
            f"unreachable_removed={stats.get('deleted_unreachable', 0)}."
        )

        print(
            f"\nPurging parsed_events_catalog entries not within "
            f"{args.radius:.0f}mi of any active user..."
        )
        loc_stats = purge_catalog_outside_user_radius(radius_miles=args.radius)
        print(
            f"Location purge: kept={loc_stats['kept']}, "
            f"out_of_range_removed={loc_stats['deleted_far']}, "
            f"kept_no_coord={loc_stats['kept_no_coord']}."
        )
    else:
        print("Skipping parsed_events_catalog purge (--skip-catalog-purge).")

    print("\nClearing user_curated_events...")
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
