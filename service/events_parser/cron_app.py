"""
Cloud Run entry point for the EventsParser cron job.

Exposes a minimal FastAPI app with a single endpoint that Cloud Scheduler
hits on a schedule. Also runnable as a CLI for manual execution.

GCP Cloud Scheduler -> Cloud Run -> this app -> /run_events_parser
"""

from __future__ import annotations

import os
import sys
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Header
from typing import Optional

app = FastAPI(title="EventsParser Cron", version="1.0.0")


@app.get("/")
def health():
    return {"status": "ok", "service": "events_parser"}


@app.post("/run_events_parser")
async def run_events_parser(
    authorization: Optional[str] = Header(default=None),
    radius_miles: float = 25.0,
    days_ahead: int = 14,
    max_events_per_user: int = 50,
):
    """
    Trigger a full EventsParser pipeline run.

    In production, this is called by Cloud Scheduler with an OIDC token.
    The authorization header can be used to verify the caller.
    """
    cron_secret = os.getenv("EVENTS_PARSER_CRON_SECRET", "")
    if cron_secret:
        token = (authorization or "").replace("Bearer ", "")
        if token != cron_secret:
            raise HTTPException(status_code=403, detail="Invalid cron secret")

    from .pipeline import run_pipeline

    try:
        summary = run_pipeline(
            radius_miles=radius_miles,
            days_ahead=days_ahead,
            max_events_per_user=max_events_per_user,
        )
        return {"status": "completed", **summary}
    except Exception as e:
        print(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """CLI entry point for manual runs."""
    import argparse

    parser = argparse.ArgumentParser(description="Run EventsParser pipeline")
    parser.add_argument("--radius", type=float, default=25.0, help="Search radius in miles")
    parser.add_argument("--days", type=int, default=14, help="Days ahead to search")
    parser.add_argument("--max-events", type=int, default=50, help="Max events per user")
    args = parser.parse_args()

    from .pipeline import run_pipeline

    summary = run_pipeline(
        radius_miles=args.radius,
        days_ahead=args.days,
        max_events_per_user=args.max_events,
    )
    print(f"\nDone: {summary}")


if __name__ == "__main__":
    main()
