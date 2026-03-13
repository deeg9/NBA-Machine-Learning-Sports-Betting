"""
CLI script to fetch and backfill injury report data into SQLite.

Uses the nbainjuries package to fetch historical injury data and stores
it in Data/InjuryData.sqlite.

Usage:
    python -m src.Process-Data.Get_Injury_Data --seasons 2023-24 2024-25
    python -m src.Process-Data.Get_Injury_Data --seasons 2023-24 --force
"""

import argparse
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(1, os.fspath(BASE_DIR))

from src.DataProviders.PlayerDataProvider import (  # noqa: E402
    INJURY_DB_PATH,
    fetch_injury_data,
    load_injury_data,
    save_injury_data,
)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch NBA injury data into SQLite."
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        required=True,
        help="Season strings, e.g. 2023-24 2024-25",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch even if season data already exists.",
    )
    args = parser.parse_args()

    print(f"Injury data database: {INJURY_DB_PATH}")

    for season in args.seasons:
        print(f"\n── Season {season} ──")

        # Check if data already exists
        if not args.force:
            existing = load_injury_data(season)
            if not existing.empty:
                print(f"  Already have {len(existing)} rows for {season}. "
                      f"Use --force to re-fetch.")
                continue

        print(f"  Fetching injury data for {season}...")
        df = fetch_injury_data(season)
        if df.empty:
            print(f"  No injury data returned for {season}.")
            continue

        print(f"  Got {len(df)} injury report rows")
        if "Player" in df.columns:
            print(f"  Players: {df['Player'].nunique()}")
        if "Team" in df.columns:
            print(f"  Teams: {df['Team'].nunique()}")

        save_injury_data(df, season)

    print("\nDone.")


if __name__ == "__main__":
    main()
