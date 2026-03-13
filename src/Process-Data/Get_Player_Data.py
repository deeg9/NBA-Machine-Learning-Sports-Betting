"""
CLI script to fetch and backfill player game logs into SQLite.

Uses nba_api's PlayerGameLogs endpoint to fetch per-player per-game stats
for specified seasons and stores them in Data/PlayerGameLogs.sqlite.

Usage:
    python -m src.Process-Data.Get_Player_Data --seasons 2023-24 2024-25
    python -m src.Process-Data.Get_Player_Data --seasons 2023-24 --force
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(1, os.fspath(BASE_DIR))

from src.DataProviders.PlayerDataProvider import (  # noqa: E402
    PLAYER_DB_PATH,
    fetch_player_game_logs,
    load_player_game_logs,
    save_player_game_logs,
)

MIN_DELAY = 2
MAX_DELAY = 5


def main():
    parser = argparse.ArgumentParser(
        description="Fetch NBA player game logs into SQLite."
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

    print(f"Player game logs database: {PLAYER_DB_PATH}")

    for season in args.seasons:
        print(f"\n── Season {season} ──")

        # Check if data already exists
        if not args.force:
            existing = load_player_game_logs(season)
            if not existing.empty:
                print(f"  Already have {len(existing)} rows for {season}. "
                      f"Use --force to re-fetch.")
                continue

        print(f"  Fetching player game logs for {season}...")
        df = fetch_player_game_logs(season)
        if df.empty:
            print(f"  No data returned for {season}.")
            continue

        print(f"  Got {len(df)} player game log rows")
        print(f"  Players: {df['PLAYER_NAME'].nunique()}, "
              f"Teams: {df['TEAM_ABBREVIATION'].nunique()}, "
              f"Games: {df['GAME_ID'].nunique()}")

        save_player_game_logs(df, season)

        # Rate-limit between seasons
        delay = random.randint(MIN_DELAY, MAX_DELAY)
        print(f"  Waiting {delay}s before next request...")
        time.sleep(delay)

    print("\nDone.")


if __name__ == "__main__":
    main()
