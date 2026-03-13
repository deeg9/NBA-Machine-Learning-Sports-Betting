"""
Provider for NBA lineup stats via nba_api.

Fetches 5-man lineup data including net rating and minutes together,
stored in SQLite for use in feature engineering.

Usage:
    from src.DataProviders.LineupDataProvider import fetch_lineup_data
"""

import random
import sqlite3
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguedashlineups

BASE_DIR = Path(__file__).resolve().parents[2]
LINEUP_DB_PATH = BASE_DIR / "Data" / "LineupData.sqlite"

MIN_DELAY = 1
MAX_DELAY = 3
MAX_RETRIES = 3


def fetch_lineup_data(
    season: str,
    group_quantity: int = 5,
    season_type: str = "Regular Season",
    measure_type: str = "Advanced",
) -> pd.DataFrame:
    """Fetch lineup stats for a season from NBA Stats API.

    Parameters
    ----------
    season : str
        NBA season string, e.g. "2023-24".
    group_quantity : int
        Number of players in lineup group (default 5 for full lineups).
    season_type : str
        "Regular Season" or "Playoffs".
    measure_type : str
        "Base" or "Advanced" (Advanced includes NET_RATING).

    Returns
    -------
    pd.DataFrame
        Lineup stats including GROUP_SET, GROUP_ID, GROUP_NAME, TEAM_ID,
        TEAM_ABBREVIATION, GP, W, L, W_PCT, MIN, NET_RATING, etc.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            lineups = leaguedashlineups.LeagueDashLineups(
                season=season,
                season_type_all_star=season_type,
                group_quantity=group_quantity,
                measure_type_detailed_defense=measure_type,
            )
            df = lineups.get_data_frames()[0]
            if not df.empty:
                return df
        except Exception as exc:
            print(f"  Attempt {attempt}/{MAX_RETRIES} failed: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(MIN_DELAY + random.random() * (MAX_DELAY - MIN_DELAY))

    return pd.DataFrame()


def save_lineup_data(df: pd.DataFrame, season: str, db_path: Path = LINEUP_DB_PATH) -> None:
    """Save lineup data to SQLite, one table per season."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        df.to_sql(season, con, if_exists="replace", index=False)
    print(f"  Saved {len(df)} lineup rows to table '{season}'")


def load_lineup_data(season: str, db_path: Path = LINEUP_DB_PATH) -> pd.DataFrame:
    """Load lineup data for a season from SQLite."""
    if not db_path.exists():
        return pd.DataFrame()
    with sqlite3.connect(db_path) as con:
        cursor = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (season,),
        )
        if cursor.fetchone() is None:
            return pd.DataFrame()
        return pd.read_sql_query(f'SELECT * FROM "{season}"', con)


def load_all_lineup_data(db_path: Path = LINEUP_DB_PATH) -> pd.DataFrame:
    """Load and concatenate lineup data from all season tables."""
    if not db_path.exists():
        return pd.DataFrame()
    with sqlite3.connect(db_path) as con:
        cursor = con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
    if not tables:
        return pd.DataFrame()

    frames = []
    for table in tables:
        df = load_lineup_data(table, db_path)
        if not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
