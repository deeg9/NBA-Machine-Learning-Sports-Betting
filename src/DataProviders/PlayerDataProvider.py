"""
Provider for player game logs and injury data.

Uses `nba_api` for per-player per-game stats and `nbainjuries` for
historical injury reports. Data is stored in SQLite databases under Data/.

Usage (as a module):
    from src.DataProviders.PlayerDataProvider import (
        fetch_player_game_logs,
        fetch_injury_data,
    )
"""

import random
import sqlite3
import time
from pathlib import Path

import pandas as pd
import requests
from nba_api.stats.endpoints import playergamelogs
from nba_api.stats.static import teams as nba_teams

BASE_DIR = Path(__file__).resolve().parents[2]
PLAYER_DB_PATH = BASE_DIR / "Data" / "PlayerGameLogs.sqlite"
INJURY_DB_PATH = BASE_DIR / "Data" / "InjuryData.sqlite"

ESPN_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"

MIN_DELAY = 1
MAX_DELAY = 3
MAX_RETRIES = 3


def get_all_team_ids() -> dict:
    """Return {team_id: full_name} for all NBA teams."""
    return {t["id"]: t["full_name"] for t in nba_teams.get_teams()}


def fetch_player_game_logs(season: str, season_type: str = "Regular Season") -> pd.DataFrame:
    """Fetch player game logs for an entire season.

    Parameters
    ----------
    season : str
        NBA season string, e.g. "2023-24".
    season_type : str
        "Regular Season" or "Playoffs".

    Returns
    -------
    pd.DataFrame
        Player game logs with columns: SEASON_YEAR, PLAYER_ID, PLAYER_NAME,
        TEAM_ID, TEAM_ABBREVIATION, TEAM_NAME, GAME_ID, GAME_DATE, MATCHUP,
        WL, MIN, FGM, FGA, FG_PCT, FG3M, FG3A, FG3_PCT, FTM, FTA, FT_PCT,
        OREB, DREB, REB, AST, TOV, STL, BLK, BLKA, PF, PFD, PTS, PLUS_MINUS,
        NBA_FANTASY_PTS, DD2, TD3.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logs = playergamelogs.PlayerGameLogs(
                season_nullable=season,
                season_type_nullable=season_type,
            )
            df = logs.get_data_frames()[0]
            if not df.empty:
                return df
        except Exception as exc:
            print(f"  Attempt {attempt}/{MAX_RETRIES} failed: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(MIN_DELAY + random.random() * (MAX_DELAY - MIN_DELAY))

    return pd.DataFrame()


def save_player_game_logs(df: pd.DataFrame, season: str, db_path: Path = PLAYER_DB_PATH) -> None:
    """Save player game logs to SQLite, one table per season.

    Parameters
    ----------
    df : pd.DataFrame
        Player game logs as returned by fetch_player_game_logs.
    season : str
        Season label used as the table name (e.g. "2023-24").
    db_path : Path
        Path to the SQLite database file.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        df.to_sql(season, con, if_exists="replace", index=False)
    print(f"  Saved {len(df)} player game log rows to table '{season}'")


def load_player_game_logs(season: str, db_path: Path = PLAYER_DB_PATH) -> pd.DataFrame:
    """Load player game logs for a season from SQLite.

    Returns empty DataFrame if the table does not exist.
    """
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


def load_all_player_game_logs(db_path: Path = PLAYER_DB_PATH) -> pd.DataFrame:
    """Load and concatenate player game logs from all season tables."""
    if not db_path.exists():
        return pd.DataFrame()
    with sqlite3.connect(db_path) as con:
        cursor = con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
    if not tables:
        return pd.DataFrame()

    frames = []
    for table in tables:
        df = load_player_game_logs(table, db_path)
        if not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_injury_data(season=None) -> pd.DataFrame:
    """Fetch current NBA injury data from ESPN's public API.

    Returns the current injury report snapshot. Since ESPN only serves the
    current report (not historical), this should be called regularly and
    appended to build a history.

    Parameters
    ----------
    season : str or None
        Currently unused (ESPN returns current data only), kept for API
        compatibility with the rest of the pipeline.

    Returns
    -------
    pd.DataFrame
        Injury data with columns: Player, Team, Date, Status, Injury_Type,
        Injury_Detail, Return_Date.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(ESPN_INJURIES_URL, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as exc:
            print(f"  Attempt {attempt}/{MAX_RETRIES} failed: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(MIN_DELAY + random.random() * (MAX_DELAY - MIN_DELAY))
            else:
                return pd.DataFrame()

    report_date = data.get("timestamp", "")
    rows = []
    for team_entry in data.get("injuries", []):
        team_name = team_entry.get("displayName", "")
        for inj in team_entry.get("injuries", []):
            athlete = inj.get("athlete", {})
            details = inj.get("details", {})
            rows.append({
                "Player": athlete.get("displayName", ""),
                "Team": team_name,
                "Date": inj.get("date", report_date),
                "Status": inj.get("status", ""),
                "Injury_Type": details.get("type", ""),
                "Injury_Detail": details.get("detail", ""),
                "Return_Date": details.get("returnDate", ""),
                "Position": athlete.get("position", {}).get("abbreviation", ""),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def save_injury_data(df: pd.DataFrame, season: str, db_path: Path = INJURY_DB_PATH) -> None:
    """Save injury data to SQLite, one table per season."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        df.to_sql(season, con, if_exists="replace", index=False)
    print(f"  Saved {len(df)} injury rows to table '{season}'")


def load_injury_data(season: str, db_path: Path = INJURY_DB_PATH) -> pd.DataFrame:
    """Load injury data for a season from SQLite."""
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


def load_all_injury_data(db_path: Path = INJURY_DB_PATH) -> pd.DataFrame:
    """Load and concatenate injury data from all season tables."""
    if not db_path.exists():
        return pd.DataFrame()
    with sqlite3.connect(db_path) as con:
        cursor = con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
    if not tables:
        return pd.DataFrame()

    frames = []
    for table in tables:
        df = load_injury_data(table, db_path)
        if not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
