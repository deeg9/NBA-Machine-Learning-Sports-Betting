"""
Injury-based features for NBA game prediction.

Computes player availability counts from the InjuryData.sqlite database
populated by PlayerDataProvider.fetch_injury_data().
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

INJURY_DB_PATH = Path(__file__).resolve().parents[2] / "Data" / "InjuryData.sqlite"

# Statuses that mean a player is definitely out
OUT_STATUSES = {"Out", "Suspension"}
# Statuses that mean a player is uncertain (game-time decision)
GTD_STATUSES = {"Day-To-Day", "Questionable", "Doubtful"}


def _load_all_injury_snapshots(db_path: Path = INJURY_DB_PATH) -> pd.DataFrame:
    """Load all injury data from all season tables."""
    if not db_path.exists():
        return pd.DataFrame()
    with sqlite3.connect(db_path) as con:
        tables = [
            r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
    if not tables:
        return pd.DataFrame()

    frames = []
    for table in tables:
        with sqlite3.connect(db_path) as con:
            df = pd.read_sql_query(f'SELECT * FROM "{table}"', con)
            if not df.empty:
                frames.append(df)
    if not frames:
        return pd.DataFrame()

    injury_df = pd.concat(frames, ignore_index=True)
    injury_df["Date"] = pd.to_datetime(injury_df["Date"], errors="coerce")
    return injury_df


def compute_injury_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add injury count features to the dataset.

    Features added:
        Injuries_Out_Home  — players with "Out"/"Suspension" status for home team
        Injuries_Out_Away  — same for away team
        Injuries_GTD_Home  — players with uncertain status for home team
        Injuries_GTD_Away  — same for away team
        Injury_Advantage   — Out_Away - Out_Home (positive = home team healthier)

    For each game, uses the most recent injury snapshot on or before the game
    date. If no injury data exists, all counts default to 0.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with TEAM_NAME, TEAM_NAME.1, Date columns.

    Returns
    -------
    pd.DataFrame
        Input dataframe with injury feature columns added.
    """
    df = df.copy()
    df["_date_parsed"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("_date_parsed").reset_index(drop=True)

    n = len(df)
    out_home = np.zeros(n, dtype=np.float64)
    out_away = np.zeros(n, dtype=np.float64)
    gtd_home = np.zeros(n, dtype=np.float64)
    gtd_away = np.zeros(n, dtype=np.float64)

    injury_df = _load_all_injury_snapshots()

    if injury_df.empty or "Status" not in injury_df.columns:
        # No injury data — leave all zeros
        df["Injuries_Out_Home"] = out_home
        df["Injuries_Out_Away"] = out_away
        df["Injuries_GTD_Home"] = gtd_home
        df["Injuries_GTD_Away"] = gtd_away
        df["Injury_Advantage"] = out_away - out_home
        df.drop(columns=["_date_parsed"], inplace=True)
        return df

    # Sort injury data by date for efficient lookup
    # Normalize to tz-naive for comparison with game dates
    injury_df["Date"] = injury_df["Date"].dt.tz_localize(None) if injury_df["Date"].dt.tz is not None else injury_df["Date"]
    injury_df = injury_df.sort_values("Date").reset_index(drop=True)

    # Get unique injury snapshot dates
    injury_dates = injury_df["Date"].dropna().unique()
    injury_dates = np.sort(injury_dates)

    for idx in range(n):
        row = df.iloc[idx]
        game_date = row["_date_parsed"]
        home = row["TEAM_NAME"]
        away = row["TEAM_NAME.1"]

        if pd.isna(game_date) or len(injury_dates) == 0:
            continue

        # Find most recent injury snapshot on or before game date
        mask = injury_dates <= game_date
        if not mask.any():
            continue

        closest_date = injury_dates[mask][-1]

        # Allow up to 14 days gap — older snapshots are stale
        if (game_date - pd.Timestamp(closest_date)).days > 14:
            continue

        snapshot = injury_df[injury_df["Date"] == closest_date]

        # Count injuries for each team
        for team, out_arr, gtd_arr in [
            (home, out_home, gtd_home),
            (away, out_away, gtd_away),
        ]:
            team_injuries = snapshot[snapshot["Team"] == team]
            if team_injuries.empty:
                continue
            out_arr[idx] = team_injuries["Status"].isin(OUT_STATUSES).sum()
            gtd_arr[idx] = team_injuries["Status"].isin(GTD_STATUSES).sum()

    df["Injuries_Out_Home"] = out_home
    df["Injuries_Out_Away"] = out_away
    df["Injuries_GTD_Home"] = gtd_home
    df["Injuries_GTD_Away"] = gtd_away
    df["Injury_Advantage"] = out_away - out_home

    df.drop(columns=["_date_parsed"], inplace=True)
    return df


def compute_live_injury_counts(home_team: str, away_team: str,
                                injury_df: pd.DataFrame) -> dict:
    """Compute injury feature values from a live injury snapshot.

    Used by dry_run_today.py for games not yet in the dataset.

    Parameters
    ----------
    home_team : str
        Home team name.
    away_team : str
        Away team name.
    injury_df : pd.DataFrame
        Current injury report (from fetch_injury_data()).

    Returns
    -------
    dict
        Feature name → value mapping.
    """
    result = {
        "Injuries_Out_Home": 0,
        "Injuries_Out_Away": 0,
        "Injuries_GTD_Home": 0,
        "Injuries_GTD_Away": 0,
        "Injury_Advantage": 0,
    }

    if injury_df.empty or "Status" not in injury_df.columns:
        return result

    for team, suffix in [(home_team, "Home"), (away_team, "Away")]:
        team_inj = injury_df[injury_df["Team"] == team]
        if team_inj.empty:
            continue
        result[f"Injuries_Out_{suffix}"] = int(team_inj["Status"].isin(OUT_STATUSES).sum())
        result[f"Injuries_GTD_{suffix}"] = int(team_inj["Status"].isin(GTD_STATUSES).sum())

    result["Injury_Advantage"] = result["Injuries_Out_Away"] - result["Injuries_Out_Home"]
    return result
