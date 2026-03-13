"""
Player-level feature engineering for NBA game prediction.

Derives features from player game logs, including:
- Top-5 player weighted stats per team (minutes-weighted)
- Star player availability flags
- Team strength differentials

These features are merged into the game-level dataset by team name and date.
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
PLAYER_DB_PATH = BASE_DIR / "Data" / "PlayerGameLogs.sqlite"
LINEUP_DB_PATH = BASE_DIR / "Data" / "LineupData.sqlite"

# Stats to aggregate for top-5 players (weighted by minutes)
# Pruned: removed AST, STL (low importance, redundant with diffs)
TOP_PLAYER_STATS = ["PTS", "REB", "PLUS_MINUS", "BLK"]

# Rolling window for player-level aggregation
PLAYER_ROLLING_WINDOW = 10

# NBA team name normalization: nba_api uses full names, dataset may differ
TEAM_NAME_MAP = {
    "LA Clippers": "Los Angeles Clippers",
}


def _normalize_team_name(name: str) -> str:
    """Normalize team names between nba_api and dataset conventions."""
    return TEAM_NAME_MAP.get(name, name)


def _load_all_from_db(db_path: Path) -> pd.DataFrame:
    """Load and concatenate all tables from a SQLite database."""
    if not db_path.exists():
        return pd.DataFrame()
    with sqlite3.connect(db_path) as con:
        cursor = con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
    if not tables:
        return pd.DataFrame()

    frames = []
    with sqlite3.connect(db_path) as con:
        for table in tables:
            df = pd.read_sql_query(f'SELECT * FROM "{table}"', con)
            if not df.empty:
                frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _compute_team_top_player_stats(player_logs: pd.DataFrame) -> pd.DataFrame:
    """Compute per-team per-date top-5 player weighted stats.

    For each game date and team, identifies the top 5 players by minutes
    played (rolling average) and computes minutes-weighted averages of
    key stats. Uses a rolling window so values are based on recent
    performance, not just the current game.

    Parameters
    ----------
    player_logs : pd.DataFrame
        Player game logs with GAME_DATE, TEAM_NAME, PLAYER_ID, MIN, PTS, etc.

    Returns
    -------
    pd.DataFrame
        Columns: Date, TEAM_NAME, Top5_WtAvg_PTS, Top5_WtAvg_AST, etc.,
        Star_Player_Available (1 if top scorer played).
    """
    if player_logs.empty:
        return pd.DataFrame()

    df = player_logs.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    df = df.dropna(subset=["GAME_DATE"])
    df = df.sort_values("GAME_DATE")

    # Convert MIN to numeric (can be "MM:SS" format or float)
    if df["MIN"].dtype == object:
        def parse_min(val):
            if pd.isna(val):
                return 0.0
            val = str(val)
            if ":" in val:
                parts = val.split(":")
                return float(parts[0]) + float(parts[1]) / 60.0
            try:
                return float(val)
            except ValueError:
                return 0.0
        df["MIN"] = df["MIN"].apply(parse_min)
    else:
        df["MIN"] = pd.to_numeric(df["MIN"], errors="coerce").fillna(0.0)

    for stat in TOP_PLAYER_STATS:
        df[stat] = pd.to_numeric(df[stat], errors="coerce").fillna(0.0)

    df["TEAM_NAME"] = df["TEAM_NAME"].apply(_normalize_team_name)

    # Build rolling player averages per team-date
    # For each team+date, compute features from that team's player history
    results = []
    team_player_history: dict[str, dict[int, list[dict]]] = {}  # team -> player_id -> [game_dicts]

    # Group by date for chronological processing
    for game_date, day_group in df.groupby("GAME_DATE"):
        # Process each team playing this day
        for team_name, team_group in day_group.groupby("TEAM_NAME"):
            team_hist = team_player_history.setdefault(team_name, {})

            # Compute pre-game features from history
            if team_hist:
                # Get recent average stats for each player on this team
                player_avgs = []
                for pid, games in team_hist.items():
                    recent = games[-PLAYER_ROLLING_WINDOW:]
                    avg_min = np.mean([g["MIN"] for g in recent])
                    avg_stats = {}
                    for stat in TOP_PLAYER_STATS:
                        avg_stats[stat] = np.mean([g[stat] for g in recent])
                    player_avgs.append({
                        "PLAYER_ID": pid,
                        "AVG_MIN": avg_min,
                        **avg_stats,
                    })

                # Sort by average minutes, take top 5
                player_avgs.sort(key=lambda x: x["AVG_MIN"], reverse=True)
                top5 = player_avgs[:5]

                if top5:
                    total_min = sum(p["AVG_MIN"] for p in top5)
                    row_data = {"Date": game_date, "TEAM_NAME": team_name}

                    for stat in TOP_PLAYER_STATS:
                        if total_min > 0:
                            weighted = sum(
                                p["AVG_MIN"] * p[stat] for p in top5
                            ) / total_min
                        else:
                            weighted = 0.0
                        row_data[f"Top5_WtAvg_{stat}"] = weighted

                    # Star player available: is the top scorer (by avg PTS) in today's game?
                    top_scorer_id = max(player_avgs, key=lambda x: x["PTS"])["PLAYER_ID"]
                    today_player_ids = set(team_group["PLAYER_ID"].values)
                    row_data["Star_Player_Available"] = int(top_scorer_id in today_player_ids)

                    results.append(row_data)

            # Update history with today's game stats
            for _, prow in team_group.iterrows():
                pid = prow["PLAYER_ID"]
                game_dict = {"MIN": prow["MIN"]}
                for stat in TOP_PLAYER_STATS:
                    game_dict[stat] = prow[stat]
                team_hist.setdefault(pid, []).append(game_dict)

    return pd.DataFrame(results) if results else pd.DataFrame()



def _compute_lineup_features(lineup_data: pd.DataFrame) -> pd.DataFrame:
    """Compute lineup-based features per team per season.

    Since lineup data is season-level (not per-game), these features
    are static within a season and merged by team name.

    Parameters
    ----------
    lineup_data : pd.DataFrame
        Lineup stats from LeagueDashLineups endpoint.

    Returns
    -------
    pd.DataFrame
        Columns: TEAM_ABBREVIATION, Best_Lineup_NetRtg, Best_Lineup_MIN,
        Lineup_Continuity (minutes of most-used lineup / total team minutes).
    """
    if lineup_data.empty:
        return pd.DataFrame()

    df = lineup_data.copy()

    # Ensure numeric columns
    for col in ["MIN", "NET_RATING", "GP", "W", "L"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    if "TEAM_ABBREVIATION" not in df.columns:
        return pd.DataFrame()

    results = []
    for team, team_group in df.groupby("TEAM_ABBREVIATION"):
        if team_group.empty:
            continue

        # Best lineup by minutes played (most-used = starter lineup proxy)
        best_idx = team_group["MIN"].idxmax()
        best_row = team_group.loc[best_idx]

        best_net_rtg = best_row.get("NET_RATING", 0.0)
        best_min = best_row.get("MIN", 0.0)
        total_min = team_group["MIN"].sum()
        continuity = best_min / total_min if total_min > 0 else 0.0

        results.append({
            "TEAM_ABBREVIATION": team,
            "Best_Lineup_NetRtg": best_net_rtg,
            "Best_Lineup_MIN": best_min,
            "Lineup_Continuity": continuity,
        })

    return pd.DataFrame(results) if results else pd.DataFrame()


def compute_player_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add player-level features to the game dataset.

    Loads player game logs and injury data from SQLite, computes derived
    features, and merges them into the dataset. Uses pre-game values only
    (no data leakage).

    Parameters
    ----------
    df : pd.DataFrame
        Game dataset with TEAM_NAME, TEAM_NAME.1, Date columns.

    Returns
    -------
    pd.DataFrame
        Input dataframe with player feature columns added.
    """
    df = df.copy()
    df["_date_parsed"] = pd.to_datetime(df["Date"], errors="coerce")
    # Strip timezone info to avoid merge mismatches
    if df["_date_parsed"].dt.tz is not None:
        df["_date_parsed"] = df["_date_parsed"].dt.tz_localize(None)

    # Load player game logs
    player_logs = _load_all_from_db(PLAYER_DB_PATH)
    if player_logs.empty:
        print("No player game log data found. Skipping player features.")
        df.drop(columns=["_date_parsed"], inplace=True)
        return df

    print(f"  Loaded {len(player_logs)} player game log rows")

    # Compute top-5 player stats per team per date
    top5_stats = _compute_team_top_player_stats(player_logs)

    if not top5_stats.empty:
        top5_stats["Date"] = pd.to_datetime(top5_stats["Date"], errors="coerce")
        if top5_stats["Date"].dt.tz is not None:
            top5_stats["Date"] = top5_stats["Date"].dt.tz_localize(None)

        # Merge for home team
        home_merge = top5_stats.rename(columns={
            col: f"{col}_Home" for col in top5_stats.columns
            if col not in ("Date", "TEAM_NAME")
        })
        df = df.merge(
            home_merge,
            left_on=["_date_parsed", "TEAM_NAME"],
            right_on=["Date", "TEAM_NAME"],
            how="left",
            suffixes=("", "_player_home"),
        )
        # Drop duplicate Date column from merge
        if "Date_player_home" in df.columns:
            df.drop(columns=["Date_player_home"], inplace=True)

        # Merge for away team
        away_merge = top5_stats.rename(columns={
            col: f"{col}_Away" for col in top5_stats.columns
            if col not in ("Date", "TEAM_NAME")
        })
        df = df.merge(
            away_merge,
            left_on=["_date_parsed", "TEAM_NAME.1"],
            right_on=["Date", "TEAM_NAME"],
            how="left",
            suffixes=("", "_player_away"),
        )
        if "TEAM_NAME_player_away" in df.columns:
            df.drop(columns=["TEAM_NAME_player_away"], inplace=True)
        if "Date_player_away" in df.columns:
            df.drop(columns=["Date_player_away"], inplace=True)

        # Compute differentials
        for stat in TOP_PLAYER_STATS:
            home_col = f"Top5_WtAvg_{stat}_Home"
            away_col = f"Top5_WtAvg_{stat}_Away"
            if home_col in df.columns and away_col in df.columns:
                df[f"Top5_{stat}_Diff"] = (
                    df[home_col].astype(float) - df[away_col].astype(float)
                )

        # Drop raw Home/Away columns for REB and BLK (keep only diffs)
        # Keep raw Home/Away only for PTS and PLUS_MINUS (top importance)
        drop_raw = []
        for stat in ["REB", "BLK"]:
            for side in ["Home", "Away"]:
                col = f"Top5_WtAvg_{stat}_{side}"
                if col in df.columns:
                    drop_raw.append(col)
        if drop_raw:
            df.drop(columns=drop_raw, inplace=True)

        print(f"  Added top-5 player stat features (pruned)")

    df.drop(columns=["_date_parsed"], inplace=True)
    return df
