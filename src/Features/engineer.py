"""
Feature engineering module for NBA game prediction.

Adds enhanced features to the existing dataset including Elo ratings,
rolling averages, home/away splits, pace-adjusted stats, back-to-back
detection, and optionally player-level features. Can be used as an
importable module or run standalone via:

    python -m src.Features.engineer
    python -m src.Features.engineer --include-players
"""

import argparse
import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ELO_START = 1500
ELO_K = 20
ELO_HOME_ADV = 100
ELO_SEASON_CARRY = 2 / 3

ROLLING_WINDOW = 10
ROLLING_STATS = ["PTS", "FG_PCT", "FG3_PCT", "REB", "AST", "TOV", "PLUS_MINUS"]

PACE_NORM_STATS = ["PTS", "AST", "TOV", "STL"]

DB_PATH = Path(__file__).resolve().parents[2] / "Data" / "dataset.sqlite"
SOURCE_TABLE = "dataset_2012-26"
DEST_TABLE = "dataset_enhanced"
DEST_TABLE_V2 = "dataset_enhanced_v2"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_season(date: pd.Timestamp) -> str:
    """Return a season label like '2023-24' from a game date.

    The NBA season straddles two calendar years. Games from October onward
    belong to the season that *starts* in that year; games before July belong
    to the season that *started* the previous year.
    """
    if date.month >= 7:
        return f"{date.year}-{str(date.year + 1)[-2:]}"
    return f"{date.year - 1}-{str(date.year)[-2:]}"


def _expected_score(rating_a: float, rating_b: float) -> float:
    """Compute expected score for player A in an Elo system."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


# ---------------------------------------------------------------------------
# 1. Elo Ratings
# ---------------------------------------------------------------------------

def compute_elo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Elo rating columns: Elo_Home, Elo_Away, Elo_Diff.

    Maintains a running Elo rating for every team. Ratings start at 1500 and
    are updated after each game (K=20, home advantage=100 points). Between
    seasons, ratings regress toward the mean (2/3 carry, 1/3 revert to 1500).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset sorted chronologically with columns TEAM_NAME, TEAM_NAME.1,
        Date, and Home-Team-Win.

    Returns
    -------
    pd.DataFrame
        Input dataframe with Elo_Home, Elo_Away, and Elo_Diff columns added.
    """
    df = df.copy()
    df["_date_parsed"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("_date_parsed").reset_index(drop=True)

    ratings: dict[str, float] = {}
    prev_season: str | None = None

    elo_home_vals = np.zeros(len(df), dtype=np.float64)
    elo_away_vals = np.zeros(len(df), dtype=np.float64)

    for idx in range(len(df)):
        row = df.iloc[idx]
        date = row["_date_parsed"]
        home = row["TEAM_NAME"]
        away = row["TEAM_NAME.1"]

        # Handle missing date gracefully
        if pd.isna(date):
            elo_home_vals[idx] = ratings.get(home, ELO_START)
            elo_away_vals[idx] = ratings.get(away, ELO_START)
            continue

        season = _extract_season(date)

        # Season carryover: regress ratings toward the mean
        if prev_season is not None and season != prev_season:
            for team in list(ratings.keys()):
                ratings[team] = (
                    ELO_SEASON_CARRY * ratings[team]
                    + (1 - ELO_SEASON_CARRY) * ELO_START
                )
        prev_season = season

        # Initialize teams we have not seen yet
        ratings.setdefault(home, ELO_START)
        ratings.setdefault(away, ELO_START)

        home_elo = ratings[home]
        away_elo = ratings[away]

        # Record *pre-game* Elo for features
        elo_home_vals[idx] = home_elo
        elo_away_vals[idx] = away_elo

        # Update ratings based on result
        home_win = row.get("Home-Team-Win")
        if pd.notna(home_win):
            actual = float(home_win)
            expected = _expected_score(home_elo + ELO_HOME_ADV, away_elo)
            ratings[home] = home_elo + ELO_K * (actual - expected)
            ratings[away] = away_elo + ELO_K * ((1 - actual) - (1 - expected))

    df["Elo_Home"] = elo_home_vals
    df["Elo_Away"] = elo_away_vals
    df["Elo_Diff"] = df["Elo_Home"] - df["Elo_Away"]
    df.drop(columns=["_date_parsed"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# 2. Rolling Averages
# ---------------------------------------------------------------------------

def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling 10-game average columns for key statistics.

    For each stat in ROLLING_STATS, adds Roll10_<stat>_Home and
    Roll10_<stat>_Away. Uses an expanding window for the first games of
    a team's history so that early games still receive values.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with Date, TEAM_NAME, TEAM_NAME.1, and stat columns.

    Returns
    -------
    pd.DataFrame
        Input dataframe with rolling average columns added.
    """
    df = df.copy()
    df["_date_parsed"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("_date_parsed").reset_index(drop=True)

    # Suffix mapping: home stats have no suffix, away stats have ".1"
    home_suffix = ""
    away_suffix = ".1"

    # Pre-allocate result arrays
    result_arrays: dict[str, np.ndarray] = {}
    for stat in ROLLING_STATS:
        result_arrays[f"Roll10_{stat}_Home"] = np.full(len(df), np.nan)
        result_arrays[f"Roll10_{stat}_Away"] = np.full(len(df), np.nan)

    # team -> list of stat values (one dict per game played)
    team_history: dict[str, list[dict[str, float]]] = {}

    for idx in range(len(df)):
        row = df.iloc[idx]
        home = row["TEAM_NAME"]
        away = row["TEAM_NAME.1"]

        # Record rolling averages *before* updating history (pre-game)
        for stat in ROLLING_STATS:
            # Home team rolling average
            if home in team_history and len(team_history[home]) > 0:
                recent = team_history[home][-ROLLING_WINDOW:]
                vals = [g[stat] for g in recent if not np.isnan(g.get(stat, np.nan))]
                if vals:
                    result_arrays[f"Roll10_{stat}_Home"][idx] = np.mean(vals)

            # Away team rolling average
            if away in team_history and len(team_history[away]) > 0:
                recent = team_history[away][-ROLLING_WINDOW:]
                vals = [g[stat] for g in recent if not np.isnan(g.get(stat, np.nan))]
                if vals:
                    result_arrays[f"Roll10_{stat}_Away"][idx] = np.mean(vals)

        # Update history with this game's stats
        home_game_stats: dict[str, float] = {}
        away_game_stats: dict[str, float] = {}
        for stat in ROLLING_STATS:
            home_col = stat + home_suffix
            away_col = stat + away_suffix
            home_val = row.get(home_col, np.nan)
            away_val = row.get(away_col, np.nan)
            home_game_stats[stat] = float(home_val) if pd.notna(home_val) else np.nan
            away_game_stats[stat] = float(away_val) if pd.notna(away_val) else np.nan

        team_history.setdefault(home, []).append(home_game_stats)
        team_history.setdefault(away, []).append(away_game_stats)

    for col, arr in result_arrays.items():
        df[col] = arr

    df.drop(columns=["_date_parsed"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# 3. Home / Away Splits
# ---------------------------------------------------------------------------

def compute_home_away_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Add win-percentage split columns.

    Home_WinPct_Split: home team's home win% minus overall win% (positive
    means team is better at home than average). Away_WinPct_Split: away
    team's away win% minus overall win% (typically negative).

    Uses an expanding window so values are based on games played so far.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with TEAM_NAME, TEAM_NAME.1, Home-Team-Win, Date.

    Returns
    -------
    pd.DataFrame
        Input dataframe with Home_WinPct_Split and Away_WinPct_Split added.
    """
    df = df.copy()
    df["_date_parsed"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("_date_parsed").reset_index(drop=True)

    # Track per-team: total wins/games, home wins/home games, away wins/away games
    team_total: dict[str, dict[str, int]] = {}  # {team: {wins, games}}
    team_home: dict[str, dict[str, int]] = {}
    team_away: dict[str, dict[str, int]] = {}

    home_split = np.full(len(df), np.nan)
    away_split = np.full(len(df), np.nan)

    for idx in range(len(df)):
        row = df.iloc[idx]
        home = row["TEAM_NAME"]
        away_team = row["TEAM_NAME.1"]
        win = row.get("Home-Team-Win")

        # Compute pre-game splits
        ht = team_total.get(home)
        hh = team_home.get(home)
        if ht and ht["games"] > 0 and hh and hh["games"] > 0:
            overall_pct = ht["wins"] / ht["games"]
            home_pct = hh["wins"] / hh["games"]
            home_split[idx] = home_pct - overall_pct

        at = team_total.get(away_team)
        aa = team_away.get(away_team)
        if at and at["games"] > 0 and aa and aa["games"] > 0:
            overall_pct = at["wins"] / at["games"]
            away_pct = aa["wins"] / aa["games"]
            away_split[idx] = away_pct - overall_pct

        # Update records
        if pd.notna(win):
            home_win = int(win)
            away_win = 1 - home_win

            team_total.setdefault(home, {"wins": 0, "games": 0})
            team_total[home]["wins"] += home_win
            team_total[home]["games"] += 1

            team_home.setdefault(home, {"wins": 0, "games": 0})
            team_home[home]["wins"] += home_win
            team_home[home]["games"] += 1

            team_total.setdefault(away_team, {"wins": 0, "games": 0})
            team_total[away_team]["wins"] += away_win
            team_total[away_team]["games"] += 1

            team_away.setdefault(away_team, {"wins": 0, "games": 0})
            team_away[away_team]["wins"] += away_win
            team_away[away_team]["games"] += 1

    df["Home_WinPct_Split"] = home_split
    df["Away_WinPct_Split"] = away_split
    df.drop(columns=["_date_parsed"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# 4. Pace-Adjusted Stats
# ---------------------------------------------------------------------------

def compute_pace_adjusted(df: pd.DataFrame) -> pd.DataFrame:
    """Add pace and per-100-possessions columns.

    Pace is estimated as FGA + 0.44*FTA - OREB + TOV. Stats (PTS, AST, TOV,
    STL) are then normalized to a per-100-possessions basis.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with FGA, FTA, OREB, TOV, PTS, AST, STL columns (home and
        away variants).

    Returns
    -------
    pd.DataFrame
        Input dataframe with Pace_Home, Pace_Away, and per-100 stat columns.
    """
    df = df.copy()

    # Home pace
    df["Pace_Home"] = (
        df["FGA"].astype(float)
        + 0.44 * df["FTA"].astype(float)
        - df["OREB"].astype(float)
        + df["TOV"].astype(float)
    )

    # Away pace
    df["Pace_Away"] = (
        df["FGA.1"].astype(float)
        + 0.44 * df["FTA.1"].astype(float)
        - df["OREB.1"].astype(float)
        + df["TOV.1"].astype(float)
    )

    # Per-100 possessions normalization
    for stat in PACE_NORM_STATS:
        home_col = stat  # home stat column
        away_col = stat + ".1"

        df[f"{stat}_per100_Home"] = np.where(
            df["Pace_Home"] > 0,
            df[home_col].astype(float) / df["Pace_Home"] * 100,
            np.nan,
        )
        df[f"{stat}_per100_Away"] = np.where(
            df["Pace_Away"] > 0,
            df[away_col].astype(float) / df["Pace_Away"] * 100,
            np.nan,
        )

    return df


# ---------------------------------------------------------------------------
# 5. Back-to-Back Detection
# ---------------------------------------------------------------------------

def compute_b2b_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add back-to-back game indicators.

    Is_B2B_Home / Is_B2B_Away: 1 if the team has only 1 day of rest.
    Both_B2B: 1 if both teams are on a back-to-back.
    B2B_Advantage: +1 if only away is B2B (home has advantage), -1 if only
    home is B2B, 0 otherwise.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with Days-Rest-Home and Days-Rest-Away columns.

    Returns
    -------
    pd.DataFrame
        Input dataframe with B2B feature columns added.
    """
    df = df.copy()

    rest_home = pd.to_numeric(df["Days-Rest-Home"], errors="coerce")
    rest_away = pd.to_numeric(df["Days-Rest-Away"], errors="coerce")

    df["Is_B2B_Home"] = (rest_home == 1).astype(int)
    df["Is_B2B_Away"] = (rest_away == 1).astype(int)
    df["Both_B2B"] = ((df["Is_B2B_Home"] == 1) & (df["Is_B2B_Away"] == 1)).astype(int)

    # B2B_Advantage from the home team's perspective
    df["B2B_Advantage"] = 0
    df.loc[(df["Is_B2B_Away"] == 1) & (df["Is_B2B_Home"] == 0), "B2B_Advantage"] = 1
    df.loc[(df["Is_B2B_Home"] == 1) & (df["Is_B2B_Away"] == 0), "B2B_Advantage"] = -1

    # Raw rest differential (captures gradient beyond binary B2B)
    df["Rest_Advantage"] = rest_home - rest_away

    return df


# ---------------------------------------------------------------------------
# 6. Main orchestration
# ---------------------------------------------------------------------------

def enhance_dataset(df: pd.DataFrame, include_players: bool = False) -> pd.DataFrame:
    """Apply all feature engineering steps to the dataset.

    Runs each feature engineering function in sequence and returns the
    enhanced DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset loaded from the SQLite database.
    include_players : bool
        If True, also compute player-level features (top-5 player stats,
        injury impact, lineup data). Requires player data in SQLite.

    Returns
    -------
    pd.DataFrame
        Enhanced dataset with all new feature columns.
    """
    df = compute_elo_features(df)
    df = compute_rolling_features(df)
    df = compute_home_away_splits(df)
    df = compute_pace_adjusted(df)
    df = compute_b2b_features(df)

    if include_players:
        from src.Features.player_features import compute_player_features

        print("Computing player-level features ...")
        df = compute_player_features(df)

    return df


def main() -> None:
    """CLI entry point: read from SQLite, enhance, and save back."""
    parser = argparse.ArgumentParser(
        description="Run feature engineering on NBA game dataset."
    )
    parser.add_argument(
        "--include-players",
        action="store_true",
        help="Include player-level features (requires player data in SQLite).",
    )
    args = parser.parse_args()

    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {DB_PATH}")

    dest_table = DEST_TABLE_V2 if args.include_players else DEST_TABLE

    print(f"Reading from {DB_PATH} table '{SOURCE_TABLE}' ...")
    conn = sqlite3.connect(str(DB_PATH))
    try:
        df = pd.read_sql_query(f'SELECT * FROM "{SOURCE_TABLE}"', conn)
    finally:
        conn.close()

    print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")
    print("Running feature engineering ...")
    df_enhanced = enhance_dataset(df, include_players=args.include_players)
    new_cols = set(df_enhanced.columns) - set(df.columns)
    print(f"Added {len(new_cols)} new columns: {sorted(new_cols)}")

    print(f"Saving to table '{dest_table}' ...")
    conn = sqlite3.connect(str(DB_PATH))
    try:
        df_enhanced.to_sql(dest_table, conn, if_exists="replace", index=False)
    finally:
        conn.close()

    print("Done.")


if __name__ == "__main__":
    main()
