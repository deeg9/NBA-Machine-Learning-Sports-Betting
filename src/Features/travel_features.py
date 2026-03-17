"""
Travel fatigue features for NBA game prediction.

Computes travel distance, timezone shifts, elevation differences, and
road trip length based on arena locations and game schedule order.
"""

import math

import numpy as np
import pandas as pd

from src.Features.arena_data import get_arena

# Earth radius in miles for haversine
_EARTH_RADIUS_MI = 3958.8

# Timezone UTC offsets (approximate, ignoring DST for simplicity)
_TZ_OFFSETS = {
    "America/New_York": -5,
    "America/Detroit": -5,
    "America/Indiana/Indianapolis": -5,
    "America/Toronto": -5,
    "America/Chicago": -6,
    "America/Denver": -7,
    "America/Phoenix": -7,
    "America/Los_Angeles": -8,
}


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in miles between two lat/lon points."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_MI * math.asin(math.sqrt(a))


def _tz_offset(tz_str: str) -> float:
    """Return approximate UTC offset for a timezone string."""
    return _TZ_OFFSETS.get(tz_str, -6)  # default to Central


def compute_travel_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add travel fatigue features to the dataset.

    Features added:
        Travel_Dist_Away  — miles from away team's previous game to this arena
        Travel_Dist_Home  — miles from home team's previous game (0 if was home)
        Timezone_Shift_Away — hours of timezone change for away team (signed)
        Elevation_Diff    — home arena elev minus away team's home arena elev (ft)
        Road_Trip_Length   — consecutive away games for the away team

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with TEAM_NAME, TEAM_NAME.1, Date columns.

    Returns
    -------
    pd.DataFrame
        Input dataframe with travel feature columns added.
    """
    df = df.copy()
    df["_date_parsed"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("_date_parsed").reset_index(drop=True)

    n = len(df)
    travel_dist_away = np.zeros(n, dtype=np.float64)
    travel_dist_home = np.zeros(n, dtype=np.float64)
    tz_shift_away = np.zeros(n, dtype=np.float64)
    elev_diff = np.zeros(n, dtype=np.float64)
    road_trip_len = np.zeros(n, dtype=np.float64)

    # Track each team's previous game location (lat, lon, tz) and
    # whether they were home or away
    team_prev_loc: dict[str, tuple[float, float, str]] = {}  # team -> (lat, lon, tz)
    team_prev_was_away: dict[str, bool] = {}
    team_consec_away: dict[str, int] = {}  # consecutive away game count

    for idx in range(n):
        row = df.iloc[idx]
        home = row["TEAM_NAME"]
        away = row["TEAM_NAME.1"]

        home_arena = get_arena(home)
        away_home_arena = get_arena(away)  # away team's HOME arena

        if home_arena is None or away_home_arena is None:
            continue

        game_lat = home_arena["lat"]
        game_lon = home_arena["lon"]
        game_tz = home_arena["tz"]

        # --- Elevation difference ---
        elev_diff[idx] = home_arena["elev_ft"] - away_home_arena["elev_ft"]

        # --- Away team travel distance ---
        if away in team_prev_loc:
            prev_lat, prev_lon, prev_tz = team_prev_loc[away]
            travel_dist_away[idx] = _haversine(prev_lat, prev_lon, game_lat, game_lon)
            tz_shift_away[idx] = _tz_offset(game_tz) - _tz_offset(prev_tz)
        # else: first game of season, defaults to 0

        # --- Home team travel distance (non-zero if returning from road) ---
        if home in team_prev_loc and team_prev_was_away.get(home, False):
            prev_lat, prev_lon, _ = team_prev_loc[home]
            travel_dist_home[idx] = _haversine(prev_lat, prev_lon, game_lat, game_lon)
        # else: home team was already home, 0

        # --- Road trip length for away team ---
        consec = team_consec_away.get(away, 0) + 1
        team_consec_away[away] = consec
        road_trip_len[idx] = consec

        # Reset home team's consecutive away counter
        team_consec_away[home] = 0

        # --- Update previous location tracking ---
        # Both teams played at the home team's arena
        team_prev_loc[home] = (game_lat, game_lon, game_tz)
        team_prev_loc[away] = (game_lat, game_lon, game_tz)
        team_prev_was_away[home] = False
        team_prev_was_away[away] = True

    df["Travel_Dist_Away"] = travel_dist_away
    df["Travel_Dist_Home"] = travel_dist_home
    df["Timezone_Shift_Away"] = tz_shift_away
    df["Elevation_Diff"] = elev_diff
    df["Road_Trip_Length"] = road_trip_len

    df.drop(columns=["_date_parsed"], inplace=True)
    return df
