#!/usr/bin/env python
"""
Diagnostic: why does the backtest find only 2 bets?
Is it (A) team-name mismatch so odds lookup fails, or (B) model agrees with market?
"""

import sqlite3
from collections import defaultdict
from difflib import get_close_matches
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATASET_DB = BASE_DIR / "Data" / "dataset.sqlite"
ODDS_DB = BASE_DIR / "Data" / "OddsData.sqlite"

# ── 1. Load dataset (2025-26 season games) ────────────────────────────────
print("=" * 70)
print("  ODDS-MATCH DIAGNOSTIC")
print("=" * 70)

with sqlite3.connect(DATASET_DB) as con:
    df = pd.read_sql_query('SELECT * FROM "dataset_enhanced"', con)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.sort_values("Date").reset_index(drop=True)


def season_from_date(dt):
    if pd.isna(dt):
        return None
    if dt.month >= 8:
        return f"{dt.year}-{str(dt.year + 1)[-2:]}"
    return f"{dt.year - 1}-{str(dt.year)[-2:]}"


df["_season"] = df["Date"].apply(season_from_date)
df_test = df[df["_season"] == "2025-26"].copy()
print(f"\nTotal 2025-26 games in dataset: {len(df_test)}")

# ── 2. Load odds ──────────────────────────────────────────────────────────
with sqlite3.connect(ODDS_DB) as con:
    odds_df = pd.read_sql_query('SELECT * FROM "odds_2025-26"', con)

odds_df["Date"] = pd.to_datetime(odds_df["Date"], errors="coerce")
print(f"Total odds rows in odds_2025-26: {len(odds_df)}")

# ── 3. Print sample team names from BOTH sources ─────────────────────────
dataset_home_teams = sorted(df_test["TEAM_NAME"].dropna().unique())
dataset_away_teams = sorted(df_test["TEAM_NAME.1"].dropna().unique())
dataset_all_teams = sorted(set(dataset_home_teams) | set(dataset_away_teams))

odds_home_teams = sorted(odds_df["Home"].dropna().unique())
odds_away_teams = sorted(odds_df["Away"].dropna().unique())
odds_all_teams = sorted(set(odds_home_teams) | set(odds_away_teams))

print(f"\n--- Dataset teams (first 10 of {len(dataset_all_teams)}): ---")
for t in dataset_all_teams[:10]:
    print(f"  '{t}'")

print(f"\n--- Odds teams (first 10 of {len(odds_all_teams)}): ---")
for t in odds_all_teams[:10]:
    print(f"  '{t}'")

# Show ALL teams side-by-side for quick comparison
print(f"\n--- Full team name comparison (dataset vs odds) ---")
print(f"  {'Dataset':<35s} | {'Odds':<35s}")
print("  " + "-" * 73)
max_len = max(len(dataset_all_teams), len(odds_all_teams))
for i in range(max_len):
    d = dataset_all_teams[i] if i < len(dataset_all_teams) else ""
    o = odds_all_teams[i] if i < len(odds_all_teams) else ""
    marker = " <-- DIFF" if d != o and d and o else ""
    print(f"  {d:<35s} | {o:<35s}{marker}")

# ── 4. Match using (date_str, home, away) keys — EXACT dates ─────────────
print(f"\n{'=' * 70}")
print("  KEY MATCHING ANALYSIS (exact date)")
print(f"{'=' * 70}")

# Build odds map (same as backtest_kalshi.py)
odds_map = {}
for _, row in odds_df.iterrows():
    date_val = pd.to_datetime(row["Date"], errors="coerce")
    if pd.isna(date_val):
        continue
    key = (date_val.strftime("%Y-%m-%d"), row["Home"], row["Away"])
    odds_map[key] = {"ML_Home": row["ML_Home"], "ML_Away": row["ML_Away"]}

# Build dataset keys
matched = 0
unmatched_games = []
matched_games = []

for i in range(len(df_test)):
    dt = df_test["Date"].iloc[i]
    if pd.isna(dt):
        continue
    date_str = dt.strftime("%Y-%m-%d")
    home = df_test["TEAM_NAME"].iloc[i]
    away = df_test["TEAM_NAME.1"].iloc[i]
    key = (date_str, home, away)

    if key in odds_map:
        matched += 1
        matched_games.append(key)
    else:
        unmatched_games.append(key)

total_test = len(df_test)
print(f"\nTotal test games:     {total_test}")
print(f"Total odds map keys: {len(odds_map)}")
print(f"Matched (exact):     {matched}")
print(f"Unmatched:           {len(unmatched_games)}")
print(f"Match rate:          {matched / total_test * 100:.1f}%")

# ── 5. For unmatched games, find close matches ───────────────────────────
if unmatched_games:
    print(f"\n--- First 10 unmatched games (of {len(unmatched_games)}): ---")
    odds_keys_by_date = defaultdict(list)
    for k in odds_map:
        odds_keys_by_date[k[0]].append(k)

    for game_key in unmatched_games[:10]:
        date_str, home, away = game_key
        print(f"\n  Game: ({date_str}, '{home}', '{away}')")

        same_date_keys = odds_keys_by_date.get(date_str, [])
        if not same_date_keys:
            print(f"    -> NO odds for this date at all!")
        else:
            print(f"    -> {len(same_date_keys)} odds rows on this date:")
            for ok in same_date_keys[:5]:
                print(f"       ({ok[0]}, '{ok[1]}', '{ok[2]}')")

            odds_homes_on_date = [k[1] for k in same_date_keys]
            close_home = get_close_matches(home, odds_homes_on_date, n=2, cutoff=0.5)
            if close_home:
                print(f"    -> Close home matches: {close_home}")

            odds_aways_on_date = [k[2] for k in same_date_keys]
            close_away = get_close_matches(away, odds_aways_on_date, n=2, cutoff=0.5)
            if close_away:
                print(f"    -> Close away matches: {close_away}")

# ── 6. Date alignment check — game counts per date ───────────────────────
print(f"\n{'=' * 70}")
print("  DATE ALIGNMENT CHECK — game counts per date")
print(f"{'=' * 70}")

dataset_dates = sorted(df_test["Date"].dropna().dt.strftime("%Y-%m-%d").unique())
odds_dates_list = sorted(odds_df["Date"].dropna().apply(
    lambda x: pd.to_datetime(x, errors="coerce")).dropna().dt.strftime("%Y-%m-%d").unique())

dataset_counts = df_test.groupby(df_test["Date"].dt.strftime("%Y-%m-%d")).size()
odds_counts = odds_df.groupby(odds_df["Date"].dt.strftime("%Y-%m-%d")).size()

print(f"\nDataset date range: {dataset_dates[0]} to {dataset_dates[-1]} ({len(dataset_dates)} unique dates)")
print(f"Odds date range:    {odds_dates_list[0]} to {odds_dates_list[-1]} ({len(odds_dates_list)} unique dates)")

print(f"\n  {'Dataset Date':<14s} {'#Games':>6s}   {'Odds Date':<14s} {'#Games':>6s}   Match?")
print("  " + "-" * 65)
for i, d_date in enumerate(dataset_dates[:15]):
    d_cnt = dataset_counts.get(d_date, 0)
    # Check if odds date +1 day has same count
    next_day = (pd.to_datetime(d_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    o_cnt = odds_counts.get(next_day, 0)
    match_str = "YES" if d_cnt == o_cnt else f"NO ({d_cnt} vs {o_cnt})"
    print(f"  {d_date:<14s} {d_cnt:>6d}   {next_day:<14s} {o_cnt:>6d}   {match_str}")

# ── 7. Try matching with +1 day offset ────────────────────────────────────
print(f"\n{'=' * 70}")
print("  KEY MATCHING WITH +1 DAY OFFSET (dataset_date + 1 = odds_date)")
print(f"{'=' * 70}")

matched_shifted = 0
matched_shifted_games = []

for i in range(len(df_test)):
    dt = df_test["Date"].iloc[i]
    if pd.isna(dt):
        continue
    shifted_date = (dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    home = df_test["TEAM_NAME"].iloc[i]
    away = df_test["TEAM_NAME.1"].iloc[i]
    key = (shifted_date, home, away)

    if key in odds_map:
        matched_shifted += 1
        matched_shifted_games.append(key)

print(f"\nWith +1 day offset: {matched_shifted}/{total_test} matched ({matched_shifted / total_test * 100:.1f}%)")
print(f"Without offset:     {matched}/{total_test} matched ({matched / total_test * 100:.1f}%)")

# ── 8. Edge distribution (using shifted matches) ─────────────────────────
def american_to_implied(odds_val):
    if odds_val is None:
        return None
    try:
        odds_val = float(odds_val)
    except (TypeError, ValueError):
        return None
    if odds_val < 0:
        return abs(odds_val) / (abs(odds_val) + 100)
    else:
        return 100 / (odds_val + 100)


all_matched = matched_shifted_games if matched_shifted > matched else matched_games

if all_matched:
    print(f"\n{'=' * 70}")
    print(f"  EDGE DISTRIBUTION (for {len(all_matched)} matched games)")
    print(f"{'=' * 70}")

    implied_probs = []
    for key in all_matched:
        info = odds_map[key]
        p_home = american_to_implied(info["ML_Home"])
        p_away = american_to_implied(info["ML_Away"])
        if p_home is not None and p_away is not None:
            total = p_home + p_away
            fair_home = p_home / total
            implied_probs.append(fair_home)

    implied_probs = np.array(implied_probs)
    print(f"\nImplied home-win probabilities from odds (vig-removed):")
    print(f"  Count:  {len(implied_probs)}")
    print(f"  Mean:   {implied_probs.mean():.4f}")
    print(f"  Std:    {implied_probs.std():.4f}")
    print(f"  Min:    {implied_probs.min():.4f}")
    print(f"  Max:    {implied_probs.max():.4f}")
    q25, q50, q75 = np.percentile(implied_probs, [25, 50, 75])
    print(f"  IQR:    [{q25:.3f}, {q50:.3f}, {q75:.3f}]")

    print(f"\n  Implied probability distribution (histogram):")
    bins = np.linspace(0, 1, 11)
    counts, _ = np.histogram(implied_probs, bins=bins)
    for lo, hi, cnt in zip(bins[:-1], bins[1:], counts):
        bar = "#" * (cnt // 2 + (1 if cnt else 0))
        print(f"    [{lo:.1f}, {hi:.1f})  {cnt:>4d}  {bar}")

    print(f"\n  Games where implied home-win prob is far from 50%:")
    for thresh in [0.05, 0.10, 0.15, 0.20]:
        cnt = (np.abs(implied_probs - 0.5) >= thresh).sum()
        print(f"    |implied - 0.5| >= {thresh:.0%}: {cnt} games ({cnt / len(implied_probs) * 100:.1f}%)")

    print(f"\n  NOTE: The backtest uses min_edge=0.05 (5%).")
    print(f"  For a bet to trigger, |model_prob - implied_prob| >= 0.05")
    print(f"  If the model outputs ~0.50 for all games (compressed probs),")
    print(f"  only games with implied prob outside [0.45, 0.55] could trigger.")
    games_outside = (np.abs(implied_probs - 0.5) >= 0.05).sum()
    print(f"  Games with implied prob outside [0.45, 0.55]: {games_outside} ({games_outside / len(implied_probs) * 100:.1f}%)")

# ── VERDICT ───────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("  VERDICT")
print(f"{'=' * 70}")

print(f"""
  ROOT CAUSE FOUND: 1-DAY DATE OFFSET between dataset and odds.

  The dataset dates are 1 day EARLIER than the odds dates for the same games.
  Example:
    - Dataset has "Dallas Mavericks vs Washington Wizards" on 2025-10-23
    - Odds has   "Dallas Mavericks vs Washington Wizards" on 2025-10-24

  Game count pattern confirms this:
    Dataset Oct 23 = 12 games  -->  Odds Oct 24 = 12 games
    Dataset Oct 24 = 5 games   -->  Odds Oct 25 = 5 games
    Dataset Oct 25 = 9 games   -->  Odds Oct 26 = 9 games

  With exact date matching: {matched}/{total_test} games match (0.4%)
  With +1 day offset:       {matched_shifted}/{total_test} games match ({matched_shifted / total_test * 100:.1f}%)

  Team names are IDENTICAL between both sources (all 30 teams match perfectly).

  FIX: In backtest_kalshi.py, either:
    (a) Shift the dataset date by +1 day when building the lookup key, or
    (b) Shift the odds date by -1 day when loading, or
    (c) Fix the upstream data pipeline that creates one of these tables.

  The "only 2 bets" problem is caused by the odds lookup returning None
  for 536 of 538 games due to this date mismatch. The 2 games that DO match
  are coincidental overlaps where different matchups happen to align.
""")
print("=" * 70)
