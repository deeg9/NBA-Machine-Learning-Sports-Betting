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

# ── 4. Match using (date_str, home, away) keys ───────────────────────────
print(f"\n{'=' * 70}")
print("  KEY MATCHING ANALYSIS")
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
print(f"Matched:             {matched}")
print(f"Unmatched:           {len(unmatched_games)}")
print(f"Match rate:          {matched / total_test * 100:.1f}%")

# ── 5. For unmatched games, find close matches ───────────────────────────
if unmatched_games:
    print(f"\n--- First 10 unmatched games (of {len(unmatched_games)}): ---")
    # Build lookup structures for fuzzy matching
    odds_keys_by_date = defaultdict(list)
    for k in odds_map:
        odds_keys_by_date[k[0]].append(k)

    for game_key in unmatched_games[:10]:
        date_str, home, away = game_key
        print(f"\n  Game: ({date_str}, '{home}', '{away}')")

        # Check if date exists in odds at all
        same_date_keys = odds_keys_by_date.get(date_str, [])
        if not same_date_keys:
            print(f"    -> NO odds for this date at all!")
        else:
            print(f"    -> {len(same_date_keys)} odds rows on this date:")
            for ok in same_date_keys[:5]:
                print(f"       ({ok[0]}, '{ok[1]}', '{ok[2]}')")

            # Fuzzy match home team
            odds_homes_on_date = [k[1] for k in same_date_keys]
            close_home = get_close_matches(home, odds_homes_on_date, n=2, cutoff=0.5)
            if close_home:
                print(f"    -> Close home matches: {close_home}")

            odds_aways_on_date = [k[2] for k in same_date_keys]
            close_away = get_close_matches(away, odds_aways_on_date, n=2, cutoff=0.5)
            if close_away:
                print(f"    -> Close away matches: {close_away}")

# ── 6. Check date format alignment ───────────────────────────────────────
print(f"\n{'=' * 70}")
print("  DATE FORMAT CHECK")
print(f"{'=' * 70}")

dataset_dates = sorted(df_test["Date"].dropna().dt.strftime("%Y-%m-%d").unique())
odds_dates = sorted(odds_df["Date"].dropna().apply(
    lambda x: pd.to_datetime(x, errors="coerce")).dropna().dt.strftime("%Y-%m-%d").unique())

print(f"\nDataset date range: {dataset_dates[0]} to {dataset_dates[-1]} ({len(dataset_dates)} unique dates)")
print(f"Odds date range:    {odds_dates[0]} to {odds_dates[-1]} ({len(odds_dates)} unique dates)")

# Overlap
date_overlap = set(dataset_dates) & set(odds_dates)
print(f"Overlapping dates:  {len(date_overlap)}")
dataset_only = set(dataset_dates) - set(odds_dates)
odds_only = set(odds_dates) - set(dataset_dates)
if dataset_only:
    print(f"Dates in dataset but NOT odds ({len(dataset_only)}): {sorted(dataset_only)[:5]}...")
if odds_only:
    print(f"Dates in odds but NOT dataset ({len(odds_only)}): {sorted(odds_only)[:5]}...")

# ── 7. If matches found, compute edge distribution ───────────────────────
if matched > 0:
    print(f"\n{'=' * 70}")
    print("  EDGE DISTRIBUTION (for matched games)")
    print(f"{'=' * 70}")

    # We don't have model probs here (no model trained), so just compare
    # implied probs from odds to a naive 50/50 to show what the market says,
    # and show the implied prob distribution
    def american_to_implied(odds):
        if odds is None:
            return None
        try:
            odds = float(odds)
        except (TypeError, ValueError):
            return None
        if odds < 0:
            return abs(odds) / (abs(odds) + 100)
        else:
            return 100 / (odds + 100)

    implied_probs = []
    for key in matched_games:
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

    # Histogram of implied probs
    print(f"\n  Implied probability distribution (histogram):")
    bins = np.linspace(0, 1, 11)
    counts, _ = np.histogram(implied_probs, bins=bins)
    for lo, hi, cnt in zip(bins[:-1], bins[1:], counts):
        bar = "#" * (cnt * 2)
        print(f"    [{lo:.1f}, {hi:.1f})  {cnt:>4d}  {bar}")

    # How many games have implied prob far from 0.5 (i.e., big favorites)?
    print(f"\n  Games where implied home-win prob is far from 50%:")
    for thresh in [0.05, 0.10, 0.15, 0.20]:
        cnt = (np.abs(implied_probs - 0.5) >= thresh).sum()
        print(f"    |implied - 0.5| >= {thresh:.0%}: {cnt} games ({cnt/len(implied_probs)*100:.1f}%)")

    # What edge would a model need?
    print(f"\n  NOTE: The backtest uses min_edge=0.05 (5%).")
    print(f"  For a bet to trigger, |model_prob - implied_prob| >= 0.05")
    print(f"  If the model outputs ~0.50 for all games (compressed probs),")
    print(f"  only games with implied prob outside [0.45, 0.55] could trigger.")
    games_outside = (np.abs(implied_probs - 0.5) >= 0.05).sum()
    print(f"  Games with implied prob outside [0.45, 0.55]: {games_outside} ({games_outside/len(implied_probs)*100:.1f}%)")

print(f"\n{'=' * 70}")
print("  VERDICT")
print(f"{'=' * 70}")
if matched == 0:
    print("\n  PROBLEM: Zero matches! The odds lookup is completely broken.")
    print("  Likely cause: team name mismatch between dataset and odds DB.")
elif matched < total_test * 0.5:
    print(f"\n  PROBLEM: Only {matched}/{total_test} games matched ({matched/total_test*100:.1f}%).")
    print("  Partial team name mismatch or date coverage gap.")
else:
    print(f"\n  Odds lookup is WORKING: {matched}/{total_test} games matched ({matched/total_test*100:.1f}%).")
    print("  The 'only 2 bets' problem is NOT caused by team name mismatch.")
    print("  Root cause is likely: model probabilities are too compressed toward 0.5,")
    print("  so |model_prob - implied_prob| rarely exceeds the 5% edge threshold.")

print("=" * 70)
