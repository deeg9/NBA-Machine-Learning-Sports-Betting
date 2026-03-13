#!/usr/bin/env python
"""
Parameter sensitivity sweep over the Kalshi backtest.

Trains the ensemble ONCE, then sweeps min_edge and kelly_fraction over the
trading logic only. Much faster than retraining 25 times.

Usage:
    python backtest_kalshi_sweep.py
"""

import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_kalshi import (
    load_dataset, extract_feature_cols, season_from_date,
    train_ensemble, predict_ensemble,
    load_odds_2025, odds_to_kalshi_price, compute_trade,
    TARGET_COLUMN, DATE_COLUMN,
)

MIN_EDGES = [0.03, 0.05, 0.08, 0.10, 0.15]
KELLY_FRACTIONS = [0.10, 0.15, 0.25, 0.35, 0.50]


def simulate_trading(model_probs, y_test, df_test, odds_map,
                     bankroll_dollars=10.0, max_position_dollars=2.0,
                     daily_loss_dollars=5.0, min_edge=0.05, kelly_fraction=0.25):
    """Run only the trading simulation (no model training)."""
    bankroll_cents = int(bankroll_dollars * 100)
    max_pos = int(max_position_dollars * 100)
    daily_limit = int(daily_loss_dollars * 100)
    starting_bankroll = bankroll_cents

    dates = df_test[DATE_COLUMN]
    home_names = df_test["TEAM_NAME"]
    away_names = df_test["TEAM_NAME.1"]

    date_groups = defaultdict(list)
    for i in range(len(df_test)):
        dt = dates.iloc[i]
        if pd.isna(dt):
            continue
        date_groups[dt.strftime("%Y-%m-%d")].append(i)

    total_bets = 0
    total_wins = 0
    trajectory = [bankroll_cents]

    for date_str in sorted(date_groups.keys()):
        day_exposure = 0
        for i in date_groups[date_str]:
            home = home_names.iloc[i]
            away = away_names.iloc[i]
            actual_home_win = bool(y_test[i] == 1)
            prob_home = model_probs[i]

            odds_info = odds_map.get((date_str, home, away))
            if odds_info is None:
                continue

            contract_price = odds_to_kalshi_price(odds_info["ML_Home"], odds_info["ML_Away"])
            if contract_price is None or contract_price <= 0 or contract_price >= 100:
                continue

            trade = compute_trade(prob_home, contract_price, bankroll_cents,
                                  max_pos, min_edge, kelly_fraction)
            if trade is None:
                continue

            if day_exposure + trade["cost_cents"] > daily_limit:
                continue

            won = actual_home_win if trade["side"] == "yes" else not actual_home_win
            if won:
                profit = trade["count"] * (100 - trade["price"])
                total_wins += 1
            else:
                profit = -trade["cost_cents"]

            bankroll_cents += profit
            day_exposure += trade["cost_cents"]
            total_bets += 1
            trajectory.append(bankroll_cents)

            if bankroll_cents <= 0:
                break
        if bankroll_cents <= 0:
            break

    final_bankroll = bankroll_cents / 100
    roi = (bankroll_cents - starting_bankroll) / starting_bankroll * 100

    peak = trajectory[0]
    max_dd = 0
    for val in trajectory:
        if val > peak:
            peak = val
        dd = (peak - val) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    return {
        "final_bankroll": final_bankroll,
        "roi": roi,
        "max_dd": max_dd,
        "total_bets": total_bets,
        "win_rate": total_wins / total_bets if total_bets > 0 else 0,
    }


def main():
    print("Loading dataset...")
    df = load_dataset("dataset_enhanced")
    feature_cols = extract_feature_cols(df)
    df["_season"] = df[DATE_COLUMN].apply(season_from_date)

    train_mask = df[DATE_COLUMN] < pd.Timestamp("2025-10-01")
    test_mask = df["_season"] == "2025-26"

    df_train = df.loc[train_mask]
    df_test = df.loc[test_mask].copy()

    X_train = np.nan_to_num(df_train[feature_cols].astype(float).values, nan=0.0)
    y_train = df_train[TARGET_COLUMN].astype(int).values
    X_test = np.nan_to_num(df_test[feature_cols].astype(float).values, nan=0.0)
    y_test = df_test[TARGET_COLUMN].astype(int).values

    print(f"Training on {len(X_train)} games, testing on {len(X_test)} games")

    print("Training stacked ensemble (once)...")
    fitted, meta_learner, sigmoid_cal, scaler, model_names = train_ensemble(X_train, y_train)

    print("Running predictions...")
    model_probs = predict_ensemble(fitted, meta_learner, sigmoid_cal, scaler, X_test)

    from sklearn.metrics import accuracy_score
    preds = (model_probs >= 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    print(f"Model accuracy: {acc:.1%}")

    print("Loading odds...")
    odds_map = load_odds_2025()

    # Reset df_test index so iloc works correctly with y_test
    df_test = df_test.reset_index(drop=True)

    print(f"\nSweeping {len(MIN_EDGES)} x {len(KELLY_FRACTIONS)} = "
          f"{len(MIN_EDGES) * len(KELLY_FRACTIONS)} combos...\n")

    results = {}
    for me in MIN_EDGES:
        for kf in KELLY_FRACTIONS:
            res = simulate_trading(model_probs, y_test, df_test, odds_map,
                                   min_edge=me, kelly_fraction=kf)
            results[(me, kf)] = res

    # ── Print results table ───────────────────────────────────────────────
    print("=" * 90)
    print("  PARAMETER SENSITIVITY SWEEP RESULTS")
    print("=" * 90)

    header = f"{'min_edge':<10}"
    for kf in KELLY_FRACTIONS:
        header += f" | kelly={kf:<4}         "
    print(header)
    print("-" * 90)

    best_roi = -float("inf")
    worst_roi = float("inf")
    best_combo = None
    worst_combo = None

    for me in MIN_EDGES:
        row = f"{me:<10.2f}"
        for kf in KELLY_FRACTIONS:
            r = results[(me, kf)]
            cell = f"${r['final_bankroll']:>7.2f} ({r['roi']:>+7.1f}%)"
            row += f" | {cell:<18}"
            if r["roi"] > best_roi:
                best_roi = r["roi"]
                best_combo = (me, kf)
            if r["roi"] < worst_roi:
                worst_roi = r["roi"]
                worst_combo = (me, kf)
        print(row)

    print("-" * 90)

    if best_combo:
        br = results[best_combo]
        print(f"\n  BEST:  min_edge={best_combo[0]}, kelly={best_combo[1]} "
              f"-> ${br['final_bankroll']:.2f} (ROI {br['roi']:+.1f}%, "
              f"DD {br['max_dd']:.1%}, {br['total_bets']} bets, "
              f"WR {br['win_rate']:.1%})")

    if worst_combo:
        wr = results[worst_combo]
        print(f"  WORST: min_edge={worst_combo[0]}, kelly={worst_combo[1]} "
              f"-> ${wr['final_bankroll']:.2f} (ROI {wr['roi']:+.1f}%, "
              f"DD {wr['max_dd']:.1%}, {wr['total_bets']} bets, "
              f"WR {wr['win_rate']:.1%})")

    print("=" * 90)


if __name__ == "__main__":
    main()
