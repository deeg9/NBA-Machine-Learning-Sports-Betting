#!/usr/bin/env python
"""
Kalshi-style backtest: simulate the live trading bot over the 2025-26 season
using historical game data and sportsbook odds as proxy for Kalshi prices.

Trains the stacked ensemble on all pre-2025-26 data, then walks through each
game day applying the same Kelly criterion logic from live_trader.py with a
$10 bankroll.

Usage:
    python backtest_kalshi.py
    python backtest_kalshi.py --bankroll 10.00 --kelly-fraction 0.25
"""

import argparse
import sqlite3
from collections import defaultdict
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import _SigmoidCalibration
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Import ensemble helpers
_ensemble_mod = import_module("src.Train-Models.Stacked_Ensemble_ML")
build_base_learners = _ensemble_mod.build_base_learners
generate_oof_predictions = _ensemble_mod.generate_oof_predictions
train_final_base_learners = _ensemble_mod.train_final_base_learners
predict_base_learners = _ensemble_mod.predict_base_learners
SCALED_MODELS = _ensemble_mod.SCALED_MODELS

BASE_DIR = Path(__file__).resolve().parent
DATASET_DB = BASE_DIR / "Data" / "dataset.sqlite"
ODDS_DB = BASE_DIR / "Data" / "OddsData.sqlite"

TARGET_COLUMN = "Home-Team-Win"
DATE_COLUMN = "Date"
DROP_COLUMNS = [
    "index", "Score", "Home-Team-Win", "TEAM_NAME", "Date",
    "index.1", "TEAM_NAME.1", "Date.1", "OU-Cover", "OU",
]


# ── Data loading ──────────────────────────────────────────────────────────

def load_dataset(table="dataset_enhanced"):
    with sqlite3.connect(DATASET_DB) as con:
        df = pd.read_sql_query(f'SELECT * FROM "{table}"', con)
    if DATE_COLUMN in df.columns:
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
        df = df.sort_values(DATE_COLUMN).reset_index(drop=True)
    return df


def load_odds_2025():
    """Load 2025-26 odds keyed by (date_str, home, away)."""
    odds_map = {}
    with sqlite3.connect(ODDS_DB) as con:
        df = pd.read_sql_query('SELECT * FROM "odds_2025-26"', con)
    for _, row in df.iterrows():
        date_val = pd.to_datetime(row["Date"], errors="coerce")
        if pd.isna(date_val):
            continue
        key = (date_val.strftime("%Y-%m-%d"), row["Home"], row["Away"])
        odds_map[key] = {
            "ML_Home": row["ML_Home"],
            "ML_Away": row["ML_Away"],
        }
    return odds_map


def extract_feature_cols(df):
    return [
        c for c in df.columns
        if c not in DROP_COLUMNS
        and df[c].dtype in ("float64", "float32", "int64", "int32")
    ]


def season_from_date(dt):
    if pd.isna(dt):
        return None
    if dt.month >= 8:
        return f"{dt.year}-{str(dt.year + 1)[-2:]}"
    return f"{dt.year - 1}-{str(dt.year)[-2:]}"


# ── Ensemble training ─────────────────────────────────────────────────────

def train_ensemble(X_train, y_train, seed=42):
    base_learners = build_base_learners(seed)
    scaler = StandardScaler()
    scaler.fit(X_train)

    oof_preds, model_names = generate_oof_predictions(
        base_learners, X_train, y_train, scaler, n_splits=5
    )

    valid_mask = ~np.isnan(oof_preds).any(axis=1)
    oof_valid = oof_preds[valid_mask]
    y_valid = y_train[valid_mask]

    meta_learner = LogisticRegression(
        C=0.5, max_iter=1000, random_state=seed, solver="lbfgs"
    )
    meta_learner.fit(oof_valid, y_valid)

    fitted = train_final_base_learners(base_learners, X_train, y_train, scaler)

    oof_meta = meta_learner.predict_proba(oof_valid)[:, 1]
    sigmoid_cal = _SigmoidCalibration()
    sigmoid_cal.fit(oof_meta, y_valid)

    return fitted, meta_learner, sigmoid_cal, scaler, model_names


def predict_ensemble(fitted, meta_learner, sigmoid_cal, scaler, X):
    preds_matrix = predict_base_learners(fitted, X, scaler)
    meta_probs = meta_learner.predict_proba(preds_matrix)[:, 1]
    if sigmoid_cal is not None:
        meta_probs = sigmoid_cal.predict(meta_probs)
    return meta_probs  # P(home_win)


# ── Odds → Kalshi contract price ─────────────────────────────────────────

def american_to_implied(odds):
    """Convert American moneyline odds to implied probability."""
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


def odds_to_kalshi_price(ml_home, ml_away):
    """Convert moneyline odds to Kalshi-style contract price in cents (1-99).

    Removes vig by normalizing the two-way implied probabilities.
    """
    p_home = american_to_implied(ml_home)
    p_away = american_to_implied(ml_away)
    if p_home is None or p_away is None:
        return None
    total = p_home + p_away  # typically > 1.0 due to vig
    fair_home = p_home / total
    return int(round(fair_home * 100))


# ── Kelly + edge (mirrors live_trader.py) ─────────────────────────────────

def compute_trade(model_prob, contract_price, bankroll_cents,
                  max_position_cents, min_edge, kelly_fraction):
    """Decide whether and how much to trade.

    Returns dict with side, edge, size_cents, or None if no trade.
    """
    implied = contract_price / 100.0
    edge_yes = model_prob - implied

    if edge_yes >= min_edge:
        side = "yes"
        edge = edge_yes
        price = contract_price
        p = model_prob
    elif -edge_yes >= min_edge:
        side = "no"
        edge = -edge_yes
        price = 100 - contract_price
        p = 1 - model_prob
    else:
        return None

    # Decimal odds for Kelly
    decimal_odds = 100 / price if price > 0 else 1
    b = decimal_odds - 1
    q = 1 - p

    if b <= 0:
        return None

    kelly_pct = ((b * p - q) / b) * kelly_fraction
    kelly_pct = max(0.0, kelly_pct)

    size_cents = int(kelly_pct * bankroll_cents)
    size_cents = max(0, min(size_cents, max_position_cents))

    if size_cents <= 0:
        return None

    count = max(1, size_cents // price)
    cost_cents = count * price

    return {
        "side": side,
        "edge": round(edge, 4),
        "kelly_pct": round(kelly_pct * 100, 2),
        "count": count,
        "price": price,
        "cost_cents": cost_cents,
    }


# ── Backtest ──────────────────────────────────────────────────────────────

def run_backtest(bankroll_dollars=10.0, max_position_dollars=2.0,
                 daily_loss_dollars=5.0, min_edge=0.05, kelly_fraction=0.25,
                 dataset="dataset_enhanced", quiet=False):
    bankroll_cents = int(bankroll_dollars * 100)
    max_pos = int(max_position_dollars * 100)
    daily_limit = int(daily_loss_dollars * 100)
    starting_bankroll = bankroll_cents

    def _print(*args, **kwargs):
        if not quiet:
            print(*args, **kwargs)

    _print("=" * 65)
    _print("  Kalshi Backtest — 2025-26 NBA Season (Ensemble Model)")
    _print("=" * 65)
    _print(f"  Starting bankroll:  ${bankroll_dollars:.2f}")
    _print(f"  Max per game:       ${max_position_dollars:.2f}")
    _print(f"  Daily loss limit:   ${daily_loss_dollars:.2f}")
    _print(f"  Kelly fraction:     {kelly_fraction}")
    _print(f"  Min edge:           {min_edge:.0%}")
    _print(f"  Dataset:            {dataset}")
    _print("=" * 65)

    # Load data
    _print("\nLoading dataset...")
    df = load_dataset(dataset)
    feature_cols = extract_feature_cols(df)
    df["_season"] = df[DATE_COLUMN].apply(season_from_date)

    # Split: train on everything before 2025-26, test on 2025-26
    train_mask = df[DATE_COLUMN] < pd.Timestamp("2025-10-01")
    test_mask = df["_season"] == "2025-26"

    df_train = df.loc[train_mask]
    df_test = df.loc[test_mask].copy()

    X_train = np.nan_to_num(df_train[feature_cols].astype(float).values, nan=0.0)
    y_train = df_train[TARGET_COLUMN].astype(int).values
    X_test = np.nan_to_num(df_test[feature_cols].astype(float).values, nan=0.0)
    y_test = df_test[TARGET_COLUMN].astype(int).values

    _print(f"Training on {len(X_train)} games (pre-2025-26)")
    _print(f"Testing on {len(X_test)} games (2025-26 season)")

    # Train ensemble
    _print("\nTraining stacked ensemble (6 base learners + meta)...")
    fitted, meta_learner, sigmoid_cal, scaler, model_names = train_ensemble(
        X_train, y_train
    )
    _print(f"Base learners: {model_names}")

    # Predict
    _print("Running predictions...")
    model_probs = predict_ensemble(fitted, meta_learner, sigmoid_cal, scaler, X_test)
    preds = (model_probs >= 0.5).astype(int)

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_test, preds)
    _print(f"Model accuracy on 2025-26: {acc:.1%}")

    # Load odds
    _print("Loading 2025-26 odds...")
    odds_map = load_odds_2025()
    _print(f"Loaded {len(odds_map)} odds rows")

    # ── Day-by-day simulation ─────────────────────────────────────────────
    _print("\n" + "-" * 65)
    _print(f"{'Date':<12} {'Games':>5} {'Bets':>5} {'Day P&L':>10} "
           f"{'Bankroll':>10} {'Cum ROI':>8}")
    _print("-" * 65)

    dates = df_test[DATE_COLUMN]
    home_names = df_test["TEAM_NAME"]
    away_names = df_test["TEAM_NAME.1"]

    # Group by date
    date_groups = defaultdict(list)
    for i in range(len(df_test)):
        dt = dates.iloc[i]
        if pd.isna(dt):
            continue
        date_str = dt.strftime("%Y-%m-%d")
        date_groups[date_str].append(i)

    total_bets = 0
    total_wins = 0
    total_games_with_edge = 0
    trajectory = [bankroll_cents]
    monthly_pnl = defaultdict(float)
    all_edges = []
    busted = False

    for date_str in sorted(date_groups.keys()):
        indices = date_groups[date_str]
        day_bets = 0
        day_pnl = 0
        day_exposure = 0

        for i in indices:
            home = home_names.iloc[i]
            away = away_names.iloc[i]
            actual_home_win = bool(y_test[i] == 1)
            prob_home = model_probs[i]

            # Get odds → Kalshi price
            key = (date_str, home, away)
            odds_info = odds_map.get(key)
            if odds_info is None:
                continue

            contract_price = odds_to_kalshi_price(
                odds_info["ML_Home"], odds_info["ML_Away"]
            )
            if contract_price is None or contract_price <= 0 or contract_price >= 100:
                continue

            # Compute trade
            trade = compute_trade(
                prob_home, contract_price, bankroll_cents,
                max_pos, min_edge, kelly_fraction,
            )
            if trade is None:
                continue

            # Daily limit check
            if day_exposure + trade["cost_cents"] > daily_limit:
                continue

            total_games_with_edge += 1
            all_edges.append(trade["edge"])

            # Simulate outcome
            if trade["side"] == "yes":
                won = actual_home_win
            else:
                won = not actual_home_win

            if won:
                # Profit = count * (100 - price) cents per contract
                profit = trade["count"] * (100 - trade["price"])
                total_wins += 1
            else:
                # Loss = cost of contracts
                profit = -trade["cost_cents"]

            bankroll_cents += profit
            day_pnl += profit
            day_exposure += trade["cost_cents"]
            day_bets += 1
            total_bets += 1

            # Bust check
            if bankroll_cents <= 0:
                busted = True
                break

        if day_bets > 0:
            month_key = date_str[:7]
            monthly_pnl[month_key] += day_pnl
            trajectory.append(bankroll_cents)
            _print(
                f"{date_str:<12} {len(indices):>5} {day_bets:>5} "
                f"${day_pnl / 100:>+9.2f} "
                f"${bankroll_cents / 100:>9.2f} "
                f"{(bankroll_cents - starting_bankroll) / starting_bankroll * 100:>+7.1f}%"
            )

        if busted:
            _print(f"\n  *** BUSTED on {date_str} ***")
            break

    # ── Summary ───────────────────────────────────────────────────────────
    final_bankroll = bankroll_cents / 100
    pnl = (bankroll_cents - starting_bankroll) / 100
    roi = (bankroll_cents - starting_bankroll) / starting_bankroll * 100

    # Max drawdown
    peak = trajectory[0]
    max_dd = 0
    for val in trajectory:
        if val > peak:
            peak = val
        dd = (peak - val) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    _print("\n" + "=" * 65)
    _print("  BACKTEST RESULTS")
    _print("=" * 65)
    _print(f"  Starting bankroll:   ${bankroll_dollars:.2f}")
    _print(f"  Final bankroll:      ${final_bankroll:.2f}")
    _print(f"  Net P&L:             ${pnl:+.2f}")
    _print(f"  ROI:                 {roi:+.1f}%")
    _print(f"  Max drawdown:        {max_dd:.1%}")
    _print(f"  Model accuracy:      {acc:.1%}")
    _print(f"  Total bets placed:   {total_bets}")
    win_rate = total_wins / total_bets if total_bets > 0 else 0
    if total_bets > 0:
        _print(f"  Win rate:            {win_rate:.1%}")
        _print(f"  Avg edge on bets:    {np.mean(all_edges):.1%}")
    _print(f"  Games with edge:     {total_games_with_edge}")

    if monthly_pnl:
        _print(f"\n  Monthly breakdown:")
        for month in sorted(monthly_pnl):
            _print(f"    {month}:  ${monthly_pnl[month] / 100:+.2f}")

    # Sparkline
    if len(trajectory) > 1:
        steps = min(len(trajectory), 30)
        indices = [int(i * (len(trajectory) - 1) / (steps - 1)) for i in range(steps)]
        vals = [trajectory[i] / 100 for i in indices]
        lo, hi = min(vals), max(vals)
        spread = hi - lo if hi != lo else 1
        chars = " _.-~*^"
        line = ""
        for v in vals:
            idx = int((v - lo) / spread * (len(chars) - 1))
            line += chars[min(idx, len(chars) - 1)]
        _print(f"\n  Bankroll trajectory:  [{line}]")
        _print(f"                        ${lo:.2f} → ${hi:.2f}")

    _print("\n" + "=" * 65)

    return {
        "final_bankroll": final_bankroll,
        "roi": roi,
        "max_dd": max_dd,
        "total_bets": total_bets,
        "win_rate": win_rate,
        "accuracy": acc,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kalshi-style NBA backtest")
    parser.add_argument("--bankroll", type=float, default=10.0,
                        help="Starting bankroll in dollars (default: $10)")
    parser.add_argument("--max-position", type=float, default=2.0,
                        help="Max position per game in dollars (default: $2)")
    parser.add_argument("--daily-limit", type=float, default=5.0,
                        help="Daily loss limit in dollars (default: $5)")
    parser.add_argument("--min-edge", type=float, default=0.05,
                        help="Minimum edge to trade (default: 0.05)")
    parser.add_argument("--kelly-fraction", type=float, default=0.25,
                        help="Kelly fraction (default: 0.25)")
    parser.add_argument("--dataset", default="dataset_enhanced",
                        help="Dataset table (default: dataset_enhanced)")
    args = parser.parse_args()

    run_backtest(
        bankroll_dollars=args.bankroll,
        max_position_dollars=args.max_position,
        daily_loss_dollars=args.daily_limit,
        min_edge=args.min_edge,
        kelly_fraction=args.kelly_fraction,
        dataset=args.dataset,
    )
