#!/usr/bin/env python
"""
Circuit Breaker Backtest: DCA simulation with circuit breaker logic.

Trains ensemble once, runs month-by-month DCA simulation ($100/month Oct-Mar),
with a circuit breaker that pauses betting when the model loses calibration
and resumes with reduced Kelly during a probation window.

Usage:
    python3 backtest_circuit_breaker.py
    python3 backtest_circuit_breaker.py --season 2024-25
"""

import argparse
import itertools
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_kalshi import (
    BASE_DIR,
    DATE_COLUMN,
    ODDS_DB,
    TARGET_COLUMN,
    compute_trade,
    extract_feature_cols,
    load_dataset,
    load_odds_2025,
    odds_to_kalshi_price,
    predict_ensemble,
    season_from_date,
    train_ensemble,
)

DCA_AMOUNT_CENTS = 10000  # $100 per month


def dca_months_for_season(season_tag):
    """Return the 6 DCA month keys (Oct-Mar) for a given season tag like '2024-25'."""
    start_year = int(season_tag.split("-")[0])
    end_year = start_year + 1
    return [
        f"{start_year}-10", f"{start_year}-11", f"{start_year}-12",
        f"{end_year}-01", f"{end_year}-02", f"{end_year}-03",
    ]


def feb_mar_months_for_season(season_tag):
    """Return the Feb-Mar month keys for a season."""
    end_year = int(season_tag.split("-")[0]) + 1
    return [f"{end_year}-02", f"{end_year}-03"]


def load_odds_for_season(season_tag):
    """Load odds for any season, keyed by (date_str, home, away).

    Tries table names in order: 'odds_{season}', '{season}', then the
    _new variant. All odds tables have a 1-day-ahead date offset.
    """
    candidates = [f"odds_{season_tag}", season_tag, f"odds_{season_tag}_new"]
    with sqlite3.connect(ODDS_DB) as con:
        tables = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        for name in candidates:
            if name in tables:
                df = pd.read_sql_query(f'SELECT * FROM "{name}"', con)
                break
        else:
            raise ValueError(
                f"No odds table found for season {season_tag}. "
                f"Tried: {candidates}. Available: {sorted(tables)}"
            )

    odds_map = {}
    for _, row in df.iterrows():
        date_val = pd.to_datetime(row["Date"], errors="coerce")
        if pd.isna(date_val):
            continue
        # Odds DB dates are 1 day ahead of dataset dates
        date_val -= pd.Timedelta(days=1)
        key = (date_val.strftime("%Y-%m-%d"), row["Home"], row["Away"])
        odds_map[key] = {"ML_Home": row["ML_Home"], "ML_Away": row["ML_Away"]}
    return odds_map


def run_dca_sim(
    model_probs,
    y_test,
    df_test,
    odds_map,
    *,
    dca_months=None,
    kelly_fraction=0.10,
    min_edge=0.05,
    max_pos_pct=0.02,
    daily_limit_pct=0.05,
    # Circuit breaker params
    cb_enabled=True,
    cb_window=30,
    cb_winrate_trip=0.28,
    cb_pnl_trip=-0.15,
    cb_cooldown_days=3,
    cb_probation_kelly_mult=0.5,
    cb_probation_bets=10,
):
    """Run DCA simulation with optional circuit breaker.

    Returns dict with monthly stats, circuit breaker events, and final state.
    """
    if dca_months is None:
        dca_months = dca_months_for_season("2025-26")

    dates = df_test[DATE_COLUMN]
    home_names = df_test["TEAM_NAME"]
    away_names = df_test["TEAM_NAME.1"]

    # Group by date
    date_groups = defaultdict(list)
    for i in range(len(df_test)):
        dt = dates.iloc[i]
        if pd.isna(dt):
            continue
        date_groups[dt.strftime("%Y-%m-%d")].append(i)

    bankroll_cents = 0
    deposited_cents = 0
    months_deposited = set()

    # Bet history for circuit breaker
    bet_history = []  # list of {'won': bool, 'profit': int}

    # Circuit breaker state
    cb_tripped = False
    cb_trip_date = None
    cb_trip_reason = ""
    cb_cooldown_remaining = 0
    cb_in_probation = False
    cb_probation_count = 0
    cb_probation_pnl = 0
    cb_events = []

    # Monthly tracking
    monthly = defaultdict(lambda: {
        "games": 0, "bets": 0, "skipped_cb": 0,
        "wins": 0, "pnl": 0, "deposits": 0,
    })

    total_bets = 0
    total_wins = 0

    for date_str in sorted(date_groups.keys()):
        month_key = date_str[:7]

        # DCA deposit at start of each month
        if month_key in dca_months and month_key not in months_deposited:
            bankroll_cents += DCA_AMOUNT_CENTS
            deposited_cents += DCA_AMOUNT_CENTS
            months_deposited.add(month_key)
            monthly[month_key]["deposits"] = DCA_AMOUNT_CENTS

        if month_key not in dca_months:
            continue

        indices = date_groups[date_str]
        monthly[month_key]["games"] += len(indices)

        # Circuit breaker cooldown
        if cb_enabled and cb_tripped and not cb_in_probation:
            cb_cooldown_remaining -= 1
            if cb_cooldown_remaining <= 0:
                # Enter probation
                cb_in_probation = True
                cb_probation_count = 0
                cb_probation_pnl = 0
                cb_events.append({
                    "date": date_str,
                    "event": "probation_start",
                    "detail": f"Cooldown ended, entering probation ({cb_probation_bets} bets at {cb_probation_kelly_mult}x Kelly)",
                })

        # Determine effective kelly
        if cb_enabled and cb_tripped and not cb_in_probation:
            # Still in cooldown — skip all bets
            monthly[month_key]["skipped_cb"] += len(indices)
            continue

        effective_kelly = kelly_fraction
        if cb_enabled and cb_in_probation:
            effective_kelly = kelly_fraction * cb_probation_kelly_mult

        max_pos = int(max_pos_pct * bankroll_cents) if bankroll_cents > 0 else 0
        daily_limit = int(daily_limit_pct * bankroll_cents) if bankroll_cents > 0 else 0
        day_exposure = 0

        for i in indices:
            # Check if still in probation and exceeded bet count
            if cb_enabled and cb_in_probation and cb_probation_count >= cb_probation_bets:
                # Evaluate probation
                if cb_probation_pnl > 0:
                    # Probation passed — restore full Kelly
                    cb_tripped = False
                    cb_in_probation = False
                    cb_events.append({
                        "date": date_str,
                        "event": "probation_passed",
                        "detail": f"Probation P&L: ${cb_probation_pnl/100:+.2f}, restoring full Kelly",
                    })
                    effective_kelly = kelly_fraction
                else:
                    # Probation failed — trip again
                    cb_in_probation = False
                    cb_cooldown_remaining = cb_cooldown_days
                    cb_trip_date = date_str
                    cb_trip_reason = f"Probation failed (P&L: ${cb_probation_pnl/100:+.2f})"
                    cb_events.append({
                        "date": date_str,
                        "event": "probation_failed",
                        "detail": cb_trip_reason,
                    })
                    monthly[month_key]["skipped_cb"] += 1
                    continue

            home = home_names.iloc[i]
            away = away_names.iloc[i]
            actual_home_win = bool(y_test[i] == 1)
            prob_home = model_probs[i]

            key = (date_str, home, away)
            odds_info = odds_map.get(key)
            if odds_info is None:
                continue

            contract_price = odds_to_kalshi_price(
                odds_info["ML_Home"], odds_info["ML_Away"]
            )
            if contract_price is None or contract_price <= 0 or contract_price >= 100:
                continue

            # Check circuit breaker before computing trade
            if cb_enabled and not cb_tripped and len(bet_history) >= cb_window:
                recent = bet_history[-cb_window:]
                rolling_wr = sum(1 for b in recent if b["won"]) / len(recent)
                rolling_pnl = sum(b["profit"] for b in recent)
                rolling_pnl_pct = rolling_pnl / bankroll_cents if bankroll_cents > 0 else 0

                if rolling_wr < cb_winrate_trip:
                    cb_tripped = True
                    cb_in_probation = False
                    cb_cooldown_remaining = cb_cooldown_days
                    cb_trip_date = date_str
                    cb_trip_reason = f"Win rate {rolling_wr:.0%} < {cb_winrate_trip:.0%} (last {cb_window} bets)"
                    cb_events.append({
                        "date": date_str,
                        "event": "tripped",
                        "detail": cb_trip_reason,
                    })
                    monthly[month_key]["skipped_cb"] += 1
                    continue

                if rolling_pnl_pct < cb_pnl_trip:
                    cb_tripped = True
                    cb_in_probation = False
                    cb_cooldown_remaining = cb_cooldown_days
                    cb_trip_date = date_str
                    cb_trip_reason = f"Rolling P&L {rolling_pnl_pct:.0%} < {cb_pnl_trip:.0%} of bankroll"
                    cb_events.append({
                        "date": date_str,
                        "event": "tripped",
                        "detail": cb_trip_reason,
                    })
                    monthly[month_key]["skipped_cb"] += 1
                    continue

            if cb_enabled and cb_tripped and not cb_in_probation:
                monthly[month_key]["skipped_cb"] += 1
                continue

            trade = compute_trade(
                prob_home, contract_price, bankroll_cents,
                max_pos, min_edge, effective_kelly,
            )
            if trade is None:
                continue

            if day_exposure + trade["cost_cents"] > daily_limit:
                continue

            # Simulate outcome
            if trade["side"] == "yes":
                won = actual_home_win
            else:
                won = not actual_home_win

            if won:
                profit = trade["count"] * (100 - trade["price"])
            else:
                profit = -trade["cost_cents"]

            bankroll_cents += profit
            day_exposure += trade["cost_cents"]
            total_bets += 1
            if won:
                total_wins += 1

            monthly[month_key]["bets"] += 1
            monthly[month_key]["pnl"] += profit
            if won:
                monthly[month_key]["wins"] += 1

            bet_history.append({"won": won, "profit": profit})

            # Track probation
            if cb_enabled and cb_in_probation:
                cb_probation_count += 1
                cb_probation_pnl += profit

            if bankroll_cents <= 0:
                break

        if bankroll_cents <= 0:
            break

    win_rate = total_wins / total_bets if total_bets > 0 else 0
    return {
        "bankroll_cents": bankroll_cents,
        "deposited_cents": deposited_cents,
        "pnl_cents": bankroll_cents - deposited_cents,
        "total_bets": total_bets,
        "total_wins": total_wins,
        "win_rate": win_rate,
        "monthly": dict(monthly),
        "cb_events": cb_events,
    }


# ── Printing helpers ─────────────────────────────────────────────────────────

def print_monthly_table(result, label="", dca_months=None):
    if dca_months is None:
        dca_months = dca_months_for_season("2025-26")
    if label:
        print(f"\n  {label}")
    print(f"  {'Month':<10} {'Games':>6} {'Bets':>6} {'Skip':>6} "
          f"{'Wins':>6} {'WR':>6} {'P&L':>10} {'Balance':>10}")
    print("  " + "-" * 72)

    balance = 0
    for month in dca_months:
        m = result["monthly"].get(month, {
            "games": 0, "bets": 0, "skipped_cb": 0,
            "wins": 0, "pnl": 0, "deposits": 0,
        })
        balance += m["deposits"] + m["pnl"]
        wr = m["wins"] / m["bets"] if m["bets"] > 0 else 0
        print(f"  {month:<10} {m['games']:>6} {m['bets']:>6} {m['skipped_cb']:>6} "
              f"{m['wins']:>6} {wr:>5.0%} {m['pnl']/100:>+10.2f} {balance/100:>10.2f}")

    pnl = result["pnl_cents"] / 100
    dep = result["deposited_cents"] / 100
    roi = pnl / dep * 100 if dep > 0 else 0
    print(f"\n  Total deposited: ${dep:.2f}  |  Final: ${result['bankroll_cents']/100:.2f}  "
          f"|  P&L: ${pnl:+.2f}  |  ROI: {roi:+.1f}%")


def print_cb_events(events):
    if not events:
        print("\n  No circuit breaker events.")
        return
    print(f"\n  Circuit Breaker Event Log:")
    print(f"  {'Date':<12} {'Event':<20} {'Detail'}")
    print("  " + "-" * 80)
    for e in events:
        print(f"  {e['date']:<12} {e['event']:<20} {e['detail']}")


def print_comparison(no_cb, with_cb, feb_mar=None):
    if feb_mar is None:
        feb_mar = ["2026-02", "2026-03"]
    print(f"\n  {'Metric':<25} {'No Breaker':>15} {'With Breaker':>15} {'Delta':>12}")
    print("  " + "-" * 70)
    rows = [
        ("Total bets", f"{no_cb['total_bets']}", f"{with_cb['total_bets']}",
         f"{with_cb['total_bets'] - no_cb['total_bets']:+d}"),
        ("Win rate", f"{no_cb['win_rate']:.1%}", f"{with_cb['win_rate']:.1%}",
         f"{(with_cb['win_rate'] - no_cb['win_rate'])*100:+.1f}pp"),
        ("Final bankroll", f"${no_cb['bankroll_cents']/100:.2f}",
         f"${with_cb['bankroll_cents']/100:.2f}",
         f"${(with_cb['bankroll_cents'] - no_cb['bankroll_cents'])/100:+.2f}"),
        ("P&L", f"${no_cb['pnl_cents']/100:+.2f}",
         f"${with_cb['pnl_cents']/100:+.2f}",
         f"${(with_cb['pnl_cents'] - no_cb['pnl_cents'])/100:+.2f}"),
        ("ROI", f"{no_cb['pnl_cents']/no_cb['deposited_cents']*100:+.1f}%",
         f"${with_cb['pnl_cents']/with_cb['deposited_cents']*100:+.1f}%",
         f"{(with_cb['pnl_cents'] - no_cb['pnl_cents'])/no_cb['deposited_cents']*100:+.1f}pp"),
    ]
    # Feb-Mar P&L
    no_fm = sum(no_cb["monthly"].get(m, {"pnl": 0})["pnl"] for m in feb_mar)
    cb_fm = sum(with_cb["monthly"].get(m, {"pnl": 0})["pnl"] for m in feb_mar)
    rows.append(("Feb-Mar P&L", f"${no_fm/100:+.2f}", f"${cb_fm/100:+.2f}",
                  f"${(cb_fm - no_fm)/100:+.2f}"))

    for label, v1, v2, delta in rows:
        print(f"  {label:<25} {v1:>15} {v2:>15} {delta:>12}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Circuit Breaker DCA Backtest")
    parser.add_argument("--season", default="2025-26",
                        help="Season to simulate, e.g. '2024-25' (default: 2025-26)")
    args = parser.parse_args()

    season = args.season
    dca_months = dca_months_for_season(season)
    feb_mar = feb_mar_months_for_season(season)
    start_year = int(season.split("-")[0])
    train_cutoff = pd.Timestamp(f"{start_year}-10-01")

    print("=" * 75)
    print(f"  Circuit Breaker Backtest — {season} NBA Season (DCA Simulation)")
    print("=" * 75)

    # Load data and train ensemble once
    print("\nLoading dataset...")
    df = load_dataset("dataset_enhanced")
    feature_cols = extract_feature_cols(df)
    df["_season"] = df[DATE_COLUMN].apply(season_from_date)

    train_mask = df[DATE_COLUMN] < train_cutoff
    test_mask = df["_season"] == season

    df_train = df.loc[train_mask]
    df_test = df.loc[test_mask].copy()

    X_train = np.nan_to_num(df_train[feature_cols].astype(float).values, nan=0.0)
    y_train = df_train[TARGET_COLUMN].astype(int).values
    X_test = np.nan_to_num(df_test[feature_cols].astype(float).values, nan=0.0)
    y_test = df_test[TARGET_COLUMN].astype(int).values

    print(f"Training set: {len(X_train)} games | Test set: {len(X_test)} games")

    print("\nTraining stacked ensemble (one-time)...")
    fitted, meta_learner, sigmoid_cal, scaler, model_names = train_ensemble(
        X_train, y_train
    )
    print(f"Base learners: {model_names}")

    model_probs = predict_ensemble(fitted, meta_learner, sigmoid_cal, scaler, X_test)

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_test, (model_probs >= 0.5).astype(int))
    print(f"Model accuracy on {season}: {acc:.1%}")

    print("\nLoading odds...")
    odds_map = load_odds_for_season(season)
    print(f"Loaded {len(odds_map)} odds rows")

    base_cfg = dict(
        kelly_fraction=0.10,
        min_edge=0.05,
        max_pos_pct=0.02,
        daily_limit_pct=0.05,
        dca_months=dca_months,
    )

    # ── Section 1 & 2: Default circuit breaker run ───────────────────────────
    print("\n" + "=" * 75)
    print("  SECTION 1: Month-by-Month (Circuit Breaker ON)")
    print("=" * 75)

    result_cb = run_dca_sim(
        model_probs, y_test, df_test, odds_map,
        cb_enabled=True, cb_window=30, cb_winrate_trip=0.28,
        cb_pnl_trip=-0.15, cb_cooldown_days=3,
        **base_cfg,
    )
    print_monthly_table(result_cb, "Default config: window=30, trip_wr=28%, cooldown=3d",
                        dca_months=dca_months)

    print("\n" + "=" * 75)
    print("  SECTION 2: Circuit Breaker Events")
    print("=" * 75)
    print_cb_events(result_cb["cb_events"])

    # ── Section 3: Comparison ────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("  SECTION 3: No Breaker vs Circuit Breaker")
    print("=" * 75)

    result_no_cb = run_dca_sim(
        model_probs, y_test, df_test, odds_map,
        cb_enabled=False, **base_cfg,
    )
    print_monthly_table(result_no_cb, "No Circuit Breaker:", dca_months=dca_months)
    print()
    print_monthly_table(result_cb, "With Circuit Breaker:", dca_months=dca_months)
    print()
    print_comparison(result_no_cb, result_cb, feb_mar=feb_mar)

    # ── Section 4: Parameter sweep ───────────────────────────────────────────
    print("\n" + "=" * 75)
    print("  SECTION 4: Parameter Sweep")
    print("=" * 75)

    windows = [20, 30, 50]
    wr_trips = [0.25, 0.28, 0.32]
    pnl_trips = [-0.10, -0.15, -0.20]
    cooldowns = [2, 3, 5]

    sweep_results = []

    total_combos = len(windows) * len(wr_trips) * len(pnl_trips) * len(cooldowns)
    print(f"\n  Running {total_combos} configurations...")

    for win, wr_t, pnl_t, cd in itertools.product(windows, wr_trips, pnl_trips, cooldowns):
        r = run_dca_sim(
            model_probs, y_test, df_test, odds_map,
            cb_enabled=True, cb_window=win, cb_winrate_trip=wr_t,
            cb_pnl_trip=pnl_t, cb_cooldown_days=cd,
            **base_cfg,
        )
        fm_pnl = sum(
            r["monthly"].get(m, {"pnl": 0})["pnl"] for m in feb_mar
        )
        sweep_results.append({
            "window": win,
            "wr_trip": wr_t,
            "pnl_trip": pnl_t,
            "cooldown": cd,
            "final": r["bankroll_cents"],
            "pnl": r["pnl_cents"],
            "bets": r["total_bets"],
            "win_rate": r["win_rate"],
            "feb_mar_pnl": fm_pnl,
            "cb_trips": sum(1 for e in r["cb_events"] if e["event"] == "tripped"),
        })

    # Sort by final bankroll descending
    sweep_results.sort(key=lambda x: x["final"], reverse=True)

    print(f"\n  Top 15 configurations (by final bankroll):")
    print(f"  {'Win':>4} {'WR%':>5} {'P&L%':>6} {'CD':>3} "
          f"{'Bets':>5} {'WR':>5} {'Trips':>5} {'FebMar':>9} {'Final':>10} {'P&L':>10}")
    print("  " + "-" * 70)

    for r in sweep_results[:15]:
        print(f"  {r['window']:>4} {r['wr_trip']:>4.0%} {r['pnl_trip']:>+5.0%} {r['cooldown']:>3} "
              f"{r['bets']:>5} {r['win_rate']:>4.0%} {r['cb_trips']:>5} "
              f"${r['feb_mar_pnl']/100:>+8.2f} ${r['final']/100:>9.2f} ${r['pnl']/100:>+9.2f}")

    # Also show baseline
    no_cb_feb_mar = sum(
        result_no_cb["monthly"].get(m, {"pnl": 0})["pnl"] for m in feb_mar
    )
    print(f"\n  Baseline (no breaker): "
          f"Bets={result_no_cb['total_bets']} WR={result_no_cb['win_rate']:.0%} "
          f"FebMar=${no_cb_feb_mar/100:+.2f} "
          f"Final=${result_no_cb['bankroll_cents']/100:.2f} "
          f"P&L=${result_no_cb['pnl_cents']/100:+.2f}")

    # ── Section 5: Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("  SECTION 5: Summary")
    print("=" * 75)

    best = sweep_results[0]
    worst = sweep_results[-1]
    no_cb_pnl = result_no_cb["pnl_cents"]

    print(f"\n  Best config:  window={best['window']}, wr_trip={best['wr_trip']:.0%}, "
          f"pnl_trip={best['pnl_trip']:+.0%}, cooldown={best['cooldown']}d")
    print(f"    Final: ${best['final']/100:.2f}  |  P&L: ${best['pnl']/100:+.2f}  "
          f"|  Feb-Mar: ${best['feb_mar_pnl']/100:+.2f}")

    print(f"\n  Worst config: window={worst['window']}, wr_trip={worst['wr_trip']:.0%}, "
          f"pnl_trip={worst['pnl_trip']:+.0%}, cooldown={worst['cooldown']}d")
    print(f"    Final: ${worst['final']/100:.2f}  |  P&L: ${worst['pnl']/100:+.2f}  "
          f"|  Feb-Mar: ${worst['feb_mar_pnl']/100:+.2f}")

    print(f"\n  No breaker:   Final=${result_no_cb['bankroll_cents']/100:.2f}  "
          f"|  P&L: ${no_cb_pnl/100:+.2f}  |  Feb-Mar: ${no_cb_feb_mar/100:+.2f}")

    # How many configs beat no-breaker?
    better = sum(1 for r in sweep_results if r["pnl"] > no_cb_pnl)
    print(f"\n  {better}/{len(sweep_results)} configurations beat no-breaker on total P&L")

    # How many configs preserved more capital through Feb-Mar?
    better_fm = sum(1 for r in sweep_results if r["feb_mar_pnl"] > no_cb_feb_mar)
    print(f"  {better_fm}/{len(sweep_results)} configurations had better Feb-Mar P&L than no-breaker")

    # Capital preservation
    if best["pnl"] > no_cb_pnl:
        saved = (best["pnl"] - no_cb_pnl) / 100
        print(f"\n  Best circuit breaker saved ${saved:+.2f} vs no-breaker")
    else:
        print(f"\n  No configuration improved on no-breaker P&L")

    print("\n" + "=" * 75)


if __name__ == "__main__":
    main()
