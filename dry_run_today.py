#!/usr/bin/env python
"""
Daily dry-run: show model predictions and hypothetical trades for today's games.

Trains the ensemble from the dataset (same as backtest), then predicts today's
games using each team's most recent features + updated days rest.
Uses odds DB for market prices where available.

Usage:
    python3 dry_run_today.py              # today's games
    python3 dry_run_today.py --date 2026-03-14  # specific date
"""

import argparse
import json
import os
import sqlite3
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

from backtest_kalshi import (
    DATE_COLUMN,
    ODDS_DB,
    TARGET_COLUMN,
    compute_trade,
    extract_feature_cols,
    load_dataset,
    odds_to_kalshi_price,
    predict_ensemble,
    season_from_date,
    train_ensemble,
)

BASE_DIR = Path(__file__).resolve().parent
SCHEDULE_PATH = BASE_DIR / "Data" / "nba-2025-UTC.csv"

# Conservative trading config (matches circuit breaker backtest)
BANKROLL_CENTS = 1000
MAX_POSITION_CENTS = 200
DAILY_LOSS_LIMIT_CENTS = 500
MIN_EDGE = 0.05
KELLY_FRACTION = 0.10


def load_odds_for_date(target_date_str):
    """Load odds for games on a specific date."""
    target = pd.to_datetime(target_date_str)
    # Odds DB dates are 1 day ahead of dataset/schedule dates
    odds_date = target + pd.Timedelta(days=1)
    odds_date_str = odds_date.strftime("%Y-%m-%d")

    odds_map = {}
    with sqlite3.connect(ODDS_DB) as con:
        tables = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        for table in ["odds_2025-26", "2025-26"]:
            if table not in tables:
                continue
            df = pd.read_sql_query(f'SELECT * FROM "{table}"', con)
            df["Date"] = pd.to_datetime(df["Date"])
            day_odds = df[df["Date"].dt.strftime("%Y-%m-%d") == odds_date_str]
            for _, row in day_odds.iterrows():
                key = (row["Home"], row["Away"])
                odds_map[key] = {"ML_Home": row["ML_Home"], "ML_Away": row["ML_Away"]}
    return odds_map


# Kalshi team abbreviation → full name
KALSHI_ABBR = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "LA Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards",
}


def fetch_kalshi_prices(target_date_str):
    """Try to fetch live Kalshi NBA game market prices. Returns dict keyed by
    (home_team, away_team) → contract_price (cents), or empty dict on failure."""
    try:
        import kalshi_python_sync as kalshi
        from kalshi_python_sync.auth import KalshiAuth
    except ImportError:
        return {}

    api_key = os.getenv("KALSHI_API_KEY", "")
    key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "./kalshi_private_key.pem")
    if not api_key or not Path(key_path).exists():
        return {}

    try:
        config = kalshi.Configuration()
        config.host = "https://api.elections.kalshi.com/trade-api/v2"
        client = kalshi.KalshiClient(configuration=config)
        client.kalshi_auth = KalshiAuth(
            key_id=api_key, private_key_pem=open(key_path).read()
        )
        market_api = kalshi.MarketApi(client)

        # Date tag in ticker: e.g. "26MAR15" for 2026-03-15
        dt = pd.to_datetime(target_date_str)
        date_tag = dt.strftime("%y%b%d").upper()  # "26MAR15"

        resp = market_api.get_markets_without_preload_content(
            series_ticker="KXNBAGAME", status="open", limit=200
        )
        data = json.loads(resp.data)
        all_markets = data.get("markets", [])

        # Filter to target date and group by game
        prices = {}
        games = {}  # group markets by game code (e.g. "CLEMIL")
        for m in all_markets:
            ticker = m.get("ticker", "")
            if date_tag not in ticker:
                continue
            # Ticker format: KXNBAGAME-26MAR15UTASAC-UTA
            parts = ticker.split("-")
            if len(parts) < 3:
                continue
            game_code = parts[1].replace(date_tag, "")  # e.g. "UTASAC"
            team_abbr = parts[2]
            price = m.get("yes_ask") or m.get("last_price")
            if team_abbr not in KALSHI_ABBR:
                continue
            games.setdefault(game_code, {})[team_abbr] = price

        # Parse game codes into (away, home) pairs — ticker has away first
        for game_code, team_prices in games.items():
            abbrs = list(team_prices.keys())
            if len(abbrs) != 2:
                continue
            # Determine away/home from title or ticker order (away @ home)
            away_abbr, home_abbr = abbrs[0], abbrs[1]
            # The first abbr in the game code is away
            if game_code.startswith(away_abbr):
                pass
            elif game_code.startswith(home_abbr):
                away_abbr, home_abbr = home_abbr, away_abbr

            home_price = team_prices.get(home_abbr)
            if home_price and home_price > 0:
                home_team = KALSHI_ABBR[home_abbr]
                away_team = KALSHI_ABBR[away_abbr]
                prices[(home_team, away_team)] = int(home_price)

        if prices:
            print(f"[Kalshi] Fetched live prices for {len(prices)} games")
        return prices
    except Exception as e:
        print(f"[Kalshi] Could not fetch markets: {e}")
        return {}


def build_feature_row_from_dataset(home_team, away_team, df, feature_cols,
                                   schedule_df, target_date):
    """Build a feature row for a game by looking up each team's latest stats
    in the dataset and updating days rest."""
    # Find most recent home game for home_team
    home_as_home = df[df["TEAM_NAME"] == home_team].sort_values(DATE_COLUMN, ascending=False)
    # Find most recent game for away_team (as away)
    away_as_away = df[df["TEAM_NAME.1"] == away_team].sort_values(DATE_COLUMN, ascending=False)

    if len(home_as_home) == 0 or len(away_as_away) == 0:
        return None

    # Start from the home team's most recent home game row
    base_row = home_as_home.iloc[0].copy()

    # Overwrite the away-team columns with the away team's most recent away stats
    away_row = away_as_away.iloc[0]
    for col in feature_cols:
        if col.endswith(".1") or col in [
            "Days-Rest-Away", "Elo_Away", "Roll10_PTS_Away", "Roll10_FG_PCT_Away",
            "Roll10_FG3_PCT_Away", "Roll10_REB_Away", "Roll10_AST_Away",
            "Roll10_TOV_Away", "Roll10_PLUS_MINUS_Away", "Away_WinPct_Split",
            "Pace_Away", "PTS_per100_Away", "AST_per100_Away", "TOV_per100_Away",
            "STL_per100_Away", "Is_B2B_Away",
        ]:
            if col in away_row.index:
                base_row[col] = away_row[col]

    # Update days rest from schedule
    target_dt = pd.to_datetime(target_date)
    for team, rest_col in [(home_team, "Days-Rest-Home"), (away_team, "Days-Rest-Away")]:
        team_games = schedule_df[
            (schedule_df["Home Team"] == team) | (schedule_df["Away Team"] == team)
        ]
        prev = team_games.loc[team_games["Date"] < target_dt].sort_values(
            "Date", ascending=False
        ).head(1)["Date"]
        if len(prev) > 0:
            days_off = (target_dt - prev.iloc[0]).days + 1
        else:
            days_off = 7
        if rest_col in base_row.index:
            base_row[rest_col] = days_off

    # Update B2B flags
    if "Is_B2B_Home" in base_row.index:
        base_row["Is_B2B_Home"] = 1 if base_row.get("Days-Rest-Home", 7) <= 1 else 0
    if "Is_B2B_Away" in base_row.index:
        base_row["Is_B2B_Away"] = 1 if base_row.get("Days-Rest-Away", 7) <= 1 else 0
    if "Both_B2B" in base_row.index:
        base_row["Both_B2B"] = 1 if base_row.get("Is_B2B_Home", 0) and base_row.get("Is_B2B_Away", 0) else 0
    if "B2B_Advantage" in base_row.index:
        base_row["B2B_Advantage"] = int(base_row.get("Is_B2B_Away", 0)) - int(base_row.get("Is_B2B_Home", 0))
    if "Rest_Advantage" in base_row.index:
        base_row["Rest_Advantage"] = base_row.get("Days-Rest-Home", 0) - base_row.get("Days-Rest-Away", 0)

    # Extract feature values in correct order
    values = []
    for col in feature_cols:
        if col in base_row.index:
            values.append(float(base_row[col]) if pd.notna(base_row[col]) else 0.0)
        else:
            values.append(0.0)

    return np.array(values)


def main():
    parser = argparse.ArgumentParser(description="Daily dry-run predictions")
    parser.add_argument("--date", default=None, help="Date YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    if args.date:
        target_date = pd.to_datetime(args.date)
    else:
        target_date = pd.to_datetime(datetime.today().strftime("%Y-%m-%d"))

    target_str = target_date.strftime("%Y-%m-%d")

    print("=" * 75)
    print(f"  NBA Trader Dry Run — {target_str}")
    print("=" * 75)

    # Load dataset and train ensemble
    print("\nLoading dataset...")
    df = load_dataset("dataset_enhanced")
    feature_cols = extract_feature_cols(df)
    df["_season"] = df[DATE_COLUMN].apply(season_from_date)

    # Train on everything before target date
    train_mask = df[DATE_COLUMN] < target_date
    df_train = df.loc[train_mask]

    X_train = np.nan_to_num(df_train[feature_cols].astype(float).values, nan=0.0)
    y_train = df_train[TARGET_COLUMN].astype(int).values

    print(f"Training on {len(X_train)} games (all data before {target_str})")

    print("Training stacked ensemble...")
    fitted, meta_learner, sigmoid_cal, scaler, model_names = train_ensemble(
        X_train, y_train
    )
    print(f"Base learners: {model_names}")

    # Check if target date has games in the dataset (already played)
    target_games = df[df[DATE_COLUMN].dt.strftime("%Y-%m-%d") == target_str]
    has_results = len(target_games) > 0

    # Load schedule for upcoming games
    schedule_df = pd.read_csv(
        SCHEDULE_PATH, parse_dates=["Date"], date_format="%d/%m/%Y %H:%M"
    )
    games_today = schedule_df[
        schedule_df["Date"].dt.strftime("%Y-%m-%d") == target_str
    ].copy()

    if len(games_today) == 0 and not has_results:
        print(f"\nNo games found for {target_str}")
        return

    # Load market prices: try Kalshi first, then odds DB
    kalshi_prices = fetch_kalshi_prices(target_str)
    odds_map = load_odds_for_date(target_str)

    # Determine which games to predict
    if has_results:
        print(f"\nFound {len(target_games)} completed games in dataset for {target_str}")
        # Use actual dataset features (most accurate)
        X_today = np.nan_to_num(target_games[feature_cols].astype(float).values, nan=0.0)
        y_today = target_games[TARGET_COLUMN].astype(int).values
        model_probs = predict_ensemble(fitted, meta_learner, sigmoid_cal, scaler, X_today)

        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_today, (model_probs >= 0.5).astype(int))
        print(f"Model accuracy on these games: {acc:.1%}")

        games_data = []
        for i, (_, row) in enumerate(target_games.iterrows()):
            home = row["TEAM_NAME"]
            away = row["TEAM_NAME.1"]
            actual = bool(row[TARGET_COLUMN] == 1)
            games_data.append({
                "home": home, "away": away,
                "model_prob": model_probs[i],
                "actual_home_win": actual,
                "has_result": True,
            })
    else:
        print(f"\nFound {len(games_today)} scheduled games for {target_str} (upcoming)")
        games_data = []
        for _, game in games_today.iterrows():
            home = game["Home Team"]
            away = game["Away Team"]
            tip_time = game["Date"].strftime("%H:%M UTC")

            features = build_feature_row_from_dataset(
                home, away, df, feature_cols, schedule_df, target_date
            )
            if features is None:
                print(f"  Skipping {away} @ {home} — team data not found")
                continue

            X = features.reshape(1, -1)
            prob = predict_ensemble(fitted, meta_learner, sigmoid_cal, scaler, X)[0]
            games_data.append({
                "home": home, "away": away,
                "model_prob": prob,
                "has_result": False,
                "tip_time": tip_time,
            })

    if kalshi_prices:
        print(f"Kalshi live prices for {len(kalshi_prices)} games")
    if odds_map:
        print(f"Odds DB prices for {len(odds_map)} games")
    if not kalshi_prices and not odds_map:
        print("No market prices available (Kalshi or odds DB)")

    # Display predictions
    print("\n" + "-" * 80)
    header = (f"  {'Matchup':<38} {'Model':>6} {'Mkt':>5} {'Edge':>6} "
              f"{'Side':>5} {'Qty':>4} {'Cost':>7}")
    if has_results:
        header += f" {'Result':>8}"
    else:
        header += f" {'Tip':>10}"
    print(header)
    print("-" * 80)

    trades = []
    total_cost = 0
    daily_exposure = 0
    wins = 0
    losses = 0
    total_pnl = 0

    for g in games_data:
        matchup = f"{g['away']} @ {g['home']}"
        prob = g["model_prob"]

        # Price lookup: Kalshi live → odds DB fallback
        contract_price = kalshi_prices.get((g["home"], g["away"]))
        price_src = "kalshi" if contract_price else None
        if not contract_price:
            odds_info = odds_map.get((g["home"], g["away"]))
            if odds_info:
                contract_price = odds_to_kalshi_price(odds_info["ML_Home"], odds_info["ML_Away"])
                price_src = "odds"
        if contract_price and 0 < contract_price < 100:
            trade = compute_trade(
                prob, contract_price, BANKROLL_CENTS,
                MAX_POSITION_CENTS, MIN_EDGE, KELLY_FRACTION,
            )
        else:
            trade = None
            contract_price = None

        # Format output
        mkt_str = f"{contract_price}c" if contract_price else "  n/a"
        if trade:
            edge_str = f"{trade['edge']:.1%}"
            side_str = trade["side"].upper()
            if daily_exposure + trade["cost_cents"] <= DAILY_LOSS_LIMIT_CENTS:
                qty_str = str(trade["count"])
                cost_str = f"${trade['cost_cents']/100:.2f}"
                daily_exposure += trade["cost_cents"]
                total_cost += trade["cost_cents"]
                trades.append({"game": g, "trade": trade})
            else:
                qty_str = "LIM"
                cost_str = "-"
        else:
            edge_str = "  n/a" if not contract_price else f"{abs(prob - contract_price/100):.1%}"
            side_str = "PASS"
            qty_str = "-"
            cost_str = "-"

        tail = ""
        if g["has_result"]:
            actual = g["actual_home_win"]
            predicted_home = prob >= 0.5
            correct = predicted_home == actual
            winner = g["home"] if actual else g["away"]

            # Compute P&L if we would have traded
            if trade:
                if trade["side"] == "yes":
                    won = actual
                else:
                    won = not actual
                if won:
                    profit = trade["count"] * (100 - trade["price"])
                    wins += 1
                else:
                    profit = -trade["cost_cents"]
                    losses += 1
                total_pnl += profit
                pnl_str = f"${profit/100:+.2f}"
                tail = f" {'OK' if correct else 'X':>3} {pnl_str:>7}" if trade else f" {'OK' if correct else 'X':>3}"
            else:
                tail = f" {'OK' if correct else 'X':>5}"
        else:
            tail = f" {g.get('tip_time', ''):>10}"

        print(f"  {matchup:<38} {prob:>5.0%} {mkt_str:>5} {edge_str:>6} "
              f"{side_str:>5} {qty_str:>4} {cost_str:>7}{tail}")

    # Summary
    print("\n" + "=" * 80)
    print("  PREDICTIONS SUMMARY")
    print("=" * 80)

    for g in sorted(games_data, key=lambda x: max(x["model_prob"], 1-x["model_prob"]), reverse=True):
        winner = g["home"] if g["model_prob"] >= 0.5 else g["away"]
        conf = max(g["model_prob"], 1 - g["model_prob"])
        marker = ""
        if g["has_result"]:
            actual_winner = g["home"] if g["actual_home_win"] else g["away"]
            marker = " OK" if winner == actual_winner else " X"
        print(f"  {winner:<30} {conf:>5.0%} confidence{marker}")

    if trades:
        print(f"\n  HYPOTHETICAL TRADES ({len(trades)} games with edge >= {MIN_EDGE:.0%}):")
        print(f"  {'Game':<38} {'Side':>5} {'Qty':>4} {'Price':>6} {'Cost':>7} {'Edge':>6}")
        print("  " + "-" * 70)
        for t in trades:
            tr = t["trade"]
            matchup = f"{t['game']['away']} @ {t['game']['home']}"
            print(f"  {matchup:<38} {tr['side'].upper():>5} {tr['count']:>4} "
                  f"{tr['price']:>5}c ${tr['cost_cents']/100:>6.2f} {tr['edge']:>5.1%}")

        print(f"\n  Total exposure: ${total_cost/100:.2f} / "
              f"${DAILY_LOSS_LIMIT_CENTS/100:.2f} daily limit")

        if has_results and trades:
            print(f"\n  ACTUAL P&L: ${total_pnl/100:+.2f}  "
                  f"({wins}W-{losses}L, {wins/(wins+losses):.0%} win rate)")
        else:
            max_profit = sum(t["trade"]["count"] * (100 - t["trade"]["price"]) for t in trades)
            max_loss = total_cost
            print(f"  If all win:  +${max_profit/100:.2f}")
            print(f"  If all lose: -${max_loss/100:.2f}")
    else:
        print(f"\n  No trades (no games with edge >= {MIN_EDGE:.0%} and available odds)")

    print(f"\n  Config: bankroll=${BANKROLL_CENTS/100:.2f} kelly={KELLY_FRACTION} "
          f"min_edge={MIN_EDGE:.0%} max_pos=${MAX_POSITION_CENTS/100:.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
