"""Live NBA trading bot for Kalshi.

Loads the stacked ensemble model, builds feature vectors for today's games,
compares model probabilities against Kalshi contract prices, and places
limit orders when edge exceeds threshold.

Usage:
    python -m src.Bot.live_trader             # live trading
    python -m src.Bot.live_trader --dry-run   # log decisions only
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]
ENSEMBLE_PATH = BASE_DIR / "Models" / "Ensemble_Models" / "stacked_ensemble_ML.pkl"
CALIBRATION_PATH = BASE_DIR / "Models" / "Ensemble_Models" / "stacked_ensemble_ML_calibration.pkl"
SCHEDULE_PATH = BASE_DIR / "Data" / "nba-2025-UTC.csv"
LOG_DIR = BASE_DIR / "logs"

DATA_URL = (
    "https://stats.nba.com/stats/leaguedashteamstats?"
    "Conference=&DateFrom=&DateTo=&Division=&GameScope=&GameSegment=&Height="
    "&ISTRound=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0"
    "&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0"
    "&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2025-26"
    "&SeasonSegment=&SeasonType=Regular%20Season&ShotClockRange=&StarterBench="
    "&TeamID=0&TwoWay=0&VsConference=&VsDivision="
)

# Config from .env
BANKROLL_CENTS = int(os.getenv("BANKROLL_CENTS", "1000"))
MAX_POSITION_CENTS = int(os.getenv("MAX_POSITION_CENTS", "200"))
DAILY_LOSS_LIMIT_CENTS = int(os.getenv("DAILY_LOSS_LIMIT_CENTS", "500"))
MIN_EDGE = float(os.getenv("MIN_EDGE", "0.05"))
KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.25"))


# ── Ensemble loading ─────────────────────────────────────────────────────

def load_ensemble():
    """Load the stacked ensemble model and calibration.

    Returns
    -------
    tuple
        (base_learners dict, meta_learner, sigmoid_calibrator, scaler,
         model_names, scaled_models set)
    """
    payload = joblib.load(ENSEMBLE_PATH)
    sigmoid_cal = None
    if CALIBRATION_PATH.exists():
        sigmoid_cal = joblib.load(CALIBRATION_PATH)

    return (
        payload["base_learners"],
        payload["meta_learner"],
        sigmoid_cal,
        payload["scaler"],
        payload["model_names"],
        payload["scaled_models"],
    )


def ensemble_predict(data, base_learners, meta_learner, sigmoid_cal, scaler,
                     model_names, scaled_models):
    """Run ensemble inference on feature matrix.

    Parameters
    ----------
    data : np.ndarray
        Feature matrix, shape (n_games, n_features).

    Returns
    -------
    np.ndarray
        P(home_win) for each game, shape (n_games,).
    """
    n = data.shape[0]
    meta_features = np.zeros((n, len(model_names)))

    for i, name in enumerate(model_names):
        model = base_learners[name]
        if name in scaled_models:
            X = scaler.transform(data)
        else:
            X = data
        probs = model.predict_proba(X)
        meta_features[:, i] = probs[:, 1]  # P(home_win)

    # Meta-learner predicts P(home_win)
    meta_probs = meta_learner.predict_proba(meta_features)[:, 1]

    # Apply sigmoid calibration if available
    if sigmoid_cal is not None:
        meta_probs = sigmoid_cal.predict(meta_probs)

    return meta_probs


# ── Feature building (reuses main.py logic) ──────────────────────────────

def build_game_features(home_team, away_team, df, schedule_df, today):
    """Build a feature vector for a single game.

    Parameters
    ----------
    home_team, away_team : str
        Full team names (matching team_index_current keys).
    df : pd.DataFrame
        Current season team stats from NBA API.
    schedule_df : pd.DataFrame
        Schedule with 'Date', 'Home Team', 'Away Team' columns.
    today : datetime
        Current date.

    Returns
    -------
    np.ndarray or None
        Feature row as float array, or None if team not found.
    """
    from src.Utils.Dictionaries import team_index_current

    if home_team not in team_index_current or away_team not in team_index_current:
        print(f"[Trader] Team not found: {home_team} or {away_team}")
        return None

    # Days rest calculation
    home_games = schedule_df[
        (schedule_df["Home Team"] == home_team) | (schedule_df["Away Team"] == home_team)
    ]
    away_games = schedule_df[
        (schedule_df["Home Team"] == away_team) | (schedule_df["Away Team"] == away_team)
    ]

    prev_home = home_games.loc[home_games["Date"] <= today].sort_values(
        "Date", ascending=False
    ).head(1)["Date"]
    prev_away = away_games.loc[away_games["Date"] <= today].sort_values(
        "Date", ascending=False
    ).head(1)["Date"]

    home_days_off = (
        timedelta(days=1) + today - prev_home.iloc[0]
        if len(prev_home) > 0
        else timedelta(days=7)
    )
    away_days_off = (
        timedelta(days=1) + today - prev_away.iloc[0]
        if len(prev_away) > 0
        else timedelta(days=7)
    )

    home_series = df.iloc[team_index_current[home_team]]
    away_series = df.iloc[team_index_current[away_team]]
    stats = pd.concat([home_series, away_series])
    stats["Days-Rest-Home"] = home_days_off.days
    stats["Days-Rest-Away"] = away_days_off.days

    # Drop non-numeric columns that the model doesn't use
    row = stats.drop(labels=["TEAM_ID", "TEAM_NAME"], errors="ignore")
    return row.values.astype(float)


# ── Edge & sizing ─────────────────────────────────────────────────────────

def find_edge(model_prob, market_price_cents):
    """Compare model probability to Kalshi market price.

    Parameters
    ----------
    model_prob : float
        P(home_win) from ensemble model (0-1).
    market_price_cents : int
        Kalshi YES contract price in cents (1-99).

    Returns
    -------
    dict
        edge, side ('yes'/'no'), recommended_size_cents
    """
    implied_prob = market_price_cents / 100.0
    edge_yes = model_prob - implied_prob
    edge_no = (1 - model_prob) - (1 - implied_prob)  # same magnitude, opposite sign

    if edge_yes >= MIN_EDGE:
        side = "yes"
        edge = edge_yes
        # Kelly sizing: contract pays (100 / price) decimal odds
        decimal_odds = 100 / market_price_cents  # e.g., 40c → 2.5x
        b = decimal_odds - 1
        p = model_prob
        q = 1 - p
        kelly_pct = ((b * p - q) / b) * KELLY_FRACTION if b > 0 else 0
        kelly_pct = max(0, kelly_pct)
    elif abs(edge_no) >= MIN_EDGE and edge_yes < 0:
        side = "no"
        edge = -edge_yes  # positive edge on no side
        no_price = 100 - market_price_cents
        decimal_odds = 100 / no_price if no_price > 0 else 1
        b = decimal_odds - 1
        p = 1 - model_prob
        q = model_prob
        kelly_pct = ((b * p - q) / b) * KELLY_FRACTION if b > 0 else 0
        kelly_pct = max(0, kelly_pct)
    else:
        return {"edge": abs(edge_yes), "side": None, "recommended_size_cents": 0}

    # Kelly % of bankroll, capped at MAX_POSITION_CENTS
    size_cents = int(kelly_pct * BANKROLL_CENTS)
    size_cents = max(0, min(size_cents, MAX_POSITION_CENTS))

    return {"edge": round(edge, 4), "side": side, "recommended_size_cents": size_cents}


# ── Trade execution ───────────────────────────────────────────────────────

def execute_trades(edges, client, dry_run=False):
    """Place limit orders for games with sufficient edge.

    Parameters
    ----------
    edges : list[dict]
        Each dict has: ticker, home_team, away_team, model_prob,
        market_price, edge_info (from find_edge).
    client : KalshiClient
    dry_run : bool
        If True, log but don't place orders.

    Returns
    -------
    list[dict]
        Trade results.
    """
    results = []
    daily_exposure = 0

    for trade in edges:
        edge_info = trade["edge_info"]
        if edge_info["side"] is None or edge_info["recommended_size_cents"] <= 0:
            continue

        # Daily loss limit check
        if daily_exposure + edge_info["recommended_size_cents"] > DAILY_LOSS_LIMIT_CENTS:
            print(f"[Trader] Daily loss limit reached, skipping {trade['ticker']}")
            continue

        side = edge_info["side"]
        price_cents = (
            trade["market_price"]
            if side == "yes"
            else 100 - trade["market_price"]
        )
        count = max(1, edge_info["recommended_size_cents"] // price_cents)

        result = {
            "ticker": trade["ticker"],
            "home_team": trade["home_team"],
            "away_team": trade["away_team"],
            "model_prob": round(trade["model_prob"], 4),
            "market_price_cents": trade["market_price"],
            "edge": edge_info["edge"],
            "side": side,
            "count": count,
            "price_cents": price_cents,
            "dry_run": dry_run,
            "timestamp": datetime.now().isoformat(),
        }

        if dry_run:
            print(
                f"[DRY RUN] {trade['home_team']} vs {trade['away_team']}: "
                f"BUY {count}x {side.upper()} @ {price_cents}c "
                f"(edge={edge_info['edge']:.1%}, model={trade['model_prob']:.1%})"
            )
            result["status"] = "dry_run"
        else:
            order = client.place_order(
                ticker=trade["ticker"],
                side=side,
                count=count,
                price_cents=price_cents,
            )
            if order:
                result["order_id"] = order["order_id"]
                result["status"] = order["status"]
                print(
                    f"[LIVE] Placed: {count}x {side.upper()} @ {price_cents}c "
                    f"on {trade['ticker']} — {order['status']}"
                )
            else:
                result["status"] = "failed"
                print(f"[LIVE] Order FAILED for {trade['ticker']}")

        daily_exposure += edge_info["recommended_size_cents"]
        results.append(result)

    return results


# ── Logging ───────────────────────────────────────────────────────────────

def log_trades(results):
    """Write trade results to a daily JSON log file."""
    LOG_DIR.mkdir(exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_path = LOG_DIR / f"trades_{date_str}.json"

    existing = []
    if log_path.exists():
        with open(log_path, "r") as f:
            existing = json.load(f)

    existing.extend(results)

    with open(log_path, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"[Trader] Logged {len(results)} trades to {log_path}")


# ── Main entry ────────────────────────────────────────────────────────────

def run(dry_run=False):
    """Main bot entry point.

    1. Load ensemble model
    2. Fetch team stats from NBA API
    3. Build features for today's games
    4. Run ensemble predictions
    5. Fetch Kalshi markets and compare
    6. Place orders where edge > threshold
    """
    from src.DataProviders.KalshiClient import KalshiClient
    from src.Utils.tools import get_json_data, to_data_frame

    print("=" * 60)
    print(f"  NBA Kalshi Trader — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print(f"  Bankroll: ${BANKROLL_CENTS / 100:.2f} | "
          f"Max/game: ${MAX_POSITION_CENTS / 100:.2f} | "
          f"Daily limit: ${DAILY_LOSS_LIMIT_CENTS / 100:.2f}")
    print(f"  Kelly fraction: {KELLY_FRACTION} | Min edge: {MIN_EDGE:.0%}")
    print("=" * 60)

    # 1. Load ensemble
    print("[Trader] Loading ensemble model...")
    base_learners, meta_learner, sigmoid_cal, scaler, model_names, scaled_models = (
        load_ensemble()
    )
    print(f"[Trader] Loaded {len(model_names)} base learners: {model_names}")

    # 2. Fetch team stats
    print("[Trader] Fetching current season stats...")
    stats_json = get_json_data(DATA_URL)
    df = to_data_frame(stats_json)

    # 3. Load schedule
    schedule_df = pd.read_csv(
        SCHEDULE_PATH, parse_dates=["Date"], date_format="%d/%m/%Y %H:%M"
    )
    today = datetime.today()

    # 4. Initialize Kalshi client
    client = KalshiClient()

    # 5. Get Kalshi NBA markets for today
    print("[Trader] Fetching Kalshi NBA markets...")
    markets = client.get_nba_game_markets(today.strftime("%Y-%m-%d"))
    if not markets:
        print("[Trader] No NBA markets found on Kalshi today.")
        return

    print(f"[Trader] Found {len(markets)} NBA markets")

    # 6. For each market, build features and predict
    edges = []
    for market in markets:
        home_team = market["home_team"]
        away_team = market["away_team"]

        features = build_game_features(home_team, away_team, df, schedule_df, today)
        if features is None:
            continue

        # Run ensemble prediction
        data = features.reshape(1, -1)
        model_prob = ensemble_predict(
            data, base_learners, meta_learner, sigmoid_cal, scaler,
            model_names, scaled_models,
        )[0]

        # Get market price
        market_price = market["yes_price"]
        if market_price is None or market_price <= 0 or market_price >= 100:
            continue

        # Compute edge
        edge_info = find_edge(model_prob, market_price)

        print(
            f"  {home_team} vs {away_team}: "
            f"model={model_prob:.1%} market={market_price}c "
            f"edge={edge_info['edge']:.1%} side={edge_info['side'] or 'PASS'}"
        )

        edges.append({
            "ticker": market["ticker"],
            "home_team": home_team,
            "away_team": away_team,
            "model_prob": model_prob,
            "market_price": market_price,
            "edge_info": edge_info,
        })

    # 7. Execute trades
    tradeable = [e for e in edges if e["edge_info"]["side"] is not None]
    if tradeable:
        print(f"\n[Trader] {len(tradeable)} games with edge >= {MIN_EDGE:.0%}")
        results = execute_trades(tradeable, client, dry_run=dry_run)
        log_trades(results)
    else:
        print(f"\n[Trader] No games with edge >= {MIN_EDGE:.0%}")
        # Still log the scan
        log_trades([{
            "timestamp": datetime.now().isoformat(),
            "action": "scan",
            "markets_checked": len(markets),
            "edges_found": 0,
        }])

    print("\n[Trader] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NBA Kalshi Live Trader")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log decisions without placing real orders",
    )
    args = parser.parse_args()
    run(dry_run=args.dry_run)
