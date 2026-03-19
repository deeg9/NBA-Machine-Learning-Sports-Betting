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
INJURY_CACHE_PATH = LOG_DIR / "injury_cache.json"

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

# Module toggles
ENABLE_BAYESIAN_UPDATES = os.getenv("ENABLE_BAYESIAN_UPDATES", "false").lower() == "true"
ENABLE_KL_DIVERGENCE = os.getenv("ENABLE_KL_DIVERGENCE", "false").lower() == "true"
ENABLE_STOIKOV_EXECUTION = os.getenv("ENABLE_STOIKOV_EXECUTION", "false").lower() == "true"

# Bayesian config
BAYESIAN_CONFIG = {
    "use_line_movement": os.getenv("BAYESIAN_USE_LINE_MOVEMENT", "true").lower() == "true",
    "use_injuries": os.getenv("BAYESIAN_USE_INJURIES", "true").lower() == "true",
    "use_reverse_line": os.getenv("BAYESIAN_USE_REVERSE_LINE", "true").lower() == "true",
    "line_strength": float(os.getenv("BAYESIAN_LINE_STRENGTH", "1.0")),
    "injury_strength": float(os.getenv("BAYESIAN_INJURY_STRENGTH", "1.0")),
    "max_total_shift": float(os.getenv("BAYESIAN_MAX_SHIFT", "0.15")),
}

# KL Divergence config
KL_THRESHOLD = float(os.getenv("KL_THRESHOLD", "0.05"))

# Stoikov config
STOIKOV_GAMMA = float(os.getenv("STOIKOV_GAMMA", "0.1"))
STOIKOV_MAX_IMPROVEMENT = int(os.getenv("STOIKOV_MAX_IMPROVEMENT", "5"))


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

def get_position_for_ticker(positions, ticker):
    """Extract net position count for a specific ticker."""
    for pos in positions:
        if hasattr(pos, "ticker") and pos.ticker == ticker:
            yes_count = getattr(pos, "yes_count", 0) or 0
            no_count = getattr(pos, "no_count", 0) or 0
            return yes_count - no_count
        if isinstance(pos, dict) and pos.get("ticker") == ticker:
            return pos.get("yes_count", 0) - pos.get("no_count", 0)
    return 0


def parse_close_time(close_time_str):
    """Parse Kalshi close_time string into datetime."""
    if close_time_str is None:
        return datetime.now() + timedelta(hours=4)  # default 4h
    if isinstance(close_time_str, datetime):
        return close_time_str
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(close_time_str, fmt)
        except ValueError:
            continue
    return datetime.now() + timedelta(hours=4)


def execute_trades(edges, client, dry_run=False):
    """Place limit orders for games with sufficient edge.

    Parameters
    ----------
    edges : list[dict]
        Each dict has: ticker, home_team, away_team, model_prob,
        market_price, close_time, edge_info (from find_edge).
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

    # Pre-fetch positions once if Stoikov is enabled
    positions = client.get_positions() if ENABLE_STOIKOV_EXECUTION else []

    for trade in edges:
        edge_info = trade["edge_info"]
        if edge_info["side"] is None or edge_info["recommended_size_cents"] <= 0:
            continue

        # Daily loss limit check
        if daily_exposure + edge_info["recommended_size_cents"] > DAILY_LOSS_LIMIT_CENTS:
            print(f"[Trader] Daily loss limit reached, skipping {trade['ticker']}")
            continue

        side = edge_info["side"]

        # Stoikov execution: calculate optimal limit price
        if ENABLE_STOIKOV_EXECUTION:
            from src.Utils.StoikovExecution import optimal_limit_price

            current_pos = get_position_for_ticker(positions, trade["ticker"])
            close_time = parse_close_time(trade.get("close_time"))
            hours_left = max(0, (close_time - datetime.now()).total_seconds() / 3600)

            orderbook = client.get_orderbook(trade["ticker"], depth=5)
            exec_result = optimal_limit_price(
                model_prob=trade["model_prob"],
                current_position=current_pos,
                time_to_close_hours=hours_left,
                orderbook=orderbook,
                side=side,
                gamma=STOIKOV_GAMMA,
                max_improvement=STOIKOV_MAX_IMPROVEMENT,
            )
            price_cents = exec_result["optimal_price"]
            naive_price = trade["market_price"] if side == "yes" else 100 - trade["market_price"]
            if exec_result["improvement_cents"] > 0:
                print(f"  [Stoikov] {trade['ticker']}: naive={naive_price}c "
                      f"optimal={price_cents}c (saving {exec_result['improvement_cents']}c, "
                      f"fill_prob={exec_result['fill_probability']:.0%})")
        else:
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


# ── Injury data ──────────────────────────────────────────────────────────

def load_injury_snapshot():
    """Fetch current injuries from ESPN and cache to disk for diffing."""
    from src.DataProviders.PlayerDataProvider import fetch_injury_data
    injury_df = fetch_injury_data()
    if not injury_df.empty:
        LOG_DIR.mkdir(exist_ok=True)
        injury_df.to_json(INJURY_CACHE_PATH, orient="records", date_format="iso")
    return injury_df


def load_previous_injury_snapshot():
    """Load the cached injury snapshot from the previous run."""
    if not INJURY_CACHE_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_json(INJURY_CACHE_PATH, orient="records")
    except Exception:
        return pd.DataFrame()


def compute_injury_delta(home_team, away_team, current_df, previous_df):
    """Diff current vs previous injury snapshots to get new OUT players."""
    from src.Features.injury_features import compute_live_injury_counts
    current = compute_live_injury_counts(home_team, away_team, current_df)
    previous = compute_live_injury_counts(home_team, away_team, previous_df)
    return {
        "home_out_delta": max(0, current["Injuries_Out_Home"] - previous["Injuries_Out_Home"]),
        "away_out_delta": max(0, current["Injuries_Out_Away"] - previous["Injuries_Out_Away"]),
    }


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
    modules = []
    if ENABLE_BAYESIAN_UPDATES:
        modules.append("Bayesian")
    if ENABLE_KL_DIVERGENCE:
        modules.append("KL-Divergence")
    if ENABLE_STOIKOV_EXECUTION:
        modules.append("Stoikov")
    if modules:
        print(f"  Modules: {', '.join(modules)}")
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

    # 4b. Fetch injury data for Bayesian updates
    if ENABLE_BAYESIAN_UPDATES and BAYESIAN_CONFIG["use_injuries"]:
        print("[Trader] Fetching injury data from ESPN...")
        previous_injuries = load_previous_injury_snapshot()
        current_injuries = load_injury_snapshot()
        print(f"[Trader] {len(current_injuries)} injury records loaded"
              f" ({len(previous_injuries)} cached from previous run)")
    else:
        current_injuries = pd.DataFrame()
        previous_injuries = pd.DataFrame()

    # 5. Get Kalshi NBA markets for today
    print("[Trader] Fetching Kalshi NBA markets...")
    markets = client.get_nba_game_markets(today.strftime("%Y-%m-%d"))
    if not markets:
        print("[Trader] No NBA markets found on Kalshi today.")
        return

    print(f"[Trader] Found {len(markets)} NBA markets")

    # 6. For each market, build features and predict
    price_cache = {}  # {ticker: previous_price} for Bayesian line movement
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

        # Bayesian update: adjust model_prob with real-time signals
        if ENABLE_BAYESIAN_UPDATES:
            from src.Utils.BayesianUpdater import update_probability

            orderbook = client.get_orderbook(market["ticker"])
            snapshot = {
                "current_price": market_price,
                "previous_price": price_cache.get(market["ticker"], market_price),
                "yes_volume": orderbook.get("yes_volume"),
                "no_volume": orderbook.get("no_volume"),
            }
            prior = model_prob
            inj_delta = compute_injury_delta(
                home_team, away_team, current_injuries, previous_injuries
            ) if not current_injuries.empty else None
            model_prob, update_log = update_probability(
                model_prob, snapshot, inj_delta, BAYESIAN_CONFIG,
            )
            price_cache[market["ticker"]] = market_price
            if update_log:
                print(f"  [Bayesian] {home_team}: {prior:.1%} → {model_prob:.1%} "
                      f"({len(update_log)} signals)")

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
            "close_time": market.get("close_time"),
            "edge_info": edge_info,
        })

    # 7. KL Divergence: rank trades by information-theoretic mispricing
    if ENABLE_KL_DIVERGENCE and edges:
        from src.Utils.KLDivergence import detect_mispricing

        kl_results = detect_mispricing(
            [{"ticker": e["ticker"], "model_prob": e["model_prob"],
              "market_price_cents": e["market_price"]} for e in edges],
            threshold=KL_THRESHOLD,
        )
        kl_map = {r["ticker"]: r for r in kl_results}
        for e in edges:
            kl_info = kl_map.get(e["ticker"])
            if kl_info:
                e["kl_divergence"] = kl_info["kl_divergence"]
                e["kl_confirmed"] = True
                print(f"  [KL] {e['ticker']}: KL={kl_info['kl_divergence']:.4f} "
                      f"{kl_info['direction']}")

    # 8. Execute trades
    tradeable = [e for e in edges if e["edge_info"]["side"] is not None]
    if ENABLE_KL_DIVERGENCE:
        tradeable.sort(key=lambda e: (
            not e.get("kl_confirmed", False), -e["edge_info"]["edge"]
        ))
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


def scan_games():
    """Run the full prediction pipeline and return structured results for the dashboard.

    Returns list of dicts with all prediction data for every game (not just tradeable ones).
    Also saves to logs/scan_YYYY-MM-DD.json.
    """
    from src.DataProviders.KalshiClient import KalshiClient
    from src.Utils.tools import get_json_data, to_data_frame

    base_learners, meta_learner, sigmoid_cal, scaler, model_names, scaled_models = (
        load_ensemble()
    )
    stats_json = get_json_data(DATA_URL)
    df = to_data_frame(stats_json)
    schedule_df = pd.read_csv(
        SCHEDULE_PATH, parse_dates=["Date"], date_format="%d/%m/%Y %H:%M"
    )
    today = datetime.today()
    client = KalshiClient()

    # Injury data
    if ENABLE_BAYESIAN_UPDATES and BAYESIAN_CONFIG["use_injuries"]:
        previous_injuries = load_previous_injury_snapshot()
        current_injuries = load_injury_snapshot()
    else:
        current_injuries = pd.DataFrame()
        previous_injuries = pd.DataFrame()

    markets = client.get_nba_game_markets(today.strftime("%Y-%m-%d"))
    if not markets:
        return []

    predictions = []
    price_cache = {}

    for market in markets:
        home_team = market["home_team"]
        away_team = market["away_team"]

        features = build_game_features(home_team, away_team, df, schedule_df, today)
        if features is None:
            continue

        data = features.reshape(1, -1)
        model_prob = float(ensemble_predict(
            data, base_learners, meta_learner, sigmoid_cal, scaler,
            model_names, scaled_models,
        )[0])

        market_price = market["yes_price"]
        if market_price is None or market_price <= 0 or market_price >= 100:
            continue

        # Bayesian update
        bayesian_log = []
        injuries_home = 0
        injuries_away = 0
        if ENABLE_BAYESIAN_UPDATES:
            from src.Utils.BayesianUpdater import update_probability

            orderbook = client.get_orderbook(market["ticker"])
            snapshot = {
                "current_price": market_price,
                "previous_price": price_cache.get(market["ticker"], market_price),
                "yes_volume": orderbook.get("yes_volume"),
                "no_volume": orderbook.get("no_volume"),
            }
            inj_delta = None
            if not current_injuries.empty:
                inj_delta = compute_injury_delta(
                    home_team, away_team, current_injuries, previous_injuries
                )
                injuries_home = inj_delta.get("home_out_delta", 0)
                injuries_away = inj_delta.get("away_out_delta", 0)
            model_prob, bayesian_log = update_probability(
                model_prob, snapshot, inj_delta, BAYESIAN_CONFIG,
            )
            price_cache[market["ticker"]] = market_price

        # Edge
        edge_info = find_edge(model_prob, market_price)

        # KL divergence
        kl_score = 0.0
        if ENABLE_KL_DIVERGENCE:
            from src.Utils.KLDivergence import symmetric_kl
            kl_score = symmetric_kl(model_prob, market_price / 100.0)

        # Stoikov
        stoikov_price = None
        stoikov_improvement = 0
        if ENABLE_STOIKOV_EXECUTION and edge_info["side"]:
            from src.Utils.StoikovExecution import optimal_limit_price as stoikov_calc

            close_time = parse_close_time(market.get("close_time"))
            hours_left = max(0, (close_time - datetime.now()).total_seconds() / 3600)
            orderbook = client.get_orderbook(market["ticker"], depth=5)
            exec_result = stoikov_calc(
                model_prob=model_prob,
                current_position=0,
                time_to_close_hours=hours_left,
                orderbook=orderbook,
                side=edge_info["side"],
                gamma=STOIKOV_GAMMA,
                max_improvement=STOIKOV_MAX_IMPROVEMENT,
            )
            stoikov_price = exec_result["optimal_price"]
            stoikov_improvement = exec_result["improvement_cents"]

        predictions.append({
            "ticker": market["ticker"],
            "home_team": home_team,
            "away_team": away_team,
            "model_prob": round(model_prob, 4),
            "kalshi_price": market_price,
            "edge": round(model_prob - market_price / 100.0, 4),
            "side": edge_info["side"],
            "kelly_size_cents": edge_info["recommended_size_cents"],
            "kl_divergence": round(kl_score, 4),
            "bayesian_log": bayesian_log,
            "injuries_home": injuries_home,
            "injuries_away": injuries_away,
            "stoikov_price": stoikov_price,
            "stoikov_improvement": stoikov_improvement,
            "close_time": market.get("close_time"),
            "timestamp": datetime.now().isoformat(),
        })

    # Save full scan
    LOG_DIR.mkdir(exist_ok=True)
    scan_path = LOG_DIR / f"scan_{today.strftime('%Y-%m-%d')}.json"
    with open(scan_path, "w") as f:
        json.dump(predictions, f, indent=2)

    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NBA Kalshi Live Trader")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log decisions without placing real orders",
    )
    args = parser.parse_args()
    run(dry_run=args.dry_run)
