"""Kalshi NBA Trading Dashboard.

Web UI for reviewing model predictions before manually placing trades.
Reads from the latest scan log or triggers a fresh model run.
"""

import json
import os
import sys
import threading
from datetime import date, datetime
from pathlib import Path

from flask import Flask, render_template, jsonify, request

BASE_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = BASE_DIR / "logs"
sys.path.insert(0, str(BASE_DIR))

app = Flask(__name__)


def get_latest_scan():
    """Read the most recent scan JSON from logs/."""
    today_str = date.today().strftime("%Y-%m-%d")
    scan_path = LOG_DIR / f"scan_{today_str}.json"
    if scan_path.exists():
        with open(scan_path) as f:
            return json.load(f), today_str
    # Check yesterday if today's doesn't exist yet
    from datetime import timedelta
    yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    scan_path = LOG_DIR / f"scan_{yesterday}.json"
    if scan_path.exists():
        with open(scan_path) as f:
            return json.load(f), yesterday
    return [], None


@app.route("/")
def index():
    predictions, scan_date = get_latest_scan()

    # Sort: trades first (by edge), then passes
    trades = [p for p in predictions if p.get("side")]
    passes = [p for p in predictions if not p.get("side")]
    trades.sort(key=lambda p: abs(p.get("edge", 0)), reverse=True)
    sorted_predictions = trades + passes

    return render_template(
        "index.html",
        predictions=sorted_predictions,
        scan_date=scan_date,
        today=date.today(),
        min_edge=float(os.getenv("MIN_EDGE", "0.15")),
    )


@app.route("/api/predictions")
def api_predictions():
    predictions, scan_date = get_latest_scan()
    return jsonify({"predictions": predictions, "scan_date": scan_date})


@app.route("/api/run-model", methods=["POST"])
def api_run_model():
    """Trigger a fresh model scan in a background thread.

    Tries live Kalshi API first. If that returns 0 markets (auth issues),
    falls back to generating predictions from the odds database.
    """
    def _run():
        try:
            from src.Bot.live_trader import scan_games
            results = scan_games()
            if results:
                return

            # Fallback: generate predictions from odds DB
            print("[Dashboard] Live Kalshi returned 0 markets, using odds DB fallback...")
            _run_from_db()
        except Exception as e:
            print(f"[Dashboard] Model run failed: {e}")
            import traceback
            traceback.print_exc()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return jsonify({"status": "started", "message": "Model is running... refresh in ~60 seconds."})


def _run_from_db():
    """Generate predictions using odds from the database (offline mode)."""
    import sqlite3
    import numpy as np
    import pandas as pd
    from datetime import datetime, date as date_mod

    sys.path.insert(0, str(BASE_DIR))
    from backtest_kalshi import (load_dataset, extract_feature_cols, train_ensemble,
                                  predict_ensemble, odds_to_kalshi_price)
    from src.Utils.KLDivergence import symmetric_kl
    from src.Utils.BayesianUpdater import update_probability
    from src.Features.injury_features import compute_live_injury_counts
    from src.DataProviders.PlayerDataProvider import load_all_injury_data

    df = load_dataset("dataset_enhanced")
    feature_cols = extract_feature_cols(df)

    # Use the latest date in the dataset (today's games may have yesterday's date due to offset)
    df["_date_str"] = df["Date"].dt.strftime("%Y-%m-%d")
    latest = df["_date_str"].max()
    test_df = df[df["_date_str"] == latest]
    today_str = latest
    print(f"[Dashboard] Using dataset date: {today_str} ({len(test_df)} games)")

    if test_df.empty:
        return

    # Train on everything before test date
    train_df = df[df["Date"] < test_df["Date"].min()]
    X_train = np.nan_to_num(train_df[feature_cols].astype(float).values, nan=0.0)
    y_train = train_df["Home-Team-Win"].astype(int).values
    X_test = np.nan_to_num(test_df[feature_cols].astype(float).values, nan=0.0)

    print(f"[Dashboard] Training ensemble on {len(X_train)} games...")
    fitted, meta, sig_cal, scaler, names = train_ensemble(X_train, y_train)
    probs = predict_ensemble(fitted, meta, sig_cal, scaler, X_test)

    # Load odds — try today's date and tomorrow (odds DB is 1 day ahead of dataset)
    odds_db = BASE_DIR / "Data" / "OddsData.sqlite"
    odds_df = pd.DataFrame()
    tomorrow_str = (pd.to_datetime(today_str) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    for odds_date in [today_str, tomorrow_str]:
        for tbl in ["odds_2025-26", "2025-26"]:
            try:
                with sqlite3.connect(odds_db) as con:
                    candidate = pd.read_sql_query(
                        f'SELECT * FROM "{tbl}" WHERE Date = ?', con, params=[odds_date])
                if not candidate.empty:
                    odds_df = candidate
                    print(f"[Dashboard] Found {len(odds_df)} odds rows in {tbl} for {odds_date}")
                    break
            except Exception:
                continue
        if not odds_df.empty:
            break

    # Load injuries
    injury_db_path = BASE_DIR / "Data" / "InjuryData.sqlite"
    injury_db = load_all_injury_data(injury_db_path) if injury_db_path.exists() else pd.DataFrame()
    if not injury_db.empty:
        injury_db["Date"] = pd.to_datetime(injury_db["Date"], errors="coerce")

    predictions = []
    for i in range(len(test_df)):
        home = test_df["TEAM_NAME"].iloc[i]
        away = test_df["TEAM_NAME.1"].iloc[i]
        prob = float(probs[i])

        # Find odds
        odds_row = odds_df[(odds_df["Home"] == home) & (odds_df["Away"] == away)]
        if odds_row.empty:
            continue
        contract_price = odds_to_kalshi_price(
            odds_row.iloc[0]["ML_Home"], odds_row.iloc[0]["ML_Away"])
        if contract_price is None or contract_price <= 0 or contract_price >= 100:
            continue

        # Bayesian injury update
        injuries_home = 0
        injuries_away = 0
        bayesian_log = []
        if not injury_db.empty:
            try:
                date_ts = pd.to_datetime(today_str, utc=True)
                day_inj = injury_db[injury_db["Date"] <= date_ts]
                if not day_inj.empty:
                    day_inj = day_inj.sort_values("Date").groupby("Player").last().reset_index()
                    counts = compute_live_injury_counts(home, away, day_inj)
                    injuries_home = counts["Injuries_Out_Home"]
                    injuries_away = counts["Injuries_Out_Away"]
                    if injuries_home > 0 or injuries_away > 0:
                        prob, bayesian_log = update_probability(
                            prob, None,
                            {"home_out_delta": injuries_home, "away_out_delta": injuries_away},
                            {"use_line_movement": False, "use_injuries": True,
                             "use_reverse_line": False, "injury_strength": 1.0,
                             "max_total_shift": 0.15},
                            backtest_mode=True)
            except Exception:
                pass

        # KL
        kl = symmetric_kl(prob, contract_price / 100.0)

        # Edge
        edge = prob - contract_price / 100.0
        min_edge = float(os.getenv("MIN_EDGE", "0.15"))
        if edge >= min_edge:
            side = "yes"
        elif -edge >= min_edge:
            side = "no"
        else:
            side = None

        predictions.append({
            "ticker": f"KXNBA-{home[:3].upper()}-{away[:3].upper()}",
            "home_team": home,
            "away_team": away,
            "model_prob": round(prob, 4),
            "kalshi_price": contract_price,
            "edge": round(edge, 4),
            "side": side,
            "kelly_size_cents": 0,
            "kl_divergence": round(kl, 4),
            "bayesian_log": bayesian_log,
            "injuries_home": injuries_home,
            "injuries_away": injuries_away,
            "stoikov_price": None,
            "stoikov_improvement": 0,
            "close_time": None,
            "timestamp": datetime.now().isoformat(),
        })

    # Save — use actual today's date for the filename so dashboard finds it
    import json as json_mod
    LOG_DIR.mkdir(exist_ok=True)
    actual_today = date_mod.today().strftime("%Y-%m-%d")
    scan_path = LOG_DIR / f"scan_{actual_today}.json"
    with open(scan_path, "w") as f:
        json_mod.dump(predictions, f, indent=2)
    print(f"[Dashboard] Saved {len(predictions)} predictions to {scan_path}")


if __name__ == "__main__":
    app.run(debug=True, port=5050)
