#!/usr/bin/env python
"""
Walk-forward backtest of NBA game prediction models against historical games.

Supports both XGBoost (single model) and stacked ensemble (6 base learners +
meta-learner) with walk-forward retraining per season.

Usage:
    python backtest.py                    # backtest all seasons (XGBoost)
    python backtest.py --season 2024-25   # backtest a single season
    python backtest.py --min-season 2018-19  # backtest from 2018-19 onward
    python backtest.py --model-type ensemble --dataset dataset_enhanced_v2
"""

import argparse
import re
import sqlite3
import sys
from importlib import import_module
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV, _SigmoidCalibration
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler

# Import ensemble helpers from training module
_ensemble_mod = import_module("src.Train-Models.Stacked_Ensemble_ML")
build_base_learners = _ensemble_mod.build_base_learners
generate_oof_predictions = _ensemble_mod.generate_oof_predictions
train_final_base_learners = _ensemble_mod.train_final_base_learners
predict_base_learners = _ensemble_mod.predict_base_learners
SCALED_MODELS = _ensemble_mod.SCALED_MODELS

# ---------------------------------------------------------------------------
# Paths & constants (matching the existing codebase conventions)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_DB = BASE_DIR / "Data" / "dataset.sqlite"
ODDS_DB = BASE_DIR / "Data" / "OddsData.sqlite"
MODEL_DIR = BASE_DIR / "Models" / "XGBoost_Models"
CONFIG_PATH = BASE_DIR / "config.toml"

DEFAULT_DATASET = "dataset_2012-26"
TARGET_COLUMN = "Home-Team-Win"
DATE_COLUMN = "Date"
DROP_COLUMNS = [
    "index",
    "Score",
    "Home-Team-Win",
    "TEAM_NAME",
    "Date",
    "index.1",
    "TEAM_NAME.1",
    "Date.1",
    "OU-Cover",
    "OU",
]

NUM_CLASSES = 2
ACCURACY_PATTERN = re.compile(r"XGBoost_(\d+(?:\.\d+)?)%_")
BET_SIZE = 100  # flat bet in dollars


# ---------------------------------------------------------------------------
# Helpers reused from the training / prediction code
# ---------------------------------------------------------------------------
class BoosterWrapper:
    """Thin wrapper so sklearn CalibratedClassifierCV can call predict_proba."""

    _estimator_type = "classifier"

    def __init__(self, booster, num_class=NUM_CLASSES):
        self.booster = booster
        self.classes_ = np.arange(num_class)

    def fit(self, X, y):
        return self

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        return self.booster.predict(xgb.DMatrix(X))


def compute_sample_weights(y, num_classes=NUM_CLASSES):
    counts = np.bincount(y, minlength=num_classes)
    total = len(y)
    class_weights = {
        cls: (total / (num_classes * count)) if count else 1.0
        for cls, count in enumerate(counts)
    }
    return np.array([class_weights[label] for label in y])


# ---------------------------------------------------------------------------
# Model / calibrator loading
# ---------------------------------------------------------------------------
def select_best_ml_model():
    """Pick the best ML model by (mtime, accuracy%) -- same logic as XGBoost_Runner."""
    candidates = list(MODEL_DIR.glob("*ML*.json"))
    if not candidates:
        return None

    def score(path):
        match = ACCURACY_PATTERN.search(path.name)
        accuracy = float(match.group(1)) if match else 0.0
        return (path.stat().st_mtime, accuracy)

    return max(candidates, key=score)


def load_pretrained_model_and_calibrator():
    """Load the best pre-trained model + its calibration pickle (if any)."""
    model_path = select_best_ml_model()
    if model_path is None:
        return None, None, None
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    cal_path = model_path.with_name(f"{model_path.stem}_calibration.pkl")
    calibrator = None
    if cal_path.exists():
        try:
            calibrator = joblib.load(cal_path)
        except Exception:
            pass
    return booster, calibrator, model_path


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def load_dataset(dataset_name=None):
    table = dataset_name or DEFAULT_DATASET
    with sqlite3.connect(DATASET_DB) as con:
        df = pd.read_sql_query(f'SELECT * FROM "{table}"', con)
    if DATE_COLUMN in df.columns:
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
        df = df.sort_values(DATE_COLUMN).reset_index(drop=True)
    return df


def extract_features(df):
    """Return (X_array, feature_column_names) dropping non-feature columns."""
    feature_cols = [
        c for c in df.columns
        if c not in DROP_COLUMNS and df[c].dtype in ("float64", "float32", "int64", "int32")
    ]
    return df[feature_cols].astype(float).values, feature_cols


def season_from_date(dt):
    """Map a game date to its NBA season label, e.g. 2024-10-22 -> '2024-25'."""
    if pd.isna(dt):
        return None
    year = dt.year
    month = dt.month
    if month >= 8:
        return f"{year}-{str(year + 1)[-2:]}"
    else:
        return f"{year - 1}-{str(year)[-2:]}"


# ---------------------------------------------------------------------------
# Odds data helpers
# ---------------------------------------------------------------------------
def _table_exists(con, table_name):
    cur = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    )
    return cur.fetchone() is not None


def _select_odds_table(con, season_key):
    for name in [
        f"odds_{season_key}_new",
        f"odds_{season_key}",
        f"{season_key}_new",
        season_key,
    ]:
        if _table_exists(con, name):
            return name
    return None


def load_odds_data():
    """Load all available odds rows keyed by (date_str, home_team, away_team)."""
    if not ODDS_DB.exists():
        return {}

    odds_map = {}
    with sqlite3.connect(ODDS_DB) as con:
        tables = [
            row[0]
            for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        for table in tables:
            try:
                df = pd.read_sql_query(f'SELECT * FROM "{table}"', con)
            except Exception:
                continue
            for row in df.itertuples(index=False):
                if not hasattr(row, "Date") or not hasattr(row, "Home") or not hasattr(row, "Away"):
                    continue
                date_val = pd.to_datetime(row.Date, errors="coerce")
                if pd.isna(date_val):
                    continue
                date_str = date_val.strftime("%Y-%m-%d")
                key = (date_str, row.Home, row.Away)
                ml_home = getattr(row, "ML_Home", None)
                ml_away = getattr(row, "ML_Away", None)
                odds_map[key] = {"ML_Home": ml_home, "ML_Away": ml_away}
    return odds_map


# ---------------------------------------------------------------------------
# EV / Kelly helpers (from src/Utils)
# ---------------------------------------------------------------------------
def american_to_decimal(american_odds):
    if american_odds >= 100:
        return american_odds / 100
    return 100 / abs(american_odds)


def payout_on_win(american_odds):
    """Net payout on a $100 bet."""
    if american_odds > 0:
        return american_odds
    return (100 / abs(american_odds)) * 100


def expected_value(p_win, american_odds):
    p_loss = 1 - p_win
    return round(p_win * payout_on_win(american_odds) - p_loss * BET_SIZE, 2)


def kelly_fraction(american_odds, model_prob):
    dec = american_to_decimal(american_odds)
    frac = round((100 * (dec * model_prob - (1 - model_prob))) / dec, 2)
    return max(frac, 0.0)


# ---------------------------------------------------------------------------
# Betting simulation
# ---------------------------------------------------------------------------
def simulate_bet(predicted_home_win, actual_home_win, ml_home, ml_away):
    """
    Simulate a flat $100 bet on the predicted winner.
    Returns (profit, odds_used).
    """
    if predicted_home_win:
        odds = ml_home
    else:
        odds = ml_away

    if odds is None:
        return None, None

    try:
        odds = float(odds)
    except (TypeError, ValueError):
        return None, None

    correct = predicted_home_win == actual_home_win
    if correct:
        profit = payout_on_win(odds)
    else:
        profit = -BET_SIZE

    return profit, odds


# ---------------------------------------------------------------------------
# Walk-forward training helper
# ---------------------------------------------------------------------------
def train_xgb_on_subset(X_train, y_train):
    """
    Train an XGBoost model on the given data.
    Uses reasonable default hyperparameters (not random-searched) to keep the
    backtest deterministic and fast.
    """
    weights = compute_sample_weights(y_train)

    # Reserve last 10% for calibration
    n = len(X_train)
    cal_start = int(n * 0.9)
    X_fit, y_fit = X_train[:cal_start], y_train[:cal_start]
    X_cal, y_cal = X_train[cal_start:], y_train[cal_start:]
    w_fit = weights[:cal_start]

    params = {
        "max_depth": 5,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma": 1.0,
        "objective": "multi:softprob",
        "num_class": NUM_CLASSES,
        "eval_metric": ["mlogloss"],
        "seed": 42,
        "tree_method": "hist",
    }

    dtrain = xgb.DMatrix(X_fit, label=y_fit, weight=w_fit)
    dval = xgb.DMatrix(X_cal, label=y_cal)

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=800,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    # Calibrate
    calibrator = CalibratedClassifierCV(
        BoosterWrapper(booster),
        method="sigmoid",
        cv="prefit",
    )
    calibrator.fit(X_cal, y_cal)

    return booster, calibrator


# ---------------------------------------------------------------------------
# Ensemble walk-forward training
# ---------------------------------------------------------------------------
def train_ensemble_on_subset(X_train, y_train, seed=42):
    """
    Train a stacked ensemble on the given data, mirroring the full
    Stacked_Ensemble_ML.py pipeline: OOF predictions → meta-learner → calibration.
    """
    base_learners = build_base_learners(seed)

    scaler = StandardScaler()
    scaler.fit(X_train)

    # OOF predictions with 5-fold TimeSeriesSplit
    oof_preds, model_names = generate_oof_predictions(
        base_learners, X_train, y_train, scaler, n_splits=5
    )

    # Filter NaN rows (first fold's training portion)
    valid_mask = ~np.isnan(oof_preds).any(axis=1)
    oof_valid = oof_preds[valid_mask]
    y_train_valid = y_train[valid_mask]

    # Meta-learner
    meta_learner = LogisticRegression(
        C=0.5, max_iter=1000, random_state=seed, solver="lbfgs"
    )
    meta_learner.fit(oof_valid, y_train_valid)

    # Train final base learners on full training set
    fitted_models = train_final_base_learners(
        base_learners, X_train, y_train, scaler
    )

    # Sigmoid calibration on OOF meta-predictions
    oof_meta_probs = meta_learner.predict_proba(oof_valid)[:, 1]
    sigmoid_cal = _SigmoidCalibration()
    sigmoid_cal.fit(oof_meta_probs, y_train_valid)

    return fitted_models, meta_learner, sigmoid_cal, scaler, model_names


def predict_ensemble(fitted_models, meta_learner, sigmoid_cal, scaler, X):
    """Generate calibrated ensemble probabilities. Returns (n_samples, 2)."""
    preds_matrix = predict_base_learners(fitted_models, X, scaler)
    meta_probs = meta_learner.predict_proba(preds_matrix)

    if sigmoid_cal is not None:
        raw_p1 = meta_probs[:, 1]
        cal_p1 = sigmoid_cal.predict(raw_p1)
        return np.column_stack([1 - cal_p1, cal_p1])

    return meta_probs


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------
def predict_probs(booster, calibrator, X):
    """Get predicted probabilities, falling back to raw if calibrator fails."""
    if calibrator is not None:
        try:
            return calibrator.predict_proba(X)
        except (ValueError, AttributeError):
            pass
    # Fall back to raw booster predictions
    return booster.predict(xgb.DMatrix(X))


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------
def calibration_table(y_true, probs, n_bins=10):
    """Return a DataFrame with predicted-probability bins vs actual win rate."""
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        count = mask.sum()
        if count == 0:
            continue
        actual = y_true[mask].mean()
        predicted = probs[mask].mean()
        rows.append({
            "bin": f"{lo:.1f}-{hi:.1f}",
            "count": int(count),
            "pred_prob": round(predicted, 4),
            "actual_rate": round(actual, 4),
            "gap": round(abs(predicted - actual), 4),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main backtest
# ---------------------------------------------------------------------------
def run_backtest(target_season=None, min_season=None, use_pretrained=False, dataset_name=None, model_type="xgboost"):
    print(f"Loading dataset... (model: {model_type})")
    df = load_dataset(dataset_name)
    if df.empty:
        print("Dataset is empty. Nothing to backtest.")
        return

    # Assign season labels
    df["_season"] = df[DATE_COLUMN].apply(season_from_date)
    seasons = sorted(df["_season"].dropna().unique())

    if target_season:
        if target_season not in seasons:
            print(f"Season {target_season} not found in dataset. Available: {seasons}")
            return
        seasons = [target_season]
    elif min_season:
        seasons = [s for s in seasons if s >= min_season]

    # Need at least one prior season for training
    all_seasons = sorted(df["_season"].dropna().unique())
    first_test_idx = all_seasons.index(seasons[0]) if seasons[0] in all_seasons else 0
    if first_test_idx == 0 and not use_pretrained:
        print(
            f"Warning: no prior season available before {seasons[0]} for training. "
            "Skipping that season or use --use-pretrained."
        )
        seasons = seasons[1:] if len(seasons) > 1 else seasons

    # Load odds data for betting simulation
    print("Loading odds data...")
    odds_map = load_odds_data()
    odds_available = len(odds_map) > 0
    if not odds_available:
        print("  No odds data found -- betting simulation will be skipped.")

    # Pre-trained model (optional fallback)
    pretrained_booster, pretrained_cal, pretrained_path = None, None, None
    if use_pretrained:
        pretrained_booster, pretrained_cal, pretrained_path = (
            load_pretrained_model_and_calibrator()
        )
        if pretrained_booster is None:
            print("No pre-trained model found. Will train walk-forward instead.")
            use_pretrained = False
        else:
            print(f"Using pre-trained model: {pretrained_path.name}")

    # Feature columns (consistent across all splits)
    _, feature_cols = extract_features(df)

    # Season-level results collection
    season_results = []
    all_y_true = []
    all_y_pred = []
    all_probs_home = []
    all_profits = []

    print()
    header = f"{'Season':<12} {'Games':>6} {'Accuracy':>9} {'LogLoss':>8}"
    if odds_available:
        header += f" {'Bets':>6} {'P&L ($)':>10} {'ROI':>8}"
    print(header)
    print("-" * len(header))

    for season in seasons:
        season_mask = df["_season"] == season
        df_test = df.loc[season_mask].copy()

        if df_test.empty:
            continue

        X_test = df_test[feature_cols].astype(float).values
        y_test = df_test[TARGET_COLUMN].astype(int).values

        # Walk-forward: train on everything before this season
        prior_mask = df[DATE_COLUMN] < df_test[DATE_COLUMN].min()
        df_train = df.loc[prior_mask]
        if len(df_train) < 200:
            continue
        X_train = df_train[feature_cols].astype(float).values
        y_train = df_train[TARGET_COLUMN].astype(int).values

        # Fill NaN with 0 (player/rolling features early in dataset)
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)

        if use_pretrained:
            probs = predict_probs(pretrained_booster, pretrained_cal, X_test)
        elif model_type == "ensemble":
            fitted, meta, sig_cal, scaler, _ = train_ensemble_on_subset(X_train, y_train)
            probs = predict_ensemble(fitted, meta, sig_cal, scaler, X_test)
        else:
            booster, calibrator = train_xgb_on_subset(X_train, y_train)
            probs = predict_probs(booster, calibrator, X_test)
        y_pred = np.argmax(probs, axis=1)
        prob_home_win = probs[:, 1]

        acc = accuracy_score(y_test, y_pred)
        ll = log_loss(y_test, probs, labels=[0, 1])

        # Betting simulation
        season_profits = []
        n_bets = 0
        if odds_available:
            dates = df_test[DATE_COLUMN]
            home_names = df_test["TEAM_NAME"] if "TEAM_NAME" in df_test.columns else [None] * len(df_test)
            away_names = df_test["TEAM_NAME.1"] if "TEAM_NAME.1" in df_test.columns else [None] * len(df_test)

            for i in range(len(df_test)):
                dt = dates.iloc[i]
                if pd.isna(dt):
                    continue
                date_str = dt.strftime("%Y-%m-%d")
                home = home_names.iloc[i] if hasattr(home_names, "iloc") else None
                away = away_names.iloc[i] if hasattr(away_names, "iloc") else None

                if home is None or away is None:
                    continue

                key = (date_str, home, away)
                odds_info = odds_map.get(key)
                if odds_info is None:
                    continue

                predicted_home = bool(y_pred[i] == 1)
                actual_home = bool(y_test[i] == 1)
                profit, odds_used = simulate_bet(
                    predicted_home, actual_home,
                    odds_info["ML_Home"], odds_info["ML_Away"],
                )
                if profit is not None:
                    season_profits.append(profit)
                    all_profits.append(profit)
                    n_bets += 1

        # Collect
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_probs_home.extend(prob_home_win)

        row = f"{season:<12} {len(y_test):>6} {acc * 100:>8.1f}% {ll:>8.4f}"
        if odds_available:
            if n_bets > 0:
                total_pnl = sum(season_profits)
                roi = total_pnl / (n_bets * BET_SIZE) * 100
                row += f" {n_bets:>6} {total_pnl:>+10.2f} {roi:>+7.1f}%"
            else:
                row += f" {'--':>6} {'--':>10} {'--':>8}"
        print(row)

        season_results.append({
            "season": season,
            "games": len(y_test),
            "accuracy": round(acc * 100, 2),
            "log_loss": round(ll, 4),
            "bets": n_bets,
            "pnl": round(sum(season_profits), 2) if season_profits else 0.0,
        })

    # ------------------------------------------------------------------
    # Overall summary
    # ------------------------------------------------------------------
    if not all_y_true:
        print("\nNo games were backtested.")
        return

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_probs_home = np.array(all_probs_home)

    total_games = len(all_y_true)
    overall_acc = accuracy_score(all_y_true, all_y_pred)

    print("-" * len(header))
    summary = f"{'TOTAL':<12} {total_games:>6} {overall_acc * 100:>8.1f}%"
    # skip log-loss width for total
    summary += " " * 9
    if odds_available and all_profits:
        total_bets = len(all_profits)
        total_pnl = sum(all_profits)
        overall_roi = total_pnl / (total_bets * BET_SIZE) * 100
        summary += f" {total_bets:>6} {total_pnl:>+10.2f} {overall_roi:>+7.1f}%"
    print(summary)

    # Calibration
    print("\n--- Calibration (predicted home-win prob vs actual) ---")
    cal_df = calibration_table(all_y_true, all_probs_home)
    if cal_df.empty:
        print("  Not enough data for calibration bins.")
    else:
        print(cal_df.to_string(index=False))
        mean_gap = cal_df["gap"].mean()
        print(f"\n  Mean absolute calibration gap: {mean_gap:.4f}")

    # Home-win base rate
    home_rate = all_y_true.mean()
    pred_rate = all_y_pred.mean()
    print(f"\n  Home-win base rate (actual): {home_rate:.3f}")
    print(f"  Home-win prediction rate:    {pred_rate:.3f}")

    print("\nBacktest complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Walk-forward backtest of XGBoost ML model on historical NBA games."
    )
    parser.add_argument(
        "--season",
        default=None,
        help="Backtest a single season (e.g. 2024-25). Default: all seasons.",
    )
    parser.add_argument(
        "--min-season",
        default=None,
        help="Backtest seasons starting from this one (e.g. 2018-19).",
    )
    parser.add_argument(
        "--use-pretrained",
        action="store_true",
        help="Use the saved pre-trained model instead of walk-forward retraining.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset table name (default: dataset_2012-26). Use 'dataset_enhanced_v2' for enhanced features.",
    )
    parser.add_argument(
        "--model-type",
        choices=["xgboost", "ensemble"],
        default="xgboost",
        help="Model type: 'xgboost' (single model) or 'ensemble' (stacked). Default: xgboost.",
    )
    args = parser.parse_args()

    if args.season and args.min_season:
        print("Error: --season and --min-season are mutually exclusive.")
        sys.exit(1)

    # Default to enhanced dataset for ensemble
    dataset = args.dataset
    if dataset is None and args.model_type == "ensemble":
        dataset = "dataset_enhanced_v2"

    run_backtest(
        target_season=args.season,
        min_season=args.min_season,
        use_pretrained=args.use_pretrained,
        dataset_name=dataset,
        model_type=args.model_type,
    )


if __name__ == "__main__":
    main()
