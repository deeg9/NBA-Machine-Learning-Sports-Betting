"""
Stacked Ensemble Model for NBA Game Prediction (Moneyline)

Based on a 2025 Scientific Reports paper achieving 83.96% accuracy using
a stacked ensemble approach. Base learners generate out-of-fold predictions
via TimeSeriesSplit, which are then fed to a Logistic Regression meta-learner.

Base learners (Level 0):
    - XGBoost
    - MLP Neural Network (scikit-learn)
    - K-Nearest Neighbors
    - AdaBoost
    - Logistic Regression
    - Random Forest

Meta-learner (Level 1):
    - Logistic Regression on stacked OOF probabilities

Usage:
    python -m src.Train-Models.Stacked_Ensemble_ML
    python -m src.Train-Models.Stacked_Ensemble_ML --dataset dataset_enhanced --seed 42
"""

import argparse
import sqlite3
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import _SigmoidCalibration
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_DB = BASE_DIR / "Data" / "dataset.sqlite"
MODEL_DIR = BASE_DIR / "Models" / "Ensemble_Models"

DEFAULT_DATASET = "dataset_enhanced"
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

# Models that require feature scaling
SCALED_MODELS = {"mlp", "knn"}


def load_dataset(dataset_name: str) -> pd.DataFrame:
    with sqlite3.connect(DATASET_DB) as con:
        return pd.read_sql_query(f'SELECT * FROM "{dataset_name}"', con)


def prepare_data(df: pd.DataFrame):
    """Sort by date, extract target, drop non-feature columns and NaN columns."""
    data = df.copy()
    if DATE_COLUMN in data.columns:
        data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], errors="coerce")
        data = data.sort_values(DATE_COLUMN)

    y = data[TARGET_COLUMN].astype(int).to_numpy()
    X = data.drop(columns=DROP_COLUMNS, errors="ignore")

    # Fill NaN with 0 (neutral value for rolling/player features early in season)
    nan_cols = X.columns[X.isna().any()].tolist()
    if nan_cols:
        print(f"Filling NaN in {len(nan_cols)} columns with 0")
        X[nan_cols] = X[nan_cols].fillna(0)

    feature_names = X.columns.tolist()
    X = X.astype(float).to_numpy()
    return X, y, feature_names


def split_train_test(X, y, test_size=0.1):
    """Time-ordered 90/10 split."""
    n = len(X)
    if n == 0:
        raise ValueError("Empty dataset.")
    test_start = int(n * (1 - test_size))
    return X[:test_start], y[:test_start], X[test_start:], y[test_start:]


def build_base_learners(seed: int) -> dict:
    """Construct base learner instances with tuned hyperparameters."""
    return {
        "xgboost": XGBClassifier(
            max_depth=5,
            learning_rate=0.05,
            n_estimators=500,
            colsample_bytree=0.8,
            subsample=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=seed,
            tree_method="hist",
            verbosity=0,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            alpha=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=seed,
            verbose=False,
        ),
        "knn": KNeighborsClassifier(
            n_neighbors=10,
            weights="distance",
            n_jobs=-1,
        ),
        "adaboost": AdaBoostClassifier(
            n_estimators=200,
            random_state=seed,
            algorithm="SAMME",
        ),
        "logistic_regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=seed,
            solver="lbfgs",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=12,
            random_state=seed,
            n_jobs=-1,
        ),
    }


def generate_oof_predictions(
    base_learners: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    scaler: StandardScaler,
    n_splits: int = 5,
) -> tuple[np.ndarray, dict]:
    """
    Generate out-of-fold probability predictions for each base learner
    using TimeSeriesSplit. Returns a matrix of OOF probabilities (n_samples x n_models)
    and a dict of fitted models from the last fold (used as a fallback reference).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    n_models = len(base_learners)
    oof_preds = np.full((len(X_train), n_models), np.nan)

    # Track per-model OOF accuracy
    model_names = list(base_learners.keys())

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        fold_X_train, fold_y_train = X_train[train_idx], y_train[train_idx]
        fold_X_val = X_train[val_idx]

        # Fit scaler on this fold's training data
        fold_scaler = StandardScaler()
        fold_X_train_scaled = fold_scaler.fit_transform(fold_X_train)
        fold_X_val_scaled = fold_scaler.transform(fold_X_val)

        print(f"  Fold {fold_idx + 1}/{n_splits} "
              f"(train: {len(train_idx)}, val: {len(val_idx)})")

        for model_idx, (name, model_template) in enumerate(base_learners.items()):
            # Clone the model for each fold
            from sklearn.base import clone
            model = clone(model_template)

            # Select scaled or raw features
            if name in SCALED_MODELS:
                tr_X, va_X = fold_X_train_scaled, fold_X_val_scaled
            else:
                tr_X, va_X = fold_X_train, fold_X_val

            model.fit(tr_X, fold_y_train)
            probs = model.predict_proba(va_X)
            # Store P(home_win) — probability of class 1
            oof_preds[val_idx, model_idx] = probs[:, 1]

    return oof_preds, model_names


def train_final_base_learners(
    base_learners: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    scaler: StandardScaler,
) -> dict:
    """Train each base learner on the full training set for final predictions."""
    from sklearn.base import clone

    fitted = {}
    for name, model_template in base_learners.items():
        model = clone(model_template)
        if name in SCALED_MODELS:
            X_fit = scaler.transform(X_train)
        else:
            X_fit = X_train
        model.fit(X_fit, y_train)
        fitted[name] = model
    return fitted


def predict_base_learners(
    fitted_models: dict,
    X: np.ndarray,
    scaler: StandardScaler,
) -> np.ndarray:
    """Generate probability predictions from all fitted base learners."""
    preds = []
    for name, model in fitted_models.items():
        if name in SCALED_MODELS:
            X_input = scaler.transform(X)
        else:
            X_input = X
        probs = model.predict_proba(X_input)
        preds.append(probs[:, 1])
    return np.column_stack(preds)


def main():
    parser = argparse.ArgumentParser(
        description="Train stacked ensemble for NBA game prediction."
    )
    parser.add_argument(
        "--dataset", default=DEFAULT_DATASET, help="Dataset table name."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed."
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load and prepare data ──────────────────────────────────────────
    print("Loading dataset:", args.dataset)
    df = load_dataset(args.dataset)
    if df.empty:
        print(f"No rows found for dataset '{args.dataset}'.")
        return

    X, y, feature_names = prepare_data(df)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    X_train, y_train, X_test, y_test = split_train_test(X, y)
    print(f"Train: {len(y_train)} | Test: {len(y_test)}")

    # ── Fit scaler on full training set ────────────────────────────────
    scaler = StandardScaler()
    scaler.fit(X_train)

    # ── Build base learners ────────────────────────────────────────────
    base_learners = build_base_learners(args.seed)
    model_names = list(base_learners.keys())
    print(f"\nBase learners: {', '.join(model_names)}")

    # ── Generate OOF predictions ───────────────────────────────────────
    print("\n── Generating out-of-fold predictions (5-fold TimeSeriesSplit) ──")
    t0 = time.time()
    oof_preds, _ = generate_oof_predictions(
        base_learners, X_train, y_train, scaler, n_splits=5
    )
    oof_time = time.time() - t0
    print(f"OOF generation completed in {oof_time:.1f}s")

    # Remove rows where OOF predictions are NaN (first fold's training portion)
    valid_mask = ~np.isnan(oof_preds).any(axis=1)
    oof_valid = oof_preds[valid_mask]
    y_train_valid = y_train[valid_mask]
    print(f"Valid OOF samples for meta-learner training: {len(y_train_valid)}")

    # ── Train meta-learner ─────────────────────────────────────────────
    print("\n── Training meta-learner (Logistic Regression) ──")
    meta_learner = LogisticRegression(
        C=0.5, max_iter=1000, random_state=args.seed, solver="lbfgs"
    )
    meta_learner.fit(oof_valid, y_train_valid)

    # Print meta-learner weights
    print("Meta-learner coefficients:")
    for name, coef in zip(model_names, meta_learner.coef_[0]):
        print(f"  {name:>22s}: {coef:+.4f}")

    # ── Train final base learners on full training set ─────────────────
    print("\n── Training final base learners on full training set ──")
    t0 = time.time()
    fitted_models = train_final_base_learners(base_learners, X_train, y_train, scaler)
    final_time = time.time() - t0
    print(f"Final training completed in {final_time:.1f}s")

    # ── Evaluate individual base models on test set ────────────────────
    print("\n── Individual base model performance on test set ──")
    test_preds_matrix = predict_base_learners(fitted_models, X_test, scaler)

    individual_results = {}
    for i, name in enumerate(model_names):
        preds_i = (test_preds_matrix[:, i] >= 0.5).astype(int)
        acc_i = accuracy_score(y_test, preds_i)
        loss_i = log_loss(y_test, test_preds_matrix[:, i], labels=[0, 1])
        individual_results[name] = {"accuracy": acc_i, "log_loss": loss_i}
        print(f"  {name:>22s}: accuracy={acc_i:.4f}  log_loss={loss_i:.4f}")

    # ── Stacked ensemble prediction ────────────────────────────────────
    print("\n── Stacked ensemble performance ──")
    meta_probs = meta_learner.predict_proba(test_preds_matrix)
    meta_preds = meta_learner.predict(test_preds_matrix)
    ensemble_acc = accuracy_score(y_test, meta_preds)
    ensemble_loss = log_loss(y_test, meta_probs, labels=[0, 1])
    print(f"  Stacked ensemble:        accuracy={ensemble_acc:.4f}  "
          f"log_loss={ensemble_loss:.4f}")

    # ── Sigmoid calibration ────────────────────────────────────────────
    print("\n── Applying sigmoid calibration ──")
    # Calibrate on the OOF meta-learner predictions
    oof_meta_probs = meta_learner.predict_proba(oof_valid)[:, 1]
    sigmoid_cal = _SigmoidCalibration()
    sigmoid_cal.fit(oof_meta_probs, y_train_valid)

    # Apply calibration to test predictions
    raw_test_p1 = meta_probs[:, 1]
    cal_test_p1 = sigmoid_cal.predict(raw_test_p1)
    cal_test_probs = np.column_stack([1 - cal_test_p1, cal_test_p1])
    cal_preds = (cal_test_p1 >= 0.5).astype(int)
    cal_acc = accuracy_score(y_test, cal_preds)
    cal_loss = log_loss(y_test, cal_test_probs, labels=[0, 1])
    print(f"  Calibrated ensemble:     accuracy={cal_acc:.4f}  "
          f"log_loss={cal_loss:.4f}")

    # ── Comparison table ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"{'Model':<24s} {'Accuracy':>10s} {'Log Loss':>10s}")
    print("-" * 60)
    for name in model_names:
        r = individual_results[name]
        print(f"  {name:<22s} {r['accuracy']:>9.4f} {r['log_loss']:>10.4f}")
    print("-" * 60)
    print(f"  {'Stacked Ensemble':<22s} {ensemble_acc:>9.4f} {ensemble_loss:>10.4f}")
    print(f"  {'+ Calibrated':<22s} {cal_acc:>9.4f} {cal_loss:>10.4f}")
    print("=" * 60)

    # ── Save ensemble ──────────────────────────────────────────────────
    ensemble_payload = {
        "base_learners": fitted_models,
        "meta_learner": meta_learner,
        "scaler": scaler,
        "model_names": model_names,
        "scaled_models": SCALED_MODELS,
        "feature_names": feature_names,
        "accuracy": ensemble_acc,
        "calibrated_accuracy": cal_acc,
    }

    model_path = MODEL_DIR / "stacked_ensemble_ML.pkl"
    joblib.dump(ensemble_payload, model_path)
    print(f"\nSaved ensemble: {model_path}")

    cal_path = MODEL_DIR / "stacked_ensemble_ML_calibration.pkl"
    joblib.dump(sigmoid_cal, cal_path)
    print(f"Saved calibration: {cal_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
