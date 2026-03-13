"""
Model evaluation and analysis module for the NBA prediction XGBoost model.

Provides SHAP analysis, calibration analysis, and comprehensive model
evaluation with walk-forward validation.

Usage:
    python -m model.evaluate
    python -m model.evaluate --dataset dataset_2012-26
    python -m model.evaluate --enhanced
"""

import argparse
import re
import sqlite3
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DB = BASE_DIR / "Data" / "dataset.sqlite"
MODEL_DIR = BASE_DIR / "Models" / "XGBoost_Models"
PLOTS_DIR = BASE_DIR / "model" / "plots"

DEFAULT_DATASET = "dataset_2012-26"
ENHANCED_DATASET = "dataset_enhanced"
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


# ---------------------------------------------------------------------------
# Data loading helpers (mirrors src/Train-Models/XGBoost_Model_ML.py)
# ---------------------------------------------------------------------------

def load_dataset(dataset_name: str) -> pd.DataFrame:
    """Load a dataset table from the SQLite database."""
    with sqlite3.connect(str(DATASET_DB)) as con:
        return pd.read_sql_query(f'SELECT * FROM "{dataset_name}"', con)


def prepare_data(df: pd.DataFrame):
    """Sort by date, split into features (X) and target (y).

    Returns X as a DataFrame (to preserve feature names) and y as a numpy
    array.
    """
    data = df.copy()
    if DATE_COLUMN in data.columns:
        data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], errors="coerce")
        data = data.sort_values(DATE_COLUMN)
    y = data[TARGET_COLUMN].astype(int).to_numpy()
    X = data.drop(columns=DROP_COLUMNS, errors="ignore").astype(float)
    return X, y


def split_train_test(X, y, test_size=0.1):
    """Walk-forward split: first (1 - test_size) for training, rest for test."""
    n = len(X)
    if n == 0:
        raise ValueError("Empty dataset.")
    test_start = int(n * (1 - test_size))
    if isinstance(X, pd.DataFrame):
        return X.iloc[:test_start], y[:test_start], X.iloc[test_start:], y[test_start:]
    return X[:test_start], y[:test_start], X[test_start:], y[test_start:]


# ---------------------------------------------------------------------------
# Model loading helpers (mirrors src/Predict/XGBoost_Runner.py)
# ---------------------------------------------------------------------------

def _select_model_path(kind: str = "ML") -> Path:
    """Select the best model file by (mtime, accuracy) from MODEL_DIR."""
    candidates = list(MODEL_DIR.glob(f"*{kind}*.json"))
    if not candidates:
        raise FileNotFoundError(f"No XGBoost {kind} model found in {MODEL_DIR}")

    def score(path):
        match = ACCURACY_PATTERN.search(path.name)
        accuracy = float(match.group(1)) if match else 0.0
        return (path.stat().st_mtime, accuracy)

    return max(candidates, key=score)


def load_model(kind: str = "ML"):
    """Load the best XGBoost booster and its calibrator (if available).

    Returns (booster, calibrator_or_None).
    """
    model_path = _select_model_path(kind)
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    print(f"Loaded model: {model_path.name}")

    calibration_path = model_path.with_name(f"{model_path.stem}_calibration.pkl")
    calibrator = None
    if calibration_path.exists():
        try:
            calibrator = joblib.load(calibration_path)
            print(f"Loaded calibrator: {calibration_path.name}")
        except Exception as exc:
            print(f"Warning: could not load calibrator ({exc})")
    else:
        print("No calibration file found; using raw probabilities.")

    return booster, calibrator


class BoosterWrapper(BaseEstimator, ClassifierMixin):
    """Thin wrapper so sklearn's CalibratedClassifierCV can call predict_proba."""

    def __init__(self, booster=None, num_class=NUM_CLASSES):
        self.booster = booster
        self.num_class = num_class
        self.classes_ = np.arange(num_class)

    def fit(self, X, y):
        self.classes_ = np.arange(self.num_class)
        return self

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.booster.predict(xgb.DMatrix(X))


# ===================================================================
# 1. SHAP Analysis
# ===================================================================

def run_shap_analysis(dataset_name: str = DEFAULT_DATASET, sample_size: int = 2000):
    """Run SHAP TreeExplainer on the trained XGBoost model.

    Generates:
        - model/plots/shap_summary.png   (bar chart, top 20 features)
        - model/plots/shap_beeswarm.png  (beeswarm plot)

    Returns the SHAP values object.
    """
    try:
        import shap
    except ImportError:
        print("ERROR: 'shap' package is required for SHAP analysis.")
        print("Install it with:  pip install shap")
        return None

    print("\n" + "=" * 60)
    print("SHAP FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)

    booster, _ = load_model("ML")
    df = load_dataset(dataset_name)
    X, y = prepare_data(df)
    _, _, X_test, y_test = split_train_test(X, y)

    # Sample for speed if test set is large
    if len(X_test) > sample_size:
        idx = np.random.default_rng(42).choice(len(X_test), sample_size, replace=False)
        idx.sort()
        X_sample = X_test.iloc[idx] if isinstance(X_test, pd.DataFrame) else X_test[idx]
    else:
        X_sample = X_test

    feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else None

    print(f"Running SHAP TreeExplainer on {len(X_sample)} samples ...")
    explainer = shap.TreeExplainer(booster)

    if isinstance(X_sample, pd.DataFrame):
        shap_values = explainer.shap_values(X_sample)
    else:
        shap_values = explainer.shap_values(xgb.DMatrix(X_sample))

    # For binary classification with multi:softprob, shap_values may be:
    #   - a list of arrays [class0_shap, class1_shap] (older SHAP)
    #   - a 3D array (n_samples, n_features, n_classes) (newer SHAP)
    #   - a 2D array (n_samples, n_features) (binary shortcut)
    if isinstance(shap_values, list) and len(shap_values) == NUM_CLASSES:
        sv = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    X_sample_np = X_sample.values if isinstance(X_sample, pd.DataFrame) else X_sample

    # --- Top 10 features by mean |SHAP| ---
    mean_abs = np.mean(np.abs(sv), axis=0).flatten()
    if feature_names:
        feat_importance = sorted(
            zip(feature_names, mean_abs), key=lambda t: t[1], reverse=True
        )
    else:
        feat_importance = sorted(
            zip(range(sv.shape[1]), mean_abs), key=lambda t: t[1], reverse=True
        )

    print("\nTop 10 features by mean |SHAP value|:")
    print("-" * 45)
    for rank, (feat, val) in enumerate(feat_importance[:10], 1):
        print(f"  {rank:>2}. {str(feat):<35s} {val:.4f}")

    # --- Bar chart (top 20) ---
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    top_n = 20
    top_feats = feat_importance[:top_n]
    names = [str(f) for f, _ in top_feats]
    vals = [v for _, v in top_feats]
    y_pos = np.arange(len(names))
    ax.barh(y_pos, vals, color="#1f77b4")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("SHAP Feature Importance (Top 20)")
    fig.tight_layout()
    summary_path = PLOTS_DIR / "shap_summary.png"
    fig.savefig(summary_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved SHAP summary bar chart -> {summary_path}")

    # --- Beeswarm plot ---
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        sv,
        X_sample_np,
        feature_names=feature_names,
        show=False,
        max_display=20,
    )
    beeswarm_path = PLOTS_DIR / "shap_beeswarm.png"
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=150)
    plt.close("all")
    print(f"Saved SHAP beeswarm plot     -> {beeswarm_path}")

    return shap_values


# ===================================================================
# 2. Calibration Analysis
# ===================================================================

def run_calibration_analysis(dataset_name: str = DEFAULT_DATASET):
    """Compare raw vs calibrated probabilities against actual outcomes.

    Generates:
        - model/plots/calibration_curve.png  (reliability diagram)

    Prints Brier score, ECE, and log loss for both raw and calibrated models.
    """
    print("\n" + "=" * 60)
    print("CALIBRATION ANALYSIS")
    print("=" * 60)

    booster, _ = load_model("ML")
    df = load_dataset(dataset_name)
    X, y = prepare_data(df)
    X_train_val, y_train_val, X_test, y_test = split_train_test(X, y)

    X_test_np = X_test.values if isinstance(X_test, pd.DataFrame) else X_test

    # Raw probabilities (class 1 = home-team-win)
    raw_probs_all = booster.predict(xgb.DMatrix(X_test_np))
    raw_probs = raw_probs_all[:, 1] if raw_probs_all.ndim == 2 else raw_probs_all

    # Fit a fresh sigmoid calibration directly (avoids sklearn version issues)
    X_tv_np = X_train_val.values if isinstance(X_train_val, pd.DataFrame) else X_train_val
    calib_start = int(len(X_tv_np) * 0.9)
    X_calib, y_calib = X_tv_np[calib_start:], y_train_val[calib_start:]
    try:
        from sklearn.calibration import _SigmoidCalibration
        calib_raw = booster.predict(xgb.DMatrix(X_calib))
        calib_raw_p1 = calib_raw[:, 1] if calib_raw.ndim == 2 else calib_raw
        sigmoid_cal = _SigmoidCalibration()
        sigmoid_cal.fit(calib_raw_p1, y_calib)
        cal_probs = sigmoid_cal.predict(raw_probs)
    except Exception as e:
        print(f"  Warning: calibration failed ({e}), using raw probs only.")
        cal_probs = None

    def _brier(y_true, probs):
        return brier_score_loss(y_true, probs)

    def _ece(y_true, probs, n_bins=10):
        """Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
            mask = (probs >= lo) & (probs < hi)
            if mask.sum() == 0:
                continue
            bin_acc = y_true[mask].mean()
            bin_conf = probs[mask].mean()
            ece += mask.sum() * abs(bin_acc - bin_conf)
        return ece / len(y_true)

    def _logloss(y_true, probs):
        p = np.column_stack([1 - probs, probs])
        return log_loss(y_true, p, labels=[0, 1])

    print("\nMetric                    Raw         Calibrated")
    print("-" * 52)
    raw_brier = _brier(y_test, raw_probs)
    raw_ece = _ece(y_test, raw_probs)
    raw_ll = _logloss(y_test, raw_probs)

    if cal_probs is not None:
        cal_brier = _brier(y_test, cal_probs)
        cal_ece = _ece(y_test, cal_probs)
        cal_ll = _logloss(y_test, cal_probs)
        print(f"  Brier Score (< 0.25)    {raw_brier:.4f}       {cal_brier:.4f}")
        print(f"  ECE                     {raw_ece:.4f}       {cal_ece:.4f}")
        print(f"  Log Loss                {raw_ll:.4f}       {cal_ll:.4f}")

        brier_delta = raw_brier - cal_brier
        print(f"\nCalibration improvement (Brier): {brier_delta:+.4f} "
              f"({'better' if brier_delta > 0 else 'worse'})")
    else:
        print(f"  Brier Score (< 0.25)    {raw_brier:.4f}       N/A")
        print(f"  ECE                     {raw_ece:.4f}       N/A")
        print(f"  Log Loss                {raw_ll:.4f}       N/A")
        print("\nNo calibrator available for comparison.")

    # --- Reliability diagram ---
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")

    raw_frac, raw_mean = calibration_curve(y_test, raw_probs, n_bins=10)
    ax.plot(raw_mean, raw_frac, "s-", label="Raw model")

    if cal_probs is not None:
        cal_frac, cal_mean = calibration_curve(y_test, cal_probs, n_bins=10)
        ax.plot(cal_mean, cal_frac, "o-", label="Calibrated (sigmoid)")

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives (actual)")
    ax.set_title("Calibration Curve (Reliability Diagram)")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    cal_path = PLOTS_DIR / "calibration_curve.png"
    fig.savefig(cal_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved calibration curve -> {cal_path}")


# ===================================================================
# 3. Comprehensive Evaluation
# ===================================================================

def run_full_evaluation(dataset_name: str = DEFAULT_DATASET):
    """Run a comprehensive evaluation with walk-forward split.

    Train on 90% of time-ordered data, test on last 10%.

    Generates:
        - model/plots/confusion_matrix.png
        - model/plots/roc_curve.png

    Prints accuracy, precision, recall, F1, ROC-AUC, confusion matrix,
    and full classification report.
    """
    print("\n" + "=" * 60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("=" * 60)

    booster, _ = load_model("ML")
    df = load_dataset(dataset_name)
    X, y = prepare_data(df)
    X_train, y_train, X_test, y_test = split_train_test(X, y)

    X_test_np = X_test.values if isinstance(X_test, pd.DataFrame) else X_test

    # Get raw predictions and apply fresh sigmoid calibration
    raw_probs = booster.predict(xgb.DMatrix(X_test_np))
    try:
        from sklearn.calibration import _SigmoidCalibration
        X_train_np = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        calib_start = int(len(X_train_np) * 0.9)
        X_calib, y_calib = X_train_np[calib_start:], y_train[calib_start:]
        calib_raw = booster.predict(xgb.DMatrix(X_calib))
        calib_raw_p1 = calib_raw[:, 1] if calib_raw.ndim == 2 else calib_raw
        sigmoid_cal = _SigmoidCalibration()
        sigmoid_cal.fit(calib_raw_p1, y_calib)
        raw_p1 = raw_probs[:, 1] if raw_probs.ndim == 2 else raw_probs
        cal_p1 = sigmoid_cal.predict(raw_p1)
        probs = np.column_stack([1 - cal_p1, cal_p1])
    except Exception:
        probs = raw_probs

    # Class 1 probabilities for ROC
    probs_class1 = probs[:, 1] if probs.ndim == 2 else probs
    y_pred = (probs_class1 >= 0.5).astype(int)

    # --- Metrics ---
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, probs_class1)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nDataset: {dataset_name}")
    print(f"Train size: {len(y_train):,}  |  Test size: {len(y_test):,}")
    print(f"\n{'Metric':<22s} {'Value':>10s}")
    print("-" * 34)
    print(f"  {'Accuracy':<20s} {acc:>9.4f}")
    print(f"  {'Precision':<20s} {prec:>9.4f}")
    print(f"  {'Recall':<20s} {rec:>9.4f}")
    print(f"  {'F1 Score':<20s} {f1:>9.4f}")
    print(f"  {'ROC-AUC':<20s} {roc_auc:>9.4f}")

    print(f"\nConfusion Matrix (rows=actual, cols=predicted):")
    print(f"  TN={cm[0][0]:>5d}   FP={cm[0][1]:>5d}")
    print(f"  FN={cm[1][0]:>5d}   TP={cm[1][1]:>5d}")

    print(f"\nClassification Report:")
    print(classification_report(
        y_test, y_pred, target_names=["Away Win (0)", "Home Win (1)"]
    ))

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Confusion matrix plot ---
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Away Win (0)", "Home Win (1)"])
    ax.set_yticklabels(["Away Win (0)", "Home Win (1)"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    # Annotate cells
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=16, fontweight="bold", color=color)

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    cm_path = PLOTS_DIR / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix plot -> {cm_path}")

    # --- ROC curve plot ---
    fpr, tpr, _ = roc_curve(y_test, probs_class1)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"XGBoost (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    roc_path = PLOTS_DIR / "roc_curve.png"
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)
    print(f"Saved ROC curve plot        -> {roc_path}")

    # Target check
    target_acc = 0.68
    if acc >= target_acc:
        print(f"\n[PASS] Accuracy {acc:.2%} meets target (>{target_acc:.0%}).")
    else:
        print(f"\n[WARN] Accuracy {acc:.2%} below target (>{target_acc:.0%}).")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
    }


# ===================================================================
# 4. Main / CLI entry point
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the NBA prediction XGBoost model."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Dataset table name (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--enhanced",
        action="store_true",
        help=f'Use the "{ENHANCED_DATASET}" table instead.',
    )
    args = parser.parse_args()

    dataset_name = ENHANCED_DATASET if args.enhanced else args.dataset

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  NBA XGBoost Model Evaluation Suite")
    print(f"  Dataset: {dataset_name}")
    print("=" * 60)

    # 1. SHAP analysis
    run_shap_analysis(dataset_name)

    # 2. Calibration analysis
    run_calibration_analysis(dataset_name)

    # 3. Comprehensive evaluation
    metrics = run_full_evaluation(dataset_name)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if metrics:
        print(f"  Accuracy:  {metrics['accuracy']:.2%}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"\n  Target: >68% accuracy with calibrated probabilities")
    print(f"  Plots saved to: {PLOTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
