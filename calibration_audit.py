#!/usr/bin/env python
"""
Calibration Audit: diagnose why a 67.3%-accurate ensemble finds almost no
bets with ≥5% edge.  Compares uncalibrated, Platt sigmoid, and isotonic
calibration on accuracy metrics AND simulated trading outcomes.

Usage:
    python3 calibration_audit.py
"""

import sys
from collections import defaultdict
from importlib import import_module

import numpy as np
import pandas as pd
from sklearn.calibration import _SigmoidCalibration
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler

# ── Reuse helpers from existing codebase ──────────────────────────────────
from backtest_kalshi import (
    DATE_COLUMN,
    TARGET_COLUMN,
    compute_trade,
    extract_feature_cols,
    load_dataset,
    load_odds_2025,
    odds_to_kalshi_price,
    season_from_date,
)
from backtest_kalshi_sweep import simulate_trading

_ensemble_mod = import_module("src.Train-Models.Stacked_Ensemble_ML")
build_base_learners = _ensemble_mod.build_base_learners
generate_oof_predictions = _ensemble_mod.generate_oof_predictions
train_final_base_learners = _ensemble_mod.train_final_base_learners
predict_base_learners = _ensemble_mod.predict_base_learners
SCALED_MODELS = _ensemble_mod.SCALED_MODELS

SEED = 42


# ── Helper: Expected Calibration Error ────────────────────────────────────

def compute_ece(y_true, probs, n_bins=10):
    """Weighted bin-level |accuracy − confidence|."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if not mask.any():
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = probs[mask].mean()
        ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)
    return ece


# ── Helper: text reliability diagram ─────────────────────────────────────

def print_reliability_diagram(y_true, probs, label):
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    print(f"\n  Reliability diagram — {label}")
    print(f"  {'Bin':>12s}  {'Predicted':>9s}  {'Actual':>8s}  {'Count':>6s}  Bar")
    print("  " + "-" * 62)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        cnt = mask.sum()
        if cnt == 0:
            pred_mean = 0
            act_mean = 0
        else:
            pred_mean = probs[mask].mean()
            act_mean = y_true[mask].mean()
        bar_pred = "#" * int(round(pred_mean * 30))
        bar_act = "=" * int(round(act_mean * 30))
        print(f"  [{lo:.1f},{hi:.1f})  "
              f"  {pred_mean:>7.3f}    {act_mean:>6.3f}  {cnt:>6d}  "
              f"P|{bar_pred}")
        print(f"  {'':>12s}  {'':>9s}  {'':>8s}  {'':>6s}  A|{bar_act}")


# ── Helper: text histogram ────────────────────────────────────────────────

def print_histogram(probs, label):
    n_bins = 20
    bins = np.linspace(0, 1, n_bins + 1)
    counts, _ = np.histogram(probs, bins=bins)
    max_count = max(counts) if counts.max() > 0 else 1
    print(f"\n  Probability histogram — {label}")
    print(f"  {'Bin':>12s}  {'Count':>6s}  Distribution")
    print("  " + "-" * 55)
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        bar_len = int(round(counts[i] / max_count * 40))
        bar = "█" * bar_len
        print(f"  [{lo:.2f},{hi:.2f})  {counts[i]:>6d}  {bar}")
    q25, q50, q75 = np.percentile(probs, [25, 50, 75])
    print(f"  mean={probs.mean():.4f}  std={probs.std():.4f}  "
          f"IQR=[{q25:.3f}, {q50:.3f}, {q75:.3f}]")


# ── Helper: save matplotlib plots if available ────────────────────────────

def try_save_plots(methods_dict, y_test):
    """Attempt to save calibration & histogram plots as PNGs."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.calibration import calibration_curve
    except ImportError:
        print("\n  (matplotlib not available — skipping PNG plots)")
        return

    # Reliability curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (name, probs) in zip(axes, methods_dict.items()):
        frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10)
        ax.plot(mean_pred, frac_pos, "s-", label=name)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_title(f"Reliability: {name}")
        ax.set_xlabel("Mean predicted")
        ax.set_ylabel("Fraction positive")
        ax.legend()
    fig.tight_layout()
    fig.savefig("calibration_reliability.png", dpi=120)
    print("  Saved calibration_reliability.png")

    # Histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, probs) in zip(axes, methods_dict.items()):
        ax.hist(probs, bins=30, edgecolor="black", alpha=0.7)
        ax.set_title(f"Distribution: {name}")
        ax.set_xlabel("P(home win)")
        ax.axvline(0.5, color="red", ls="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig("calibration_histograms.png", dpi=120)
    print("  Saved calibration_histograms.png")

    plt.close("all")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  CALIBRATION AUDIT")
    print("  Diagnosing probability compression & edge detection")
    print("=" * 70)

    # ── Load & split data ─────────────────────────────────────────────────
    print("\n1. Loading dataset...")
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

    print(f"   Train: {len(X_train)} games | Test: {len(X_test)} games")

    # ── Train ensemble (once) ─────────────────────────────────────────────
    print("\n2. Training stacked ensemble (6 base learners + meta)...")
    base_learners = build_base_learners(SEED)
    scaler = StandardScaler()
    scaler.fit(X_train)

    oof_preds, model_names = generate_oof_predictions(
        base_learners, X_train, y_train, scaler, n_splits=5
    )

    valid_mask = ~np.isnan(oof_preds).any(axis=1)
    oof_valid = oof_preds[valid_mask]
    y_valid = y_train[valid_mask]
    print(f"   Valid OOF samples: {len(y_valid)}")

    meta_learner = LogisticRegression(
        C=0.5, max_iter=1000, random_state=SEED, solver="lbfgs"
    )
    meta_learner.fit(oof_valid, y_valid)

    # ── Capture OOF meta probabilities (calibration training data) ────────
    oof_meta = meta_learner.predict_proba(oof_valid)[:, 1]

    # ── Fit calibrators ───────────────────────────────────────────────────
    print("\n3. Fitting calibrators on OOF meta probabilities...")
    sigmoid_cal = _SigmoidCalibration()
    sigmoid_cal.fit(oof_meta, y_valid)
    print(f"   Platt sigmoid params: a={sigmoid_cal.a_:.4f}, b={sigmoid_cal.b_:.4f}")

    isotonic_cal = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    isotonic_cal.fit(oof_meta, y_valid)

    # ── Train final base learners & predict on test ───────────────────────
    print("\n4. Training final base learners on full train set...")
    fitted = train_final_base_learners(base_learners, X_train, y_train, scaler)

    print("   Predicting on test set...")
    test_preds_matrix = predict_base_learners(fitted, X_test, scaler)
    raw_meta = meta_learner.predict_proba(test_preds_matrix)[:, 1]

    # Apply calibrators
    platt_probs = sigmoid_cal.predict(raw_meta)
    isotonic_probs = isotonic_cal.predict(raw_meta)

    methods = {
        "Uncalibrated": raw_meta,
        "Platt (sigmoid)": platt_probs,
        "Isotonic": isotonic_probs,
    }

    # ── Section 1: Metrics table ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SECTION 1: CALIBRATION METRICS")
    print("=" * 70)
    print(f"\n  {'Method':<20s}  {'Brier':>8s}  {'ECE':>8s}  {'LogLoss':>8s}  {'Accuracy':>8s}")
    print("  " + "-" * 58)

    for name, probs in methods.items():
        brier = brier_score_loss(y_test, probs)
        ece = compute_ece(y_test, probs)
        ll = log_loss(y_test, probs)
        preds = (probs >= 0.5).astype(int)
        acc = accuracy_score(y_test, preds)
        print(f"  {name:<20s}  {brier:>8.4f}  {ece:>8.4f}  {ll:>8.4f}  {acc:>7.1%}")

    # ── Section 2: Reliability diagrams ───────────────────────────────────
    print("\n" + "=" * 70)
    print("  SECTION 2: RELIABILITY DIAGRAMS")
    print("=" * 70)

    for name, probs in methods.items():
        print_reliability_diagram(y_test, probs, name)

    # ── Section 3: Probability histograms ─────────────────────────────────
    print("\n" + "=" * 70)
    print("  SECTION 3: PROBABILITY DISTRIBUTIONS")
    print("=" * 70)

    for name, probs in methods.items():
        print_histogram(probs, name)

    # Spread comparison
    print("\n  Spread comparison:")
    print(f"  {'Method':<20s}  {'Min':>7s}  {'Max':>7s}  {'Range':>7s}  "
          f"{'Std':>7s}  {'|p-0.5|>0.1':>12s}")
    print("  " + "-" * 68)
    for name, probs in methods.items():
        pmin, pmax = probs.min(), probs.max()
        spread = pmax - pmin
        std = probs.std()
        far_from_half = (np.abs(probs - 0.5) > 0.10).sum()
        print(f"  {name:<20s}  {pmin:>7.4f}  {pmax:>7.4f}  {spread:>7.4f}  "
              f"{std:>7.4f}  {far_from_half:>12d}")

    # ── Section 4: Trading simulation ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SECTION 4: TRADING SIMULATION")
    print("=" * 70)

    print("\n  Loading odds...")
    odds_map = load_odds_2025()
    df_test_reset = df_test.reset_index(drop=True)

    edges = [0.03, 0.05]
    print(f"\n  {'Method':<20s}  {'Edge':>5s}  {'Bets':>5s}  {'WinRate':>8s}  "
          f"{'ROI':>8s}  {'MaxDD':>8s}  {'Final$':>8s}")
    print("  " + "-" * 68)

    for name, probs in methods.items():
        for min_edge in edges:
            res = simulate_trading(
                probs, y_test, df_test_reset, odds_map,
                bankroll_dollars=10.0, max_position_dollars=2.0,
                daily_loss_dollars=5.0, min_edge=min_edge, kelly_fraction=0.25,
            )
            wr = f"{res['win_rate']:.1%}" if res["total_bets"] > 0 else "N/A"
            print(f"  {name:<20s}  {min_edge:>5.2f}  {res['total_bets']:>5d}  "
                  f"{wr:>8s}  {res['roi']:>+7.1f}%  {res['max_dd']:>7.1%}  "
                  f"${res['final_bankroll']:>7.2f}")

    # ── Section 5: Edge distribution analysis ─────────────────────────────
    print("\n" + "=" * 70)
    print("  SECTION 5: EDGE DISTRIBUTION ANALYSIS")
    print("=" * 70)

    print("\n  For each method, how many games have edges at various thresholds:")
    print(f"\n  {'Method':<20s}  {'≥1%':>6s}  {'≥3%':>6s}  {'≥5%':>6s}  "
          f"{'≥8%':>6s}  {'≥10%':>6s}  {'≥15%':>6s}")
    print("  " + "-" * 58)

    for name, probs in methods.items():
        # Count games where |model_prob - implied_prob| >= threshold
        # We need the implied probs from the odds
        edge_counts = {t: 0 for t in [0.01, 0.03, 0.05, 0.08, 0.10, 0.15]}
        dates = df_test_reset[DATE_COLUMN]
        home_names = df_test_reset["TEAM_NAME"]
        away_names = df_test_reset["TEAM_NAME.1"]

        for i in range(len(df_test_reset)):
            dt = dates.iloc[i]
            if pd.isna(dt):
                continue
            date_str = dt.strftime("%Y-%m-%d")
            home = home_names.iloc[i]
            away = away_names.iloc[i]

            odds_info = odds_map.get((date_str, home, away))
            if odds_info is None:
                continue

            contract_price = odds_to_kalshi_price(odds_info["ML_Home"], odds_info["ML_Away"])
            if contract_price is None or contract_price <= 0 or contract_price >= 100:
                continue

            implied = contract_price / 100.0
            edge = max(abs(probs[i] - implied), abs((1 - probs[i]) - (1 - implied)))

            for t in edge_counts:
                if edge >= t:
                    edge_counts[t] += 1

        counts = list(edge_counts.values())
        print(f"  {name:<20s}  {counts[0]:>6d}  {counts[1]:>6d}  {counts[2]:>6d}  "
              f"{counts[3]:>6d}  {counts[4]:>6d}  {counts[5]:>6d}")

    # ── Section 6: Summary ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SECTION 6: DIAGNOSIS & SUMMARY")
    print("=" * 70)

    # Determine which method is best by Brier score
    best_brier = None
    best_brier_name = None
    most_bets_name = None
    most_bets = -1

    for name, probs in methods.items():
        brier = brier_score_loss(y_test, probs)
        if best_brier is None or brier < best_brier:
            best_brier = brier
            best_brier_name = name

    # Check which finds most bets at 5% edge
    for name, probs in methods.items():
        res = simulate_trading(
            probs, y_test, df_test_reset, odds_map,
            min_edge=0.05, kelly_fraction=0.25,
        )
        if res["total_bets"] > most_bets:
            most_bets = res["total_bets"]
            most_bets_name = name

    raw_std = methods["Uncalibrated"].std()
    platt_std = methods["Platt (sigmoid)"].std()
    iso_std = methods["Isotonic"].std()

    print(f"""
  FINDINGS:

  1. Probability spread:
     - Uncalibrated std: {raw_std:.4f}
     - Platt std:        {platt_std:.4f}  ({'COMPRESSES' if platt_std < raw_std else 'EXPANDS'} vs raw)
     - Isotonic std:     {iso_std:.4f}  ({'COMPRESSES' if iso_std < raw_std else 'EXPANDS'} vs raw)

  2. Best calibration (Brier score): {best_brier_name} ({best_brier:.4f})

  3. Most bets at ≥5% edge: {most_bets_name} ({most_bets} bets)

  4. Diagnosis:
     If Platt std << Uncalibrated std, the sigmoid is compressing
     probabilities toward 0.5, killing perceived edges.
     Compression ratio: {platt_std / raw_std:.2f}x (1.0 = no change, <1 = compression)
""")

    # ── Optional: save plots ──────────────────────────────────────────────
    try_save_plots(methods, y_test)

    print("=" * 70)
    print("  Audit complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
