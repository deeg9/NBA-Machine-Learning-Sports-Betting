"""Bayesian probability updater for real-time signal incorporation.

Turns the static ensemble model probability into a living number that
updates based on market signals (line movements, volume), injury reports,
and reverse line movement (sharp money detection).

Each signal produces a likelihood ratio (LR) applied via Bayes' theorem.
A max_total_shift cap prevents the posterior from diverging too far
from the calibrated ensemble output.
"""

from __future__ import annotations

import math

# Default configuration
DEFAULT_CONFIG = {
    "use_line_movement": True,
    "use_injuries": True,
    "use_reverse_line": True,
    "line_strength": 1.0,
    "injury_strength": 1.0,
    "max_total_shift": 0.15,
}


def _clamp_prob(p: float) -> float:
    """Clamp probability to [0.01, 0.99] to avoid Bayesian degeneration."""
    return max(0.01, min(0.99, p))


def bayesian_update(prior: float, likelihood_ratio: float) -> float:
    """Apply a single Bayesian update.

    Parameters
    ----------
    prior : float
        Prior probability P(home_win), (0-1).
    likelihood_ratio : float
        LR = P(signal | home_win) / P(signal | home_loss).
        LR > 1 means signal favors home win.
        LR < 1 means signal favors home loss.
        LR = 1 means no information.

    Returns
    -------
    float
        Posterior probability.
    """
    prior = _clamp_prob(prior)
    if likelihood_ratio <= 0:
        return prior

    # Bayes: posterior = (prior * LR) / (prior * LR + (1 - prior))
    numerator = prior * likelihood_ratio
    denominator = numerator + (1 - prior)
    return numerator / denominator


def line_movement_lr(
    current_price: int,
    previous_price: int,
    strength: float = 1.0,
) -> float:
    """Likelihood ratio from Kalshi YES price change.

    A price increase means the market is moving toward YES (home win),
    implying new information favoring the home team.

    Parameters
    ----------
    current_price : int
        Current YES contract price in cents.
    previous_price : int
        Previous YES contract price in cents.
    strength : float
        Multiplier on the LR magnitude.

    Returns
    -------
    float
        Likelihood ratio. > 1 if price moved up, < 1 if down.
    """
    delta = current_price - previous_price
    if delta == 0:
        return 1.0
    # Each cent of movement contributes 2% to the LR
    return 1.0 + 0.02 * delta * strength


def injury_lr(out_count_delta: int, is_home: bool) -> float:
    """Likelihood ratio from injury report changes.

    Each new player ruled OUT reduces that team's win probability.

    Parameters
    ----------
    out_count_delta : int
        Number of newly OUT players (positive = more players out).
    is_home : bool
        True if the injuries are on the home team.

    Returns
    -------
    float
        Likelihood ratio for P(home_win).
        Home injuries → LR < 1 (hurts home).
        Away injuries → LR > 1 (helps home).
    """
    if out_count_delta <= 0:
        return 1.0

    # Each OUT player shifts LR by ~8%
    per_player_lr = 0.92
    raw_lr = per_player_lr ** out_count_delta

    if is_home:
        # Home team injuries hurt home win probability
        return raw_lr  # < 1
    else:
        # Away team injuries help home win probability
        return 1.0 / raw_lr  # > 1


def reverse_line_lr(price_delta: int, heavier_volume_side: str) -> float:
    """Likelihood ratio for reverse line movement.

    When price moves in one direction but volume is heavier on the other
    side, this signals sharp (informed) money on the price-movement side.

    Parameters
    ----------
    price_delta : int
        Change in YES price (positive = YES price went up).
    heavier_volume_side : str
        'yes' or 'no' — which side has more volume.

    Returns
    -------
    float
        Likelihood ratio. 1.08 if reverse line detected, 1.0 otherwise.
    """
    if price_delta == 0:
        return 1.0

    price_favors_yes = price_delta > 0
    volume_favors_yes = heavier_volume_side == "yes"

    # Reverse line: price and volume disagree → sharp money on price side
    if price_favors_yes != volume_favors_yes:
        # Sharp money is on the price direction
        return 1.08 if price_favors_yes else 0.93
    return 1.0


def update_probability(
    prior: float,
    market_snapshot: dict | None = None,
    injury_delta: dict | None = None,
    config: dict | None = None,
    backtest_mode: bool = False,
) -> tuple[float, list[dict]]:
    """Apply all enabled Bayesian signals to update probability.

    Parameters
    ----------
    prior : float
        Ensemble model probability P(home_win).
    market_snapshot : dict or None
        Keys: current_price, previous_price, yes_volume, no_volume.
    injury_delta : dict or None
        Keys: home_out_delta, away_out_delta.
    config : dict or None
        Signal toggles and strengths. Uses DEFAULT_CONFIG if None.
    backtest_mode : bool
        If True, skip market-data signals (line movement, reverse line).

    Returns
    -------
    tuple[float, list[dict]]
        (posterior, update_log) where update_log records each signal applied.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    posterior = _clamp_prob(prior)
    update_log = []

    # Signal 1: Line movement
    if (
        cfg["use_line_movement"]
        and not backtest_mode
        and market_snapshot
        and market_snapshot.get("previous_price") is not None
    ):
        lr = line_movement_lr(
            market_snapshot["current_price"],
            market_snapshot["previous_price"],
            strength=cfg["line_strength"],
        )
        if lr != 1.0:
            new_posterior = bayesian_update(posterior, lr)
            update_log.append({
                "signal": "line_movement",
                "prior": round(posterior, 4),
                "lr": round(lr, 4),
                "posterior": round(new_posterior, 4),
            })
            posterior = new_posterior

    # Signal 2: Injury updates
    if cfg["use_injuries"] and injury_delta:
        # Home team injuries
        home_delta = injury_delta.get("home_out_delta", 0)
        if home_delta > 0:
            lr = injury_lr(home_delta, is_home=True)
            # Apply injury strength modifier
            lr = 1.0 + (lr - 1.0) * cfg["injury_strength"]
            new_posterior = bayesian_update(posterior, lr)
            update_log.append({
                "signal": "injury_home",
                "prior": round(posterior, 4),
                "lr": round(lr, 4),
                "posterior": round(new_posterior, 4),
            })
            posterior = new_posterior

        # Away team injuries
        away_delta = injury_delta.get("away_out_delta", 0)
        if away_delta > 0:
            lr = injury_lr(away_delta, is_home=False)
            lr = 1.0 + (lr - 1.0) * cfg["injury_strength"]
            new_posterior = bayesian_update(posterior, lr)
            update_log.append({
                "signal": "injury_away",
                "prior": round(posterior, 4),
                "lr": round(lr, 4),
                "posterior": round(new_posterior, 4),
            })
            posterior = new_posterior

    # Signal 3: Reverse line movement
    if (
        cfg["use_reverse_line"]
        and not backtest_mode
        and market_snapshot
        and market_snapshot.get("previous_price") is not None
    ):
        price_delta = (
            market_snapshot["current_price"] - market_snapshot["previous_price"]
        )
        yes_vol = market_snapshot.get("yes_volume") or 0
        no_vol = market_snapshot.get("no_volume") or 0
        if yes_vol > 0 or no_vol > 0:
            heavier_side = "yes" if yes_vol >= no_vol else "no"
            lr = reverse_line_lr(price_delta, heavier_side)
            if lr != 1.0:
                new_posterior = bayesian_update(posterior, lr)
                update_log.append({
                    "signal": "reverse_line",
                    "prior": round(posterior, 4),
                    "lr": round(lr, 4),
                    "posterior": round(new_posterior, 4),
                })
                posterior = new_posterior

    # Enforce max_total_shift cap
    max_shift = cfg["max_total_shift"]
    if abs(posterior - prior) > max_shift:
        if posterior > prior:
            posterior = prior + max_shift
        else:
            posterior = prior - max_shift
        update_log.append({
            "signal": "max_shift_cap",
            "capped_at": round(posterior, 4),
        })

    return round(_clamp_prob(posterior), 4), update_log
