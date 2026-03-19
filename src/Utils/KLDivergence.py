"""KL Divergence module for detecting mispricing between model and market.

Formalizes edge detection using information theory. Ranks trades by how
divergent the model's probability distribution is from the market's implied
distribution. Also detects cross-market inconsistencies when multiple
contract types exist for the same game.
"""

from __future__ import annotations

import math


def _clamp(p: float, lo: float = 0.001, hi: float = 0.999) -> float:
    """Clamp probability to avoid log(0)."""
    return max(lo, min(hi, p))


def kl_divergence_binary(p: float, q: float) -> float:
    """KL divergence KL(P || Q) between two binary distributions.

    Parameters
    ----------
    p : float
        Model probability (0-1).
    q : float
        Market-implied probability (0-1).

    Returns
    -------
    float
        Non-negative KL divergence value.
    """
    p = _clamp(p)
    q = _clamp(q)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def symmetric_kl(p: float, q: float) -> float:
    """Symmetric KL divergence (average of both directions).

    More stable than directional KL for threshold-based decisions.

    Parameters
    ----------
    p, q : float
        Probabilities (0-1).

    Returns
    -------
    float
        (KL(P||Q) + KL(Q||P)) / 2
    """
    return (kl_divergence_binary(p, q) + kl_divergence_binary(q, p)) / 2


def detect_mispricing(
    games: list[dict],
    threshold: float = 0.05,
) -> list[dict]:
    """Score each game by KL divergence between model and market.

    Parameters
    ----------
    games : list[dict]
        Each dict must have: ticker, model_prob, market_price_cents.
    threshold : float
        Minimum symmetric KL divergence to flag as mispriced.

    Returns
    -------
    list[dict]
        Games exceeding threshold, sorted by divergence descending.
        Each result has: ticker, model_prob, market_implied, kl_divergence,
        direction ('model_higher' or 'market_higher'), edge.
    """
    results = []
    for game in games:
        model_prob = game["model_prob"]
        market_implied = game["market_price_cents"] / 100.0

        kl = symmetric_kl(model_prob, market_implied)

        if kl >= threshold:
            edge = model_prob - market_implied
            results.append({
                "ticker": game["ticker"],
                "model_prob": round(model_prob, 4),
                "market_implied": round(market_implied, 4),
                "kl_divergence": round(kl, 6),
                "direction": "model_higher" if edge > 0 else "market_higher",
                "edge": round(edge, 4),
            })

    results.sort(key=lambda r: r["kl_divergence"], reverse=True)
    return results


def scan_cross_market(markets: list[dict]) -> list[dict]:
    """Compare implied probabilities across different market types for the same game.

    Parameters
    ----------
    markets : list[dict]
        Each dict should have: ticker, market_type ('moneyline', 'spread',
        'over_under'), implied_prob, home_team, away_team.

    Returns
    -------
    list[dict]
        Pairs of markets with inconsistent implied probabilities.
        Each result has: ticker_a, ticker_b, prob_a, prob_b, kl_divergence.
        Returns empty list if fewer than 2 market types exist per game.
    """
    # Group markets by game (home_team + away_team)
    game_markets: dict[str, list[dict]] = {}
    for m in markets:
        key = f"{m.get('home_team', '')}:{m.get('away_team', '')}"
        game_markets.setdefault(key, []).append(m)

    results = []
    for game_key, group in game_markets.items():
        if len(group) < 2:
            continue

        # Compare every pair
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                pa = a.get("implied_prob", 0.5)
                pb = b.get("implied_prob", 0.5)
                kl = symmetric_kl(pa, pb)

                if kl > 0.01:  # low threshold for cross-market
                    results.append({
                        "ticker_a": a["ticker"],
                        "ticker_b": b["ticker"],
                        "market_type_a": a.get("market_type", "unknown"),
                        "market_type_b": b.get("market_type", "unknown"),
                        "prob_a": round(pa, 4),
                        "prob_b": round(pb, 4),
                        "kl_divergence": round(kl, 6),
                    })

    results.sort(key=lambda r: r["kl_divergence"], reverse=True)
    return results
