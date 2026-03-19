"""Avellaneda-Stoikov execution model for optimal limit order placement.

Instead of placing limit orders at the current market price, this module
calculates a "reservation price" that accounts for inventory risk and
time decay. On Kalshi's order book, this means placing limit orders at
better prices that save cents per contract.

Core formula:
    reservation_price = fair_value - q * gamma * sigma^2 * T

Where:
    fair_value = model_prob * 100 (in cents)
    q = current inventory (positive = long YES)
    gamma = risk aversion parameter
    sigma = contract price volatility
    T = time to market close (fraction of day)
"""

from __future__ import annotations

import math


def estimate_volatility(
    current_price: int,
    price_history: list[int] | None = None,
) -> float:
    """Estimate contract price volatility.

    For binary contracts, the analytical volatility is sqrt(p * (1-p))
    where p = price/100. A 50c contract has max vol (0.5), while extreme
    prices have lower vol.

    If price_history is provided, uses realized volatility instead.

    Parameters
    ----------
    current_price : int
        Current YES price in cents (1-99).
    price_history : list[int] or None
        Historical prices for realized vol calculation.

    Returns
    -------
    float
        Estimated volatility (in probability units, 0-0.5).
    """
    if price_history and len(price_history) >= 3:
        # Realized volatility from price changes
        returns = []
        for i in range(1, len(price_history)):
            if price_history[i - 1] > 0:
                ret = (price_history[i] - price_history[i - 1]) / price_history[i - 1]
                returns.append(ret)
        if returns:
            mean_ret = sum(returns) / len(returns)
            variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
            return min(math.sqrt(variance), 0.5)

    # Analytical binary contract vol
    p = max(0.01, min(0.99, current_price / 100.0))
    return math.sqrt(p * (1 - p))


def reservation_price(
    fair_value_cents: float,
    inventory: int,
    gamma: float,
    sigma: float,
    time_remaining: float,
) -> float:
    """Compute Avellaneda-Stoikov reservation price.

    Parameters
    ----------
    fair_value_cents : float
        Model's fair value for the YES contract in cents.
    inventory : int
        Current net position (positive = long YES contracts).
    gamma : float
        Risk aversion parameter (0.01 = aggressive, 1.0 = very conservative).
    sigma : float
        Contract price volatility.
    time_remaining : float
        Time to market close as fraction of a day (0-1).

    Returns
    -------
    float
        Reservation price in cents.
    """
    # r = s - q * gamma * sigma^2 * T
    inventory_penalty = inventory * gamma * (sigma ** 2) * time_remaining * 100
    return fair_value_cents - inventory_penalty


def optimal_limit_price(
    model_prob: float,
    current_position: int,
    time_to_close_hours: float,
    orderbook: dict,
    side: str,
    gamma: float = 0.1,
    max_improvement: int = 5,
    price_history: list[int] | None = None,
) -> dict:
    """Calculate the optimal limit order price for a directional trade.

    Parameters
    ----------
    model_prob : float
        Model's P(home_win) probability (0-1).
    current_position : int
        Net position in this contract (positive = long YES).
    time_to_close_hours : float
        Hours until market closes.
    orderbook : dict
        From KalshiClient.get_orderbook(): {yes_bid, yes_ask, no_bid, no_ask}.
    side : str
        'yes' or 'no' — which side we're buying.
    gamma : float
        Risk aversion (default 0.1).
    max_improvement : int
        Maximum cents below market price to bid (safety cap).
    price_history : list[int] or None
        Historical YES prices for vol estimation.

    Returns
    -------
    dict
        reservation_price: float (cents),
        optimal_price: int (1-99, what to submit),
        improvement_cents: int (savings vs naive),
        fill_probability: float (estimated chance order fills).
    """
    fair_value_cents = model_prob * 100

    # Determine naive (market) price
    if side == "yes":
        market_price = orderbook.get("yes_ask") or int(round(fair_value_cents))
    else:
        market_price = orderbook.get("no_ask") or int(round((1 - model_prob) * 100))

    # Safety: if very close to market close, bypass Stoikov
    if time_to_close_hours < 0.1:
        return {
            "reservation_price": fair_value_cents,
            "optimal_price": max(1, min(99, market_price)),
            "improvement_cents": 0,
            "fill_probability": 1.0,
        }

    # Estimate volatility
    price_for_vol = market_price if side == "yes" else (100 - market_price)
    sigma = estimate_volatility(price_for_vol, price_history)

    # Time remaining as fraction of day
    time_remaining = max(0.0, time_to_close_hours / 24.0)

    # Adjust inventory sign for the side we're trading
    # If buying YES, positive inventory means we already have YES exposure
    # If buying NO, flip the sign (NO inventory = negative YES inventory)
    effective_inventory = current_position if side == "yes" else -current_position

    # Compute reservation price
    res_price = reservation_price(
        fair_value_cents=fair_value_cents if side == "yes" else (100 - model_prob * 100),
        inventory=effective_inventory,
        gamma=gamma,
        sigma=sigma,
        time_remaining=time_remaining,
    )

    # Optimal bid: slightly below reservation price
    # For directional trading, bid at reservation - small spread
    spread = gamma * sigma * time_remaining * 50  # half-spread
    raw_optimal = res_price - spread

    # Round to integer (Kalshi tick = 1 cent)
    optimal = int(round(raw_optimal))

    # Apply max_improvement cap: don't go more than N cents below market
    if market_price - optimal > max_improvement:
        optimal = market_price - max_improvement

    # Clamp to valid range
    optimal = max(1, min(99, optimal))

    # Don't bid above market (that's worse than naive)
    if optimal > market_price:
        optimal = market_price

    improvement = max(0, market_price - optimal)

    # Estimate fill probability: exponential decay with distance from market
    if improvement == 0:
        fill_prob = 0.95  # near-certain fill at market
    else:
        # Each cent away from market reduces fill probability
        fill_prob = math.exp(-0.3 * improvement)
        fill_prob = max(0.05, min(0.95, fill_prob))

    return {
        "reservation_price": round(res_price, 2),
        "optimal_price": optimal,
        "improvement_cents": improvement,
        "fill_probability": round(fill_prob, 3),
    }
