"""
Enhanced value betting strategy module.

Builds on the basic EV and Kelly Criterion calculations with configurable
thresholds, fractional Kelly sizing, closing line value tracking, bankroll
simulation, and ROI reporting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional


# ---------------------------------------------------------------------------
# 1. Implied probability
# ---------------------------------------------------------------------------

def implied_probability(american_odds: float, vig_adjust: bool = False,
                        overround: float = 0.0) -> float:
    """Convert American odds to implied probability.

    Parameters
    ----------
    american_odds : float
        American-format odds (e.g. -110, +250).
    vig_adjust : bool
        If *True*, remove the bookmaker vig using *overround*.
    overround : float
        Total overround of the market (e.g. 0.05 for a 105 % book).
        Only used when *vig_adjust* is True.

    Returns
    -------
    float
        Implied probability in the range [0, 1].
    """
    if american_odds < 0:
        raw = abs(american_odds) / (abs(american_odds) + 100)
    else:
        raw = 100 / (american_odds + 100)

    if vig_adjust and overround > 0:
        raw = raw / (1 + overround)

    return round(raw, 6)


# ---------------------------------------------------------------------------
# 2. Configurable EV threshold
# ---------------------------------------------------------------------------

@dataclass
class BetRecommendation:
    should_bet: bool
    edge: float
    ev: float


def should_bet(model_prob: float, american_odds: float,
               threshold: float = 0.05) -> BetRecommendation:
    """Recommend a bet only when the model edge exceeds *threshold*.

    Parameters
    ----------
    model_prob : float
        Model's estimated win probability (0-1).
    american_odds : float
        American-format odds offered by the book.
    threshold : float
        Minimum edge required to recommend the bet (default 5 %).

    Returns
    -------
    BetRecommendation
        Named result with *should_bet*, *edge*, and *ev* fields.
    """
    imp_prob = implied_probability(american_odds)
    edge = round(model_prob - imp_prob, 6)

    # Decimal payout per $1 wagered (profit portion only)
    if american_odds < 0:
        decimal_profit = 100 / abs(american_odds)
    else:
        decimal_profit = american_odds / 100

    ev = round(model_prob * decimal_profit - (1 - model_prob), 6)

    return BetRecommendation(
        should_bet=edge >= threshold,
        edge=edge,
        ev=ev,
    )


# ---------------------------------------------------------------------------
# 3. Fractional Kelly
# ---------------------------------------------------------------------------

def fractional_kelly(american_odds: float, model_prob: float,
                     fraction: float = 0.25,
                     max_pct: float = 5.0) -> float:
    """Compute fractional Kelly stake as a percentage of bankroll.

    Full Kelly is notoriously aggressive for sports betting.  This function
    defaults to quarter-Kelly and hard-caps the stake at *max_pct* %.

    Parameters
    ----------
    american_odds : float
        American-format odds.
    model_prob : float
        Model's estimated win probability (0-1).
    fraction : float
        Kelly fraction (0.25 = quarter, 0.50 = half, 1.0 = full).
    max_pct : float
        Maximum bet size as a percentage of bankroll.

    Returns
    -------
    float
        Recommended stake as a percentage of bankroll (0 if negative edge).
    """
    if american_odds < 0:
        decimal_odds = 1 + 100 / abs(american_odds)
    else:
        decimal_odds = 1 + american_odds / 100

    b = decimal_odds - 1  # net odds (profit per $1)
    p = model_prob
    q = 1 - p

    if b <= 0:
        return 0.0

    kelly_pct = ((b * p - q) / b) * 100  # full Kelly %

    if kelly_pct <= 0:
        return 0.0

    sized = kelly_pct * fraction
    return round(min(sized, max_pct), 4)


# ---------------------------------------------------------------------------
# 4. CLV Tracking
# ---------------------------------------------------------------------------

@dataclass
class CLVRecord:
    game_id: str
    bet_time_odds: float
    closing_odds: float
    model_prob: float
    result: int  # 1 = win, 0 = loss


@dataclass
class CLVReport:
    total_bets: int
    avg_clv: float
    pct_beat_close: float
    cumulative_clv: float


class ClosingLineTracker:
    """Track whether bets consistently beat the closing line."""

    def __init__(self) -> None:
        self.records: List[CLVRecord] = []

    def add(self, game_id: str, bet_time_odds: float, closing_odds: float,
            model_prob: float, result: int) -> None:
        self.records.append(CLVRecord(
            game_id=game_id,
            bet_time_odds=bet_time_odds,
            closing_odds=closing_odds,
            model_prob=model_prob,
            result=result,
        ))

    def clv_for_record(self, rec: CLVRecord) -> float:
        """CLV = implied_prob(closing) - implied_prob(bet_time).

        A positive value means the line moved in our favour after we bet,
        indicating we captured value.
        """
        return (implied_probability(rec.closing_odds)
                - implied_probability(rec.bet_time_odds))

    def report(self) -> CLVReport:
        if not self.records:
            return CLVReport(0, 0.0, 0.0, 0.0)

        clvs = [self.clv_for_record(r) for r in self.records]
        beats = sum(1 for c in clvs if c > 0)

        return CLVReport(
            total_bets=len(clvs),
            avg_clv=round(sum(clvs) / len(clvs), 6),
            pct_beat_close=round(beats / len(clvs) * 100, 2),
            cumulative_clv=round(sum(clvs), 6),
        )


# ---------------------------------------------------------------------------
# 5. Bankroll Simulator
# ---------------------------------------------------------------------------

@dataclass
class BetRecord:
    american_odds: float
    model_prob: float
    result: int  # 1 = win, 0 = loss


@dataclass
class SimulationResult:
    final_bankroll: float
    max_drawdown: float
    roi: float
    num_bets: int
    win_rate: float
    trajectory: List[float]


def simulate_bankroll(bets: List[BetRecord],
                      starting_bankroll: float = 1000.0,
                      kelly_fraction: float = 0.25,
                      max_pct: float = 5.0,
                      ev_threshold: float = 0.05) -> SimulationResult:
    """Simulate bankroll trajectory over a sequence of bets.

    Parameters
    ----------
    bets : list[BetRecord]
        Historical or hypothetical bets with known results.
    starting_bankroll : float
        Initial bankroll in dollars.
    kelly_fraction : float
        Fraction of Kelly to wager (default quarter-Kelly).
    max_pct : float
        Maximum bet as a percentage of current bankroll.
    ev_threshold : float
        Minimum edge to place a bet.

    Returns
    -------
    SimulationResult
        Summary statistics and the full bankroll trajectory.
    """
    bankroll = starting_bankroll
    trajectory: List[float] = [bankroll]
    peak = bankroll
    max_drawdown = 0.0
    bets_placed = 0
    wins = 0

    for bet in bets:
        rec = should_bet(bet.model_prob, bet.american_odds, threshold=ev_threshold)
        if not rec.should_bet:
            trajectory.append(bankroll)
            continue

        stake_pct = fractional_kelly(
            bet.american_odds, bet.model_prob,
            fraction=kelly_fraction, max_pct=max_pct,
        )
        stake = bankroll * stake_pct / 100

        if stake <= 0:
            trajectory.append(bankroll)
            continue

        bets_placed += 1

        if bet.result == 1:
            if bet.american_odds < 0:
                profit = stake * (100 / abs(bet.american_odds))
            else:
                profit = stake * (bet.american_odds / 100)
            bankroll += profit
            wins += 1
        else:
            bankroll -= stake

        trajectory.append(round(bankroll, 2))

        if bankroll > peak:
            peak = bankroll
        dd = (peak - bankroll) / peak if peak > 0 else 0.0
        if dd > max_drawdown:
            max_drawdown = dd

    roi = ((bankroll - starting_bankroll) / starting_bankroll) * 100 if starting_bankroll else 0.0

    return SimulationResult(
        final_bankroll=round(bankroll, 2),
        max_drawdown=round(max_drawdown * 100, 2),
        roi=round(roi, 2),
        num_bets=bets_placed,
        win_rate=round(wins / bets_placed * 100, 2) if bets_placed else 0.0,
        trajectory=trajectory,
    )


# ---------------------------------------------------------------------------
# 6. ROI Report
# ---------------------------------------------------------------------------

@dataclass
class MonthlyBreakdown:
    month: str
    bets: int
    wins: int
    pnl: float
    roi: float


@dataclass
class ROIReport:
    total_bets: int
    bets_placed: int
    win_rate: float
    roi_flat: float
    roi_kelly: float
    max_drawdown: float
    sharpe_ratio: float
    monthly: List[MonthlyBreakdown]


def _sharpe(daily_returns: List[float]) -> float:
    """Annualised Sharpe ratio from a list of daily return percentages."""
    if len(daily_returns) < 2:
        return 0.0
    mean = sum(daily_returns) / len(daily_returns)
    var = sum((r - mean) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
    std = math.sqrt(var) if var > 0 else 0.0
    if std == 0:
        return 0.0
    return round((mean / std) * math.sqrt(365), 4)


def generate_roi_report(
    bets: List[BetRecord],
    dates: Optional[List[str]] = None,
    starting_bankroll: float = 1000.0,
    kelly_fraction: float = 0.25,
    max_pct: float = 5.0,
    ev_threshold: float = 0.05,
) -> ROIReport:
    """Generate a comprehensive ROI report from historical predictions.

    Parameters
    ----------
    bets : list[BetRecord]
        Each entry holds odds, model probability, and actual result.
    dates : list[str] or None
        ISO date strings (YYYY-MM-DD) aligned with *bets*.  Used for
        monthly breakdown and Sharpe calculation.  If *None*, all bets
        are treated as a single period.
    starting_bankroll : float
        Starting bankroll for Kelly simulation.
    kelly_fraction : float
        Fraction of Kelly to use.
    max_pct : float
        Max bet cap.
    ev_threshold : float
        Minimum edge.

    Returns
    -------
    ROIReport
    """
    sim = simulate_bankroll(bets, starting_bankroll, kelly_fraction,
                            max_pct, ev_threshold)

    # Flat-bet ROI: assume $100 per qualifying bet
    flat_bankroll = 0.0
    flat_bets = 0
    flat_wins = 0
    for bet in bets:
        rec = should_bet(bet.model_prob, bet.american_odds, threshold=ev_threshold)
        if not rec.should_bet:
            continue
        flat_bets += 1
        if bet.result == 1:
            if bet.american_odds < 0:
                flat_bankroll += 100 * (100 / abs(bet.american_odds))
            else:
                flat_bankroll += 100 * (bet.american_odds / 100)
            flat_wins += 1
        else:
            flat_bankroll -= 100

    flat_roi = round(flat_bankroll / (flat_bets * 100) * 100, 2) if flat_bets else 0.0

    # Daily returns for Sharpe
    daily_returns: List[float] = []
    if dates:
        day_pnl: dict[str, float] = {}
        day_bank: dict[str, float] = {}
        running = starting_bankroll
        for i, bet in enumerate(bets):
            d = dates[i]
            rec = should_bet(bet.model_prob, bet.american_odds, threshold=ev_threshold)
            if not rec.should_bet:
                continue
            stake_pct = fractional_kelly(bet.american_odds, bet.model_prob,
                                         fraction=kelly_fraction, max_pct=max_pct)
            stake = running * stake_pct / 100
            if stake <= 0:
                continue
            if bet.result == 1:
                if bet.american_odds < 0:
                    pnl = stake * (100 / abs(bet.american_odds))
                else:
                    pnl = stake * (bet.american_odds / 100)
            else:
                pnl = -stake
            running += pnl
            day_pnl[d] = day_pnl.get(d, 0.0) + pnl
            day_bank[d] = running

        prev = starting_bankroll
        for d in sorted(day_pnl):
            ret = day_pnl[d] / prev * 100 if prev else 0.0
            daily_returns.append(ret)
            prev = day_bank[d]

    sharpe = _sharpe(daily_returns)

    # Monthly breakdown
    monthly: List[MonthlyBreakdown] = []
    if dates:
        month_data: dict[str, dict] = {}
        running = starting_bankroll
        for i, bet in enumerate(bets):
            d = dates[i]
            month_key = d[:7]  # YYYY-MM
            rec = should_bet(bet.model_prob, bet.american_odds, threshold=ev_threshold)
            if not rec.should_bet:
                continue
            if month_key not in month_data:
                month_data[month_key] = {"bets": 0, "wins": 0, "pnl": 0.0,
                                         "start_bank": running}
            stake_pct = fractional_kelly(bet.american_odds, bet.model_prob,
                                         fraction=kelly_fraction, max_pct=max_pct)
            stake = running * stake_pct / 100
            if stake <= 0:
                continue
            month_data[month_key]["bets"] += 1
            if bet.result == 1:
                if bet.american_odds < 0:
                    pnl = stake * (100 / abs(bet.american_odds))
                else:
                    pnl = stake * (bet.american_odds / 100)
                month_data[month_key]["wins"] += 1
            else:
                pnl = -stake
            running += pnl
            month_data[month_key]["pnl"] += pnl

        for m in sorted(month_data):
            md = month_data[m]
            mroi = round(md["pnl"] / md["start_bank"] * 100, 2) if md["start_bank"] else 0.0
            monthly.append(MonthlyBreakdown(
                month=m,
                bets=md["bets"],
                wins=md["wins"],
                pnl=round(md["pnl"], 2),
                roi=mroi,
            ))

    return ROIReport(
        total_bets=len(bets),
        bets_placed=sim.num_bets,
        win_rate=sim.win_rate,
        roi_flat=flat_roi,
        roi_kelly=sim.roi,
        max_drawdown=sim.max_drawdown,
        sharpe_ratio=sharpe,
        monthly=monthly,
    )


# ---------------------------------------------------------------------------
# 7. Kalshi edge helper
# ---------------------------------------------------------------------------

def kalshi_edge(model_prob: float, contract_price_cents: int,
                threshold: float = 0.05) -> BetRecommendation:
    """Compute edge between model probability and a Kalshi contract price.

    Converts Kalshi contract price (1-99 cents) to implied probability,
    then delegates to :func:`should_bet` using synthetic American odds.

    Parameters
    ----------
    model_prob : float
        Model's estimated probability for the YES outcome (0-1).
    contract_price_cents : int
        Kalshi YES contract price in cents (1-99).
    threshold : float
        Minimum edge to recommend a bet (default 5 %).

    Returns
    -------
    BetRecommendation
        Named result with *should_bet*, *edge*, and *ev* fields.
    """
    # Convert contract price to implied probability
    implied = contract_price_cents / 100.0

    # Convert to equivalent American odds for should_bet()
    if implied >= 0.5:
        american_odds = -(implied / (1 - implied)) * 100
    else:
        american_odds = ((1 - implied) / implied) * 100

    return should_bet(model_prob, american_odds, threshold=threshold)


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

def _demo() -> None:
    """Run a quick demonstration with synthetic data."""
    import random
    random.seed(42)

    print("=" * 60)
    print("  NBA Value Betting Strategy - Demo")
    print("=" * 60)

    # --- Implied probability ---
    print("\n1. Implied Probability")
    for odds in [-110, +150, -200, +300]:
        ip = implied_probability(odds)
        print(f"   Odds {odds:>+5d}  ->  implied prob = {ip:.4f}")

    # --- Should bet ---
    print("\n2. Should-Bet Evaluation (threshold = 5 %)")
    cases = [
        (0.60, -110, "Model 60 %, odds -110"),
        (0.55, -110, "Model 55 %, odds -110"),
        (0.45, +150, "Model 45 %, odds +150"),
    ]
    for prob, odds, label in cases:
        rec = should_bet(prob, odds)
        print(f"   {label:30s}  ->  bet={rec.should_bet}  edge={rec.edge:+.4f}  "
              f"ev={rec.ev:+.4f}")

    # --- Fractional Kelly ---
    print("\n3. Fractional Kelly Sizing")
    for frac, label in [(0.25, "Quarter"), (0.50, "Half"), (1.0, "Full")]:
        pct = fractional_kelly(-110, 0.60, fraction=frac)
        print(f"   {label:8s} Kelly @ 60 % / -110  ->  {pct:.2f} % of bankroll")

    # --- CLV Tracker ---
    print("\n4. Closing Line Value Tracking")
    tracker = ClosingLineTracker()
    tracker.add("G001", -110, -120, 0.58, 1)
    tracker.add("G002", +150, +140, 0.42, 0)
    tracker.add("G003", -105, -115, 0.56, 1)
    tracker.add("G004", +200, +180, 0.38, 1)
    clv_rpt = tracker.report()
    print(f"   Total bets:       {clv_rpt.total_bets}")
    print(f"   Avg CLV:          {clv_rpt.avg_clv:+.4f}")
    print(f"   Beat close:       {clv_rpt.pct_beat_close:.1f} %")
    print(f"   Cumulative CLV:   {clv_rpt.cumulative_clv:+.4f}")

    # --- Bankroll simulation ---
    print("\n5. Bankroll Simulation (quarter-Kelly, 200 bets)")
    sample_bets: List[BetRecord] = []
    months = ["2025-11", "2025-12", "2026-01", "2026-02"]
    sample_dates: List[str] = []
    for i in range(200):
        odds = random.choice([-110, -120, +130, +150, -105, +200])
        true_prob = implied_probability(odds) + random.uniform(-0.08, 0.12)
        true_prob = max(0.05, min(0.95, true_prob))
        win = 1 if random.random() < true_prob else 0
        sample_bets.append(BetRecord(
            american_odds=odds,
            model_prob=true_prob,
            result=win,
        ))
        m = months[i * len(months) // 200]
        day = (i % 28) + 1
        sample_dates.append(f"{m}-{day:02d}")

    sim = simulate_bankroll(sample_bets, starting_bankroll=1000.0)
    print(f"   Starting bankroll: $1,000.00")
    print(f"   Final bankroll:    ${sim.final_bankroll:,.2f}")
    print(f"   ROI:               {sim.roi:+.2f} %")
    print(f"   Max drawdown:      {sim.max_drawdown:.2f} %")
    print(f"   Bets placed:       {sim.num_bets}")
    print(f"   Win rate:          {sim.win_rate:.1f} %")

    # Mini sparkline of trajectory
    traj = sim.trajectory
    n = len(traj)
    steps = min(n, 20)
    indices = [int(i * (n - 1) / (steps - 1)) for i in range(steps)]
    vals = [traj[i] for i in indices]
    lo, hi = min(vals), max(vals)
    spread = hi - lo if hi != lo else 1
    bar_chars = " _.-~*^"
    line = ""
    for v in vals:
        idx = int((v - lo) / spread * (len(bar_chars) - 1))
        line += bar_chars[idx]
    print(f"   Trajectory:        [{line}]")

    # --- ROI Report ---
    print("\n6. ROI Report")
    rpt = generate_roi_report(sample_bets, dates=sample_dates,
                              starting_bankroll=1000.0)
    print(f"   Total bets:        {rpt.total_bets}")
    print(f"   Qualifying bets:   {rpt.bets_placed}")
    print(f"   Win rate:          {rpt.win_rate:.1f} %")
    print(f"   ROI (flat):        {rpt.roi_flat:+.2f} %")
    print(f"   ROI (Kelly):       {rpt.roi_kelly:+.2f} %")
    print(f"   Max drawdown:      {rpt.max_drawdown:.2f} %")
    print(f"   Sharpe ratio:      {rpt.sharpe_ratio:.4f}")
    if rpt.monthly:
        print("   Monthly breakdown:")
        for mb in rpt.monthly:
            print(f"     {mb.month}  bets={mb.bets:3d}  wins={mb.wins:3d}  "
                  f"P&L=${mb.pnl:>+9.2f}  ROI={mb.roi:+.2f} %")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    _demo()
