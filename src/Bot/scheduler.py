"""Game-time scheduler for the NBA Kalshi trading bot.

Reads the NBA schedule CSV, finds today's games, and schedules
the live trader to run 15 minutes before each tip-off.

Usage:
    python -m src.Bot.scheduler
"""

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import schedule

BASE_DIR = Path(__file__).resolve().parents[2]
SCHEDULE_PATH = BASE_DIR / "Data" / "nba-2025-UTC.csv"

# Eastern Time for game tip-offs; schedule CSV is in UTC
ET = ZoneInfo("America/New_York")
UTC = timezone.utc

LEAD_TIME_MINUTES = 15


def get_todays_games():
    """Load today's games from the schedule CSV.

    Returns
    -------
    list[dict]
        Each dict has: home_team, away_team, tipoff_utc (datetime)
    """
    df = pd.read_csv(
        SCHEDULE_PATH, parse_dates=["Date"], date_format="%d/%m/%Y %H:%M"
    )

    today = datetime.now(UTC).date()
    games = []

    for _, row in df.iterrows():
        game_date = row["Date"]
        if hasattr(game_date, "date"):
            if game_date.date() == today:
                games.append({
                    "home_team": row["Home Team"],
                    "away_team": row["Away Team"],
                    "tipoff_utc": game_date.replace(tzinfo=UTC),
                })

    return games


def schedule_trading_runs():
    """Schedule live_trader.run() 15 min before each game tip-off."""
    games = get_todays_games()

    if not games:
        print(f"[Scheduler] No games found for {datetime.now(UTC).date()}")
        return

    print(f"[Scheduler] Found {len(games)} games today:")

    now = datetime.now(UTC)
    scheduled_count = 0

    for game in games:
        run_time = game["tipoff_utc"] - timedelta(minutes=LEAD_TIME_MINUTES)
        et_time = game["tipoff_utc"].astimezone(ET)

        if run_time <= now:
            print(
                f"  {game['home_team']} vs {game['away_team']} "
                f"@ {et_time.strftime('%I:%M %p ET')} — ALREADY PASSED"
            )
            continue

        # Schedule at the specific UTC time
        run_time_str = run_time.strftime("%H:%M")
        schedule.every().day.at(run_time_str, "UTC").do(
            _run_trader, game["home_team"], game["away_team"]
        )
        scheduled_count += 1

        print(
            f"  {game['home_team']} vs {game['away_team']} "
            f"@ {et_time.strftime('%I:%M %p ET')} "
            f"— trader runs at {run_time.strftime('%H:%M UTC')}"
        )

    print(f"\n[Scheduler] Scheduled {scheduled_count} trading runs")

    # Also run once immediately if any games are within the next hour
    upcoming = [
        g for g in games
        if timedelta(0) < (g["tipoff_utc"] - now) < timedelta(hours=1)
    ]
    if upcoming:
        print(f"[Scheduler] {len(upcoming)} games within 1 hour — running trader now")
        _run_trader_now()


def _run_trader(home_team, away_team):
    """Execute the live trader (called by scheduler)."""
    print(f"\n[Scheduler] Triggering trader for {home_team} vs {away_team}")
    _run_trader_now()
    return schedule.CancelJob  # One-shot — don't repeat


def _run_trader_now():
    """Import and run the live trader."""
    from src.Bot.live_trader import run
    run(dry_run=False)


def main():
    """Entry point: schedule today's runs and loop."""
    print("=" * 60)
    print(f"  NBA Kalshi Scheduler — {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    schedule_trading_runs()

    if not schedule.get_jobs():
        print("[Scheduler] No jobs to run. Exiting.")
        return

    next_run = schedule.next_run()
    print(f"\n[Scheduler] Next run: {next_run}")
    print("[Scheduler] Waiting for scheduled jobs... (Ctrl+C to stop)\n")

    try:
        while schedule.get_jobs():
            schedule.run_pending()
            time.sleep(30)
    except KeyboardInterrupt:
        print("\n[Scheduler] Stopped.")


if __name__ == "__main__":
    main()
