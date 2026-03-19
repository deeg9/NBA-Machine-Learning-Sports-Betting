"""Daily scheduler for the NBA Kalshi trading bot.

Runs the model at 6:30 PM ET daily (30 min before most tip-offs).
Generates predictions in dry-run mode and saves to logs/scan_YYYY-MM-DD.json
for the dashboard to display.

Usage:
    python -m src.Bot.scheduler
"""

import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import schedule

ET = ZoneInfo("America/New_York")
UTC = timezone.utc

DAILY_RUN_TIME = "18:30"  # 6:30 PM ET


def run_daily_scan():
    """Run the full prediction scan and save results."""
    print(f"\n[Scheduler] Starting daily scan at {datetime.now(ET).strftime('%I:%M %p ET')}")
    try:
        from src.Bot.live_trader import scan_games
        predictions = scan_games()
        print(f"[Scheduler] Scan complete: {len(predictions)} games analyzed")
        trades = [p for p in predictions if p.get("side")]
        if trades:
            print(f"[Scheduler] {len(trades)} trades found:")
            for t in trades:
                print(f"  {t['away_team']} @ {t['home_team']}: "
                      f"BUY {t['side'].upper()} (edge={t['edge']:+.1%})")
        else:
            print("[Scheduler] No trades met the edge threshold")
    except Exception as e:
        print(f"[Scheduler] Scan failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Entry point: schedule daily scan at 6:30 PM ET."""
    print("=" * 60)
    print(f"  NBA Kalshi Scheduler — {datetime.now(ET).strftime('%Y-%m-%d %I:%M %p ET')}")
    print(f"  Daily scan time: {DAILY_RUN_TIME} ET")
    print("=" * 60)

    # Schedule daily run
    schedule.every().day.at(DAILY_RUN_TIME, "America/New_York").do(run_daily_scan)

    next_run = schedule.next_run()
    print(f"\n[Scheduler] Next run: {next_run}")
    print("[Scheduler] Waiting for scheduled jobs... (Ctrl+C to stop)\n")

    try:
        while True:
            schedule.run_pending()
            time.sleep(30)
    except KeyboardInterrupt:
        print("\n[Scheduler] Stopped.")


if __name__ == "__main__":
    main()
