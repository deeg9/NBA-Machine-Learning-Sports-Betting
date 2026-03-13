"""Kalshi API wrapper for NBA game markets.

Uses the kalshi_python_sync SDK with RSA authentication to fetch markets,
orderbooks, and place limit orders on NBA moneyline contracts.
"""

import os
import uuid
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

# Kalshi display name → model team name mapping
KALSHI_TEAM_MAP = {
    "Hawks": "Atlanta Hawks",
    "Atlanta Hawks": "Atlanta Hawks",
    "Celtics": "Boston Celtics",
    "Boston Celtics": "Boston Celtics",
    "Nets": "Brooklyn Nets",
    "Brooklyn Nets": "Brooklyn Nets",
    "Hornets": "Charlotte Hornets",
    "Charlotte Hornets": "Charlotte Hornets",
    "Bulls": "Chicago Bulls",
    "Chicago Bulls": "Chicago Bulls",
    "Cavaliers": "Cleveland Cavaliers",
    "Cleveland Cavaliers": "Cleveland Cavaliers",
    "Mavericks": "Dallas Mavericks",
    "Dallas Mavericks": "Dallas Mavericks",
    "Nuggets": "Denver Nuggets",
    "Denver Nuggets": "Denver Nuggets",
    "Pistons": "Detroit Pistons",
    "Detroit Pistons": "Detroit Pistons",
    "Warriors": "Golden State Warriors",
    "Golden State Warriors": "Golden State Warriors",
    "Rockets": "Houston Rockets",
    "Houston Rockets": "Houston Rockets",
    "Pacers": "Indiana Pacers",
    "Indiana Pacers": "Indiana Pacers",
    "Clippers": "LA Clippers",
    "LA Clippers": "LA Clippers",
    "Los Angeles Clippers": "LA Clippers",
    "Lakers": "Los Angeles Lakers",
    "LA Lakers": "Los Angeles Lakers",
    "Los Angeles Lakers": "Los Angeles Lakers",
    "Grizzlies": "Memphis Grizzlies",
    "Memphis Grizzlies": "Memphis Grizzlies",
    "Heat": "Miami Heat",
    "Miami Heat": "Miami Heat",
    "Bucks": "Milwaukee Bucks",
    "Milwaukee Bucks": "Milwaukee Bucks",
    "Timberwolves": "Minnesota Timberwolves",
    "Minnesota Timberwolves": "Minnesota Timberwolves",
    "Pelicans": "New Orleans Pelicans",
    "New Orleans Pelicans": "New Orleans Pelicans",
    "Knicks": "New York Knicks",
    "New York Knicks": "New York Knicks",
    "Thunder": "Oklahoma City Thunder",
    "Oklahoma City Thunder": "Oklahoma City Thunder",
    "Magic": "Orlando Magic",
    "Orlando Magic": "Orlando Magic",
    "76ers": "Philadelphia 76ers",
    "Philadelphia 76ers": "Philadelphia 76ers",
    "Suns": "Phoenix Suns",
    "Phoenix Suns": "Phoenix Suns",
    "Trail Blazers": "Portland Trail Blazers",
    "Portland Trail Blazers": "Portland Trail Blazers",
    "Kings": "Sacramento Kings",
    "Sacramento Kings": "Sacramento Kings",
    "Spurs": "San Antonio Spurs",
    "San Antonio Spurs": "San Antonio Spurs",
    "Raptors": "Toronto Raptors",
    "Toronto Raptors": "Toronto Raptors",
    "Jazz": "Utah Jazz",
    "Utah Jazz": "Utah Jazz",
    "Wizards": "Washington Wizards",
    "Washington Wizards": "Washington Wizards",
}


class KalshiClient:
    """Wrapper around kalshi_python_sync SDK for NBA markets."""

    def __init__(self, api_key=None, private_key_path=None, env=None):
        self.api_key = api_key or os.getenv("KALSHI_API_KEY", "")
        self.private_key_path = private_key_path or os.getenv(
            "KALSHI_PRIVATE_KEY_PATH", "./kalshi_private_key.pem"
        )
        self.env = env or os.getenv("KALSHI_ENV", "demo")

        if self.env == "prod":
            self.base_url = "https://api.elections.kalshi.com"
        else:
            self.base_url = "https://demo-api.kalshi.co"

        self._client = None
        self._init_client()

    def _init_client(self):
        """Initialize the Kalshi SDK client with RSA key authentication."""
        try:
            import kalshi_python_sync as kalshi

            config = kalshi.Configuration()
            config.host = self.base_url + "/trade-api/v2"

            # Read private key
            with open(self.private_key_path, "r") as f:
                private_key = f.read()

            self._api_instance = kalshi.AuthApi(kalshi.ApiClient(config))
            login_response = self._api_instance.login(
                kalshi.LoginRequest(
                    email=self.api_key,
                    password="",  # RSA auth uses key, not password
                )
            )
            config.api_key["Authorization"] = login_response.token
            config.api_key_prefix["Authorization"] = "Bearer"

            api_client = kalshi.ApiClient(config)
            self._market_api = kalshi.MarketApi(api_client)
            self._portfolio_api = kalshi.PortfolioApi(api_client)
            print(f"[KalshiClient] Connected to {self.env} environment")
        except ImportError:
            print("[KalshiClient] kalshi_python_sync not installed. "
                  "Install with: pip install kalshi_python_sync")
            self._market_api = None
            self._portfolio_api = None
        except FileNotFoundError:
            print(f"[KalshiClient] Private key not found at {self.private_key_path}")
            self._market_api = None
            self._portfolio_api = None
        except Exception as e:
            print(f"[KalshiClient] Auth failed: {e}")
            self._market_api = None
            self._portfolio_api = None

    def get_nba_game_markets(self, date=None):
        """Fetch NBA game markets for a given date.

        Parameters
        ----------
        date : str or None
            Date in YYYY-MM-DD format. Defaults to today.

        Returns
        -------
        list[dict]
            Each dict has: ticker, title, home_team, away_team,
            yes_price, no_price, close_time
        """
        if self._market_api is None:
            print("[KalshiClient] Not connected — returning empty markets")
            return []

        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        try:
            # Search for NBA markets — Kalshi uses event tickers like KXNBA-*
            response = self._market_api.get_markets(
                status="open",
                series_ticker="KXNBA",
                limit=200,
            )
            markets = []
            for market in response.markets:
                # Parse team names from market title
                home_team, away_team = self._parse_teams(market.title)
                if home_team and away_team:
                    markets.append({
                        "ticker": market.ticker,
                        "title": market.title,
                        "home_team": home_team,
                        "away_team": away_team,
                        "yes_price": market.yes_ask if hasattr(market, "yes_ask") else market.last_price,
                        "no_price": market.no_ask if hasattr(market, "no_ask") else (100 - market.last_price),
                        "close_time": market.close_time,
                    })
            return markets
        except Exception as e:
            print(f"[KalshiClient] Error fetching markets: {e}")
            return []

    def get_orderbook(self, ticker, depth=1):
        """Get best bid/ask for a market.

        Returns
        -------
        dict with yes_bid, yes_ask, no_bid, no_ask (cents)
        """
        if self._market_api is None:
            return {}
        try:
            book = self._market_api.get_market_orderbook(
                ticker=ticker, depth=depth
            )
            return {
                "yes_bid": book.orderbook.yes[0][0] if book.orderbook.yes else None,
                "yes_ask": book.orderbook.no[0][0] if book.orderbook.no else None,
                "no_bid": book.orderbook.no[0][0] if book.orderbook.no else None,
                "no_ask": book.orderbook.yes[0][0] if book.orderbook.yes else None,
            }
        except Exception as e:
            print(f"[KalshiClient] Orderbook error for {ticker}: {e}")
            return {}

    def place_order(self, ticker, side, count, price_cents):
        """Place a limit order on Kalshi.

        Parameters
        ----------
        ticker : str
            Market ticker.
        side : str
            'yes' or 'no'.
        count : int
            Number of contracts.
        price_cents : int
            Limit price in cents (1-99).

        Returns
        -------
        dict with order_id and status, or None on failure.
        """
        if self._portfolio_api is None:
            print("[KalshiClient] Not connected — cannot place order")
            return None

        client_order_id = str(uuid.uuid4())

        try:
            import kalshi_python_sync as kalshi

            order_request = kalshi.CreateOrderRequest(
                ticker=ticker,
                client_order_id=client_order_id,
                side=side,
                action="buy",
                count=count,
                type="limit",
                yes_price=price_cents if side == "yes" else None,
                no_price=price_cents if side == "no" else None,
            )
            response = self._portfolio_api.create_order(order_request)
            return {
                "order_id": response.order.order_id,
                "client_order_id": client_order_id,
                "status": response.order.status,
                "ticker": ticker,
                "side": side,
                "count": count,
                "price_cents": price_cents,
            }
        except Exception as e:
            print(f"[KalshiClient] Order failed: {e}")
            return None

    def get_positions(self):
        """Get current open positions."""
        if self._portfolio_api is None:
            return []
        try:
            response = self._portfolio_api.get_positions(
                settlement_status="unsettled"
            )
            return response.market_positions
        except Exception as e:
            print(f"[KalshiClient] Positions error: {e}")
            return []

    def get_balance(self):
        """Get available account balance in cents."""
        if self._portfolio_api is None:
            return 0
        try:
            response = self._portfolio_api.get_balance()
            return response.balance
        except Exception as e:
            print(f"[KalshiClient] Balance error: {e}")
            return 0

    def _parse_teams(self, title):
        """Parse home and away team names from a Kalshi market title.

        Kalshi titles typically look like:
        'Will the Lakers beat the Celtics?' or 'Lakers vs Celtics'

        Returns (home_team, away_team) using our model's naming, or (None, None).
        """
        if not title:
            return None, None

        title_lower = title.lower()

        # Try 'Team1 vs Team2' pattern
        for sep in [" vs ", " vs. ", " v ", " @ "]:
            if sep in title_lower:
                parts = title.split(sep if sep in title else sep.strip())
                if len(parts) == 2:
                    away_name = self._resolve_team(parts[0].strip())
                    home_name = self._resolve_team(parts[1].strip())
                    if away_name and home_name:
                        return home_name, away_name

        # Try 'Will the X beat the Y?' pattern
        if "beat" in title_lower:
            import re
            match = re.search(
                r"will\s+(?:the\s+)?(.+?)\s+beat\s+(?:the\s+)?(.+?)[\?\.]",
                title,
                re.IGNORECASE,
            )
            if match:
                team1 = self._resolve_team(match.group(1).strip())
                team2 = self._resolve_team(match.group(2).strip())
                if team1 and team2:
                    # Team1 beating Team2 → Team1 is the subject (usually home)
                    return team1, team2

        return None, None

    def _resolve_team(self, name):
        """Resolve a partial or full team name to our model's team name."""
        # Direct lookup
        if name in KALSHI_TEAM_MAP:
            return KALSHI_TEAM_MAP[name]

        # Try matching against known names (case-insensitive)
        name_lower = name.lower()
        for key, value in KALSHI_TEAM_MAP.items():
            if key.lower() in name_lower or name_lower in key.lower():
                return value

        return None
