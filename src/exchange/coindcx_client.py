import requests
import yaml
import time
from datetime import datetime, timezone, timedelta
from src.utils.logger import setup_logger

class CoinDCXClient:
    def __init__(self):
        with open("config/config.yaml", "r") as file:
            self.config = yaml.safe_load(file)
        self.public_base_url = "https://public.coindcx.com"  # Public endpoints
        self.api_base_url = "https://api.coindcx.com"       # Authenticated endpoints
        self.logger = setup_logger()

    def get_ticker(self, symbol, retries=3):
        """Get current market price for a futures symbol with retries."""
        for attempt in range(retries):
            try:
                response = requests.get(f"{self.public_base_url}/market_data/v3/current_prices/futures/rt", timeout=5)
                response.raise_for_status()
                data = response.json()
                if symbol in data.get("prices", {}):
                    price_data = data["prices"][symbol]
                    return float(price_data.get("ls", 0))
                self.logger.error(f"Symbol {symbol} not found in market data")
                return None
            except Exception as e:
                self.logger.error(f"Attempt {attempt + 1}/{retries} - Error fetching ticker for {symbol}: {e}")
                if attempt < retries - 1:
                    time.sleep(1)
                continue
        self.logger.error(f"Failed to fetch ticker for {symbol} after {retries} attempts")
        return None

    def get_candles(self, symbol, timeframe, limit=100, retries=3):
        """Get historical candlestick data with retries."""
        for attempt in range(retries):
            try:
                # Map timeframe to resolution (e.g., '5m' to '5', '1h' to '60', '1D' to '1D')
                time_map = {"5m": "5", "15m": "15", "1h": "60", "1D": "1D"}
                resolution = time_map.get(timeframe, timeframe)
                if resolution not in ["1", "5", "15", "60", "1D"]:
                    raise ValueError(f"Invalid resolution: {resolution} for {symbol}")
                # Calculate from and to timestamps in seconds (EPOCH)
                to_time = int(time.time())
                from_time = to_time - (limit * int(resolution if resolution.isdigit() else 1440))  # 1440 min = 1 day for '1D'
                response = requests.get(
                    f"{self.public_base_url}/market_data/candlesticks",
                    params={
                        "pair": symbol,
                        "from": from_time,
                        "to": to_time,
                        "resolution": resolution,
                        "pcode": "f"
                    },
                    timeout=5
                )
                response.raise_for_status()
                data = response.json()
                if data.get("s") == "ok" and data.get("data"):
                    return data["data"]
                self.logger.error(f"No candlestick data for {symbol} in response: {data}")
                return []
            except Exception as e:
                self.logger.error(f"Attempt {attempt + 1}/{retries} - Error fetching candles for {symbol}: {e}")
                if attempt < retries - 1:
                    time.sleep(1)
                continue
        self.logger.error(f"Failed to fetch candlestick data for {symbol} after {retries} attempts")
        return []

    def get_trade_history(self, symbol, retries=3):
        """Get real-time trade history for a futures symbol with retries."""
        for attempt in range(retries):
            try:
                response = requests.get(
                    f"{self.api_base_url}/exchange/v1/derivatives/futures/data/trades?pair={symbol}",
                    timeout=5
                )
                response.raise_for_status()
                data = response.json()
                if data and isinstance(data, list):
                    return data
                self.logger.error(f"No trade history data for {symbol} in response: {data}")
                return []
            except Exception as e:
                self.logger.error(f"Attempt {attempt + 1}/{retries} - Error fetching trade history for {symbol}: {e}")
                if attempt < retries - 1:
                    time.sleep(1)
                continue
        self.logger.error(f"Failed to fetch trade history for {symbol} after {retries} attempts")
        return []

    def get_ist_timestamp(self):
        """Get current timestamp in IST."""
        ist = timezone(timedelta(hours=5, minutes=30))
        return datetime.now(ist).isoformat()