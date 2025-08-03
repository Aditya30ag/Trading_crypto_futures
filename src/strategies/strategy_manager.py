from src.strategies.scalping import ScalpingStrategy
from src.strategies.swing import SwingStrategy
from src.strategies.long_swing import LongSwingStrategy
from src.strategies.trend import TrendStrategy
from src.data.fetcher import CoinDCXFetcher
import yaml
import requests
from src.utils.logger import setup_logger

class StrategyManager:
    def __init__(self, config=None):
        self.fetcher = CoinDCXFetcher()
        if config is None:
            with open("config/config.yaml", "r") as file:
                self.config = yaml.safe_load(file)
        else:
            self.config = config
        self.balance = self.config["trading"]["initial_balance"]
        self.strategies = {
            "scalping": ScalpingStrategy(self.balance),
            "swing": SwingStrategy(self.balance),
            "long_swing": LongSwingStrategy(self.balance),
            "trend": TrendStrategy(self.balance)
        }
        self.logger = setup_logger()
        self.logger.info("StrategyManager initialized with all strategies")

    def get_top_instruments(self):
        """Select top 35 volatile/high-volume instruments."""
        try:
            self.logger.info("Fetching top instruments...")
            active_instruments = self.fetcher.fetch_active_instruments("USDT")
            if not active_instruments:
                self.logger.warning("No active instruments fetched, using default instruments")
                return self.config["trading"]["instruments"]
            
            self.logger.info(f"Found {len(active_instruments)} active instruments")
            
            # Fetch market data for all active instruments using public_base_url
            market_data = {}
            response = requests.get(f"{self.fetcher.public_base_url}/market_data/v3/current_prices/futures/rt", timeout=5)
            response.raise_for_status()
            data = response.json()
            for symbol in active_instruments:
                if symbol in data.get("prices", {}):
                    price_data = data["prices"][symbol]
                    market_data[symbol] = {
                        "volume": float(price_data.get("v", 0))
                    }
            # Sort by volume only - REMOVED 24hr change component
            sorted_instruments = sorted(
                market_data.keys(),
                key=lambda x: market_data[x]["volume"],
                reverse=True
            )
            top_instruments = sorted_instruments[:35]
            self.logger.info(f"Selected top {len(top_instruments)} instruments for analysis")
            return top_instruments
        except Exception as e:
            self.logger.error(f"Error fetching top instruments: {e}")
            self.logger.info("Using default instruments from config")
            return self.config["trading"]["instruments"]

    def generate_signals(self):
        """Generate signals for all instruments, timeframes, and strategies."""
        signals = []
        instruments = self.get_top_instruments()
        self.logger.info(f"Starting signal generation for {len(instruments)} instruments")
        
        total_attempts = 0
        for symbol in instruments:
            self.logger.debug(f"Analyzing {symbol}...")
            for strategy_name, strategy_obj in self.strategies.items():
                for timeframe in self.config["trading"]["timeframes"][strategy_name]:
                    total_attempts += 1
                    try:
                        signal = strategy_obj.generate_signal(symbol, timeframe)
                        if signal:
                            signals.append(signal)
                            self.logger.debug(f"Generated {strategy_name} signal for {symbol} {timeframe}")
                    except Exception as e:
                        self.logger.error(f"Error generating {strategy_name} signal for {symbol} {timeframe}: {e}")
        
        self.logger.info(f"Signal generation complete: {len(signals)} signals from {total_attempts} attempts")
        signals.sort(key=lambda x: x["estimated_profit_inr"], reverse=True)
        return signals