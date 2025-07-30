from src.strategies.scalping import ScalpingStrategy
from src.strategies.swing import SwingStrategy
from src.strategies.long_swing import LongSwingStrategy
from src.strategies.trend import TrendStrategy
from src.data.fetcher import CoinDCXFetcher
import yaml
import requests
from src.utils.logger import setup_logger
from datetime import datetime, timedelta
import pytz
import json
import os

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
        
        # Track recent signals to prevent symbol repetition
        self.recent_signals_file = "data/recent_signals.json"
        self.recent_signals = self._load_recent_signals()

    def _load_recent_signals(self):
        """Load recent signals from file to track symbol usage."""
        try:
            if os.path.exists(self.recent_signals_file):
                with open(self.recent_signals_file, 'r') as f:
                    signals = json.load(f)
                # Clean old signals (older than 4 hours)
                current_time = datetime.now(pytz.timezone('Asia/Kolkata'))
                cleaned_signals = {}
                for symbol, signal_data in signals.items():
                    signal_time = datetime.fromisoformat(signal_data['timestamp'].replace('Z', '+00:00'))
                    if signal_time.astimezone(pytz.timezone('Asia/Kolkata')) > current_time - timedelta(hours=4):
                        cleaned_signals[symbol] = signal_data
                return cleaned_signals
            return {}
        except Exception as e:
            self.logger.error(f"Error loading recent signals: {e}")
            return {}

    def _save_recent_signals(self):
        """Save recent signals to file."""
        try:
            os.makedirs(os.path.dirname(self.recent_signals_file), exist_ok=True)
            with open(self.recent_signals_file, 'w') as f:
                json.dump(self.recent_signals, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving recent signals: {e}")

    def _is_low_volatility_period(self):
        """Check if current time is during low volatility hours (12 AM - 6 AM IST)."""
        ist = pytz.timezone('Asia/Kolkata')
        current_time = datetime.now(ist)
        current_hour = current_time.hour
        
        # Low volatility period: 12 AM (0) to 6 AM (6)
        is_low_vol_time = 0 <= current_hour < 6
        
        if is_low_vol_time:
            self.logger.info(f"Current time {current_time.strftime('%H:%M IST')} is in low volatility period (12 AM - 6 AM IST)")
        
        return is_low_vol_time

    def _check_volatility_filter(self, symbol, strategy_name):
        """Check if symbol meets minimum volatility requirements."""
        try:
            # Fetch recent candlestick data for volatility analysis
            candles = self.fetcher.fetch_candlestick_data(symbol, "15m", limit=50)
            if not candles or len(candles) < 20:
                self.logger.debug(f"Insufficient data for volatility check: {symbol}")
                return False

            # Calculate ATR for volatility measurement
            from src.data.indicators import TechnicalIndicators
            indicators = TechnicalIndicators()
            atr = indicators.calculate_atr(candles, period=14)
            
            if atr is None:
                self.logger.debug(f"ATR calculation failed for {symbol}")
                return False

            # Get current price for ATR percentage calculation
            current_price = float(candles[-1]["close"])
            atr_percentage = (atr / current_price) * 100

            # Calculate Bollinger Band width as secondary volatility measure
            bb_upper, bb_lower = indicators.calculate_bollinger_bands(candles, period=20, std_dev=2)
            if bb_upper and bb_lower:
                bb_width_percentage = ((bb_upper - bb_lower) / current_price) * 100
            else:
                bb_width_percentage = 0

            # Set minimum volatility thresholds based on strategy
            if strategy_name == "scalping":
                min_atr_pct = 0.8  # Minimum 0.8% ATR for scalping
                min_bb_width_pct = 2.0  # Minimum 2% BB width for scalping
            elif strategy_name in ["swing", "long_swing"]:
                min_atr_pct = 1.2  # Minimum 1.2% ATR for swing trades
                min_bb_width_pct = 3.0  # Minimum 3% BB width for swing trades
            else:
                min_atr_pct = 1.0  # Default minimum ATR
                min_bb_width_pct = 2.5  # Default minimum BB width

            # Check if volatility meets requirements
            volatility_ok = atr_percentage >= min_atr_pct and bb_width_percentage >= min_bb_width_pct

            if not volatility_ok:
                self.logger.debug(f"Volatility filter failed for {symbol} ({strategy_name}): "
                                f"ATR={atr_percentage:.2f}% (min {min_atr_pct}%), "
                                f"BB_width={bb_width_percentage:.2f}% (min {min_bb_width_pct}%)")
            else:
                self.logger.debug(f"Volatility filter passed for {symbol} ({strategy_name}): "
                                f"ATR={atr_percentage:.2f}%, BB_width={bb_width_percentage:.2f}%")

            return volatility_ok

        except Exception as e:
            self.logger.error(f"Error checking volatility filter for {symbol}: {e}")
            return False

    def _should_skip_symbol(self, symbol, strategy_name):
        """Check if symbol should be skipped due to recent usage."""
        try:
            if symbol not in self.recent_signals:
                return False

            recent_signal = self.recent_signals[symbol]
            signal_time = datetime.fromisoformat(recent_signal['timestamp'].replace('Z', '+00:00'))
            current_time = datetime.now(pytz.timezone('Asia/Kolkata'))
            
            # Check if signal is within 4-hour window
            time_diff = current_time - signal_time.astimezone(pytz.timezone('Asia/Kolkata'))
            if time_diff < timedelta(hours=4):
                # Check if major indicator values have changed significantly
                if self._has_major_indicator_change(symbol, recent_signal):
                    self.logger.info(f"Major indicator change detected for {symbol}, allowing new signal")
                    return False
                else:
                    self.logger.debug(f"Skipping {symbol} - used {time_diff.total_seconds()/3600:.1f} hours ago")
                    return True
            
            return False

        except Exception as e:
            self.logger.error(f"Error checking symbol skip for {symbol}: {e}")
            return False

    def _has_major_indicator_change(self, symbol, recent_signal):
        """Check if major indicators have changed significantly since last signal."""
        try:
            # Get current market data
            candles = self.fetcher.fetch_candlestick_data(symbol, "15m", limit=50)
            if not candles or len(candles) < 20:
                return False

            from src.data.indicators import TechnicalIndicators
            indicators = TechnicalIndicators()
            
            # Calculate current indicators
            current_rsi = indicators.calculate_rsi(candles)
            current_macd = indicators.calculate_macd(candles)
            current_ema_20 = indicators.calculate_ema(candles, 20)
            current_ema_50 = indicators.calculate_ema(candles, 50)

            if any(x is None for x in [current_rsi, current_macd, current_ema_20, current_ema_50]):
                return False

            # Get previous indicator values
            prev_indicators = recent_signal.get('indicators', {})
            prev_rsi = prev_indicators.get('rsi', current_rsi)
            prev_macd = prev_indicators.get('macd', current_macd)
            prev_ema_20 = prev_indicators.get('ema_20', current_ema_20)
            prev_ema_50 = prev_indicators.get('ema_50', current_ema_50)

            # Check for significant changes (thresholds for major changes)
            rsi_change = abs(current_rsi - prev_rsi) > 15  # RSI change > 15 points
            macd_change = abs(current_macd - prev_macd) / abs(prev_macd) > 0.3 if prev_macd != 0 else False  # MACD change > 30%
            ema_20_change = abs(current_ema_20 - prev_ema_20) / prev_ema_20 > 0.02 if prev_ema_20 != 0 else False  # EMA20 change > 2%
            ema_50_change = abs(current_ema_50 - prev_ema_50) / prev_ema_50 > 0.02 if prev_ema_50 != 0 else False  # EMA50 change > 2%

            # Major change if any 2 or more indicators changed significantly
            major_changes = sum([rsi_change, macd_change, ema_20_change, ema_50_change])
            has_major_change = major_changes >= 2

            if has_major_change:
                self.logger.debug(f"Major indicator changes for {symbol}: RSI={rsi_change}, MACD={macd_change}, "
                                f"EMA20={ema_20_change}, EMA50={ema_50_change}")

            return has_major_change

        except Exception as e:
            self.logger.error(f"Error checking indicator changes for {symbol}: {e}")
            return False

    def _record_signal(self, signal):
        """Record signal to prevent symbol repetition."""
        try:
            symbol = signal['symbol']
            self.recent_signals[symbol] = {
                'timestamp': signal['timestamp'],
                'strategy': signal['strategy'],
                'score': signal['score'],
                'indicators': signal.get('indicators', {})
            }
            self._save_recent_signals()
        except Exception as e:
            self.logger.error(f"Error recording signal: {e}")

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
                        "volume": float(price_data.get("v", 0)),
                        "change_24h": float(price_data.get("pc", 0))
                    }
            # Sort by volatility * volume
            sorted_instruments = sorted(
                market_data.keys(),
                key=lambda x: market_data[x]["volume"] * abs(market_data[x]["change_24h"]),
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
        """Generate signals for all instruments, timeframes, and strategies with enhanced filters."""
        signals = []
        instruments = self.get_top_instruments()
        self.logger.info(f"Starting signal generation for {len(instruments)} instruments")
        
        # Check if it's low volatility time period
        is_low_vol_time = self._is_low_volatility_period()
        
        total_attempts = 0
        filtered_out = {
            'volatility': 0,
            'time_based': 0,
            'symbol_repetition': 0,
            'low_score': 0
        }
        
        for symbol in instruments:
            self.logger.debug(f"Analyzing {symbol}...")
            
            # Check if symbol should be skipped due to recent usage
            for strategy_name, strategy_obj in self.strategies.items():
                if self._should_skip_symbol(symbol, strategy_name):
                    filtered_out['symbol_repetition'] += len(self.config["trading"]["timeframes"][strategy_name])
                    continue
                
                # Check volatility filter for all strategies
                if not self._check_volatility_filter(symbol, strategy_name):
                    filtered_out['volatility'] += len(self.config["trading"]["timeframes"][strategy_name])
                    continue
                
                for timeframe in self.config["trading"]["timeframes"][strategy_name]:
                    total_attempts += 1
                    try:
                        signal = strategy_obj.generate_signal(symbol, timeframe)
                        if signal:
                            # Apply minimum score filter (≥ 10)
                            if signal.get('score', 0) < 10:
                                filtered_out['low_score'] += 1
                                self.logger.debug(f"Filtered out {strategy_name} signal for {symbol} {timeframe}: "
                                                f"score {signal.get('score', 0)} < 10")
                                continue
                            
                            # Apply time-based filter for swing and long_swing strategies
                            if strategy_name in ['swing', 'long_swing'] and is_low_vol_time:
                                # Only allow high-scoring signals during low volatility hours
                                if signal.get('score', 0) < 12:
                                    filtered_out['time_based'] += 1
                                    self.logger.debug(f"Filtered out {strategy_name} signal for {symbol} {timeframe} "
                                                    f"during low volatility hours: score {signal.get('score', 0)} < 12")
                                    continue
                                else:
                                    self.logger.info(f"Allowing high-score {strategy_name} signal during low volatility hours: "
                                                   f"{symbol} {timeframe} (score: {signal.get('score', 0)})")
                            
                            # Update max_hold_time based on strategy
                            if strategy_name == "scalping":
                                signal["max_hold_time"] = 0.5  # 30 minutes
                            elif strategy_name == "swing":
                                signal["max_hold_time"] = 4  # 4 hours
                            elif strategy_name == "long_swing":
                                signal["max_hold_time"] = 24  # 24 hours (keep existing)
                            
                            signals.append(signal)
                            self._record_signal(signal)  # Record signal to prevent repetition
                            self.logger.debug(f"Generated {strategy_name} signal for {symbol} {timeframe} "
                                            f"(score: {signal.get('score', 0)})")
                    except Exception as e:
                        self.logger.error(f"Error generating {strategy_name} signal for {symbol} {timeframe}: {e}")
        
        # Log filtering statistics
        self.logger.info(f"Signal generation complete: {len(signals)} signals from {total_attempts} attempts")
        self.logger.info(f"Filtered out - Volatility: {filtered_out['volatility']}, "
                        f"Time-based: {filtered_out['time_based']}, "
                        f"Symbol repetition: {filtered_out['symbol_repetition']}, "
                        f"Low score: {filtered_out['low_score']}")
        
        signals.sort(key=lambda x: x["estimated_profit_inr"], reverse=True)
        return signals