from src.data.fetcher import CoinDCXFetcher
from src.data.indicators import TechnicalIndicators
from src.utils.logger import setup_logger
import yaml
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class SimpleEnhancedGenerator:
    """
    Simple enhanced signal generator that produces 2-3 high-quality signals
    with improved take profit consistency using existing data sources.
    """
    
    def __init__(self, balance: float = 10000):
        self.fetcher = CoinDCXFetcher()
        self.indicators = TechnicalIndicators()
        with open("config/config.yaml", "r") as file:
            self.config = yaml.safe_load(file)
        self.balance = balance
        self.logger = setup_logger()
        
        # Relaxed thresholds for current market conditions
        self.min_score_threshold = 6  # Reduced from 8
        self.min_profit_threshold = 200  # Reduced from 500
        self.max_signals = 3
        
    def generate_signals(self) -> List[Dict]:
        """Generate 2-3 high-quality signals."""
        self.logger.info("Starting simple enhanced signal generation")
        
        # Use existing instruments
        instruments = self.config["trading"]["instruments"][:5]  # Top 5 instruments
        self.logger.info(f"Analyzing {len(instruments)} instruments")
        
        all_signals = []
        
        for symbol in instruments:
            try:
                signal = self._generate_signal(symbol)
                if signal:
                    all_signals.append(signal)
                    self.logger.info(f"Generated signal for {symbol}: {signal['direction']}")
            except Exception as e:
                self.logger.error(f"Error generating signal for {symbol}: {e}")
                continue
        
        # Sort by quality score and return top 3
        all_signals.sort(key=lambda x: x['score'], reverse=True)
        top_signals = all_signals[:self.max_signals]
        
        self.logger.info(f"Signal generation complete: {len(top_signals)} signals")
        return top_signals
    
    def _generate_signal(self, symbol: str) -> Optional[Dict]:
        """Generate a single high-quality signal."""
        try:
            # Get market data
            market_data = self.fetcher.fetch_market_data(symbol)
            if not market_data or "last_price" not in market_data:
                return None
            
            current_price = market_data["last_price"]
            
            # Get candlestick data for 15m timeframe
            candles = self.fetcher.fetch_candlestick_data(symbol, "15m", limit=100)
            if not candles or len(candles) < 50:
                return None
            
            # Calculate indicators
            indicators = self._calculate_indicators(candles)
            if not indicators:
                return None
            
            # Determine signal direction
            signal_direction = self._determine_direction(indicators, current_price)
            if not signal_direction:
                return None
            
            # Calculate entry, stop loss, and take profit
            entry_price = current_price
            stop_loss, take_profit = self._calculate_exits(signal_direction, entry_price, indicators)
            
            # Calculate profit
            position_size = self.balance * 0.25
            leverage = 25
            usdt_inr = self.fetcher.fetch_usdt_inr_rate() or 93.0
            
            quantity = (position_size / usdt_inr) / entry_price * leverage
            
            if signal_direction == 'long':
                profit_usdt = (take_profit - entry_price) * quantity
            else:
                profit_usdt = (entry_price - take_profit) * quantity
            
            profit_inr = profit_usdt * usdt_inr
            taker_fee = profit_inr * self.config["trading"]["fees"]["taker"]
            net_profit = profit_inr - taker_fee
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(indicators, signal_direction)
            
            # Only return if meets minimum requirements
            if quality_score >= self.min_score_threshold and net_profit >= self.min_profit_threshold:
                return {
                    "symbol": symbol,
                    "timeframe": "15m",
                    "strategy": "enhanced",
                    "direction": f"TradeDirection.{signal_direction.upper()}",
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "tp1": take_profit,
                    "tp2": take_profit,
                    "max_hold_time": 6,  # 6 hours
                    "score": quality_score,
                    "score_max": 10,
                    "score_reasons": self._get_score_reasons(indicators, signal_direction),
                    "indicators": indicators,
                    "timestamp": datetime.now().isoformat(),
                    "estimated_profit_inr": net_profit,
                    "confidence": quality_score / 10.0,
                    "quality_score": quality_score * 10,
                    "normalized_score": quality_score * 10,
                    "side": signal_direction
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in signal generation for {symbol}: {e}")
            return None
    
    def _calculate_indicators(self, candles: List[Dict]) -> Optional[Dict]:
        """Calculate technical indicators."""
        try:
            rsi = self.indicators.calculate_rsi(candles)
            ema_20 = self.indicators.calculate_ema(candles, 20)
            ema_50 = self.indicators.calculate_ema(candles, 50)
            macd = self.indicators.calculate_macd(candles)
            bb_upper, bb_lower = self.indicators.calculate_bollinger_bands(candles)
            stoch_rsi = self.indicators.calculate_stoch_rsi(candles)
            atr = self.indicators.calculate_atr(candles)
            
            if any(x is None for x in [rsi, ema_20, ema_50, macd, bb_upper, bb_lower, stoch_rsi, atr]):
                return None
            
            return {
                "rsi": rsi,
                "ema_20": ema_20,
                "ema_50": ema_50,
                "macd": macd,
                "bollinger_upper": bb_upper,
                "bollinger_lower": bb_lower,
                "stoch_rsi": stoch_rsi,
                "atr": atr
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            return None
    
    def _determine_direction(self, indicators: Dict, current_price: float) -> Optional[str]:
        """Determine signal direction based on indicators."""
        try:
            # Long conditions
            long_score = 0
            if indicators['ema_20'] > indicators['ema_50']:
                long_score += 1
            if indicators['rsi'] > 40 and indicators['rsi'] < 70:
                long_score += 1
            if indicators['macd'] > 0:
                long_score += 1
            if current_price > indicators['ema_20']:
                long_score += 1
            if indicators['stoch_rsi']['%K'] > 20 and indicators['stoch_rsi']['%K'] < 80:
                long_score += 1
            
            # Short conditions
            short_score = 0
            if indicators['ema_20'] < indicators['ema_50']:
                short_score += 1
            if indicators['rsi'] < 60 and indicators['rsi'] > 30:
                short_score += 1
            if indicators['macd'] < 0:
                short_score += 1
            if current_price < indicators['ema_20']:
                short_score += 1
            if indicators['stoch_rsi']['%K'] > 20 and indicators['stoch_rsi']['%K'] < 80:
                short_score += 1
            
            # Need at least 3 out of 5 conditions
            if long_score >= 3 and long_score > short_score:
                return 'long'
            elif short_score >= 3 and short_score > long_score:
                return 'short'
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error determining direction: {e}")
            return None
    
    def _calculate_exits(self, direction: str, entry_price: float, indicators: Dict) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels."""
        try:
            atr = indicators['atr']
            if atr is None:
                atr = entry_price * 0.02
            
            # Improved risk/reward ratio (1:2)
            if direction == 'long':
                stop_loss = entry_price - (atr * 1.5)
                take_profit = entry_price + (atr * 3.0)
            else:
                stop_loss = entry_price + (atr * 1.5)
                take_profit = entry_price - (atr * 3.0)
            
            return stop_loss, take_profit
            
        except Exception as e:
            self.logger.error(f"Error calculating exits: {e}")
            # Fallback
            if direction == 'long':
                return entry_price * 0.97, entry_price * 1.06
            else:
                return entry_price * 1.03, entry_price * 0.94
    
    def _calculate_quality_score(self, indicators: Dict, direction: str) -> int:
        """Calculate quality score for signal."""
        try:
            score = 0
            
            # Trend alignment
            if direction == 'long' and indicators['ema_20'] > indicators['ema_50']:
                score += 2
            elif direction == 'short' and indicators['ema_20'] < indicators['ema_50']:
                score += 2
            
            # RSI in optimal range
            if 40 <= indicators['rsi'] <= 70:
                score += 2
            
            # MACD alignment
            if direction == 'long' and indicators['macd'] > 0:
                score += 2
            elif direction == 'short' and indicators['macd'] < 0:
                score += 2
            
            # StochRSI not extreme
            stoch_k = indicators['stoch_rsi']['%K']
            if 20 <= stoch_k <= 80:
                score += 2
            
            # ATR for volatility
            if indicators['atr'] > 0:
                score += 2
            
            return score
            
        except Exception as e:
            self.logger.error(f"Error calculating quality score: {e}")
            return 0
    
    def _get_score_reasons(self, indicators: Dict, direction: str) -> List[str]:
        """Get reasons for the quality score."""
        reasons = []
        
        if direction == 'long' and indicators['ema_20'] > indicators['ema_50']:
            reasons.append("EMA trend bullish")
        elif direction == 'short' and indicators['ema_20'] < indicators['ema_50']:
            reasons.append("EMA trend bearish")
        
        if 40 <= indicators['rsi'] <= 70:
            reasons.append("RSI in optimal range")
        
        if direction == 'long' and indicators['macd'] > 0:
            reasons.append("MACD bullish")
        elif direction == 'short' and indicators['macd'] < 0:
            reasons.append("MACD bearish")
        
        stoch_k = indicators['stoch_rsi']['%K']
        if 20 <= stoch_k <= 80:
            reasons.append("StochRSI not extreme")
        
        if indicators['atr'] > 0:
            reasons.append("Sufficient volatility")
        
        return reasons 