from src.data.fetcher import CoinDCXFetcher
from src.data.indicators import TechnicalIndicators
from src.utils.logger import setup_logger
import yaml
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class EnhancedSignalGenerator:
    """
    Enhanced signal generator focused on producing 2-3 highly confirmed signals
    with better take profit consistency through:
    1. Stricter multi-timeframe trend confirmation
    2. Market structure analysis
    3. Improved risk/reward ratios
    4. Volume and liquidity validation
    5. Support/resistance confluence
    """
    
    def __init__(self, balance: float = 10000):
        self.fetcher = CoinDCXFetcher()
        self.indicators = TechnicalIndicators()
        with open("config/config.yaml", "r") as file:
            self.config = yaml.safe_load(file)
        self.balance = balance
        self.logger = setup_logger()
        
        # Enhanced thresholds for higher quality signals
        self.min_score_threshold = 8  # Increased from 6
        self.min_confidence_threshold = 0.7  # Increased from 0.5
        self.min_profit_threshold = 500  # Increased from 100
        self.max_signals = 3  # Focus on top 3 signals only
        
    def generate_enhanced_signals(self) -> List[Dict]:
        """Generate 2-3 highly confirmed signals with strict filtering."""
        self.logger.info("Starting enhanced signal generation for high-quality signals only")
        
        # Get top instruments with high liquidity
        instruments = self._get_high_liquidity_instruments()
        self.logger.info(f"Analyzing {len(instruments)} high-liquidity instruments")
        
        all_signals = []
        
        for symbol in instruments:
            try:
                # Multi-timeframe analysis with strict confirmation
                signal = self._generate_enhanced_signal(symbol)
                if signal:
                    all_signals.append(signal)
                    self.logger.info(f"Generated enhanced signal for {symbol}: {signal['direction']} score={signal['score']}")
            except Exception as e:
                self.logger.error(f"Error generating enhanced signal for {symbol}: {e}")
                continue
        
        # Sort by quality score and return top 3
        all_signals.sort(key=lambda x: x['quality_score'], reverse=True)
        top_signals = all_signals[:self.max_signals]
        
        self.logger.info(f"Enhanced signal generation complete: {len(top_signals)} high-quality signals")
        return top_signals
    
    def _get_high_liquidity_instruments(self) -> List[str]:
        """Get instruments with high liquidity and volume."""
        try:
            # Use existing instruments from config
            instruments = self.config["trading"]["instruments"]
            
            # Filter by checking market data for each instrument - FOCUS ON HIGH VOLUME ONLY
            high_liquidity = []
            for symbol in instruments:
                try:
                    market_data = self.fetcher.fetch_market_data(symbol)
                    if market_data and 'volume' in market_data:
                        volume = float(market_data.get('volume', 0))
                        
                        # High volume criteria - REMOVED 24hr change requirement
                        if volume > 500000:  # Increased volume threshold for highly active symbols
                            high_liquidity.append(symbol)
                except Exception as e:
                    self.logger.debug(f"Error checking liquidity for {symbol}: {e}")
                    continue
            
            return high_liquidity[:10] if high_liquidity else instruments[:5]  # Top 10 or first 5
        except Exception as e:
            self.logger.error(f"Error fetching high liquidity instruments: {e}")
            return self.config["trading"]["instruments"][:5]  # Return first 5 instruments
    
    def _generate_enhanced_signal(self, symbol: str) -> Optional[Dict]:
        """Generate a single enhanced signal with strict filtering."""
        try:
            # Multi-timeframe data collection
            timeframes = ['5m', '15m', '1h']
            multi_tf_data = {}
            
            for tf in timeframes:
                candles = self.fetcher.fetch_candlestick_data(symbol, tf, limit=200)
                if candles and len(candles) >= 50:
                    multi_tf_data[tf] = candles
            
            if len(multi_tf_data) < 2:  # Need at least 2 timeframes
                return None
            
            # Market structure analysis
            structure_analysis = self._analyze_market_structure(multi_tf_data)
            if not structure_analysis['valid_setup']:
                return None
            
            # Enhanced trend confirmation
            trend_analysis = self._analyze_enhanced_trend(multi_tf_data)
            if trend_analysis['strength'] < 0.7:  # Strong trend required
                return None
            
            # Support/Resistance confluence
            sr_analysis = self._analyze_support_resistance(multi_tf_data)
            
            # Volume and liquidity validation
            volume_analysis = self._analyze_volume_profile(multi_tf_data)
            if not volume_analysis['high_volume']:
                return None
            
            # Generate signal based on confluence
            signal = self._create_enhanced_signal(
                symbol, multi_tf_data, structure_analysis, 
                trend_analysis, sr_analysis, volume_analysis
            )
            
            return signal if signal and signal['score'] >= self.min_score_threshold else None
            
        except Exception as e:
            self.logger.error(f"Error in enhanced signal generation for {symbol}: {e}")
            return None
    
    def _analyze_market_structure(self, multi_tf_data: Dict) -> Dict:
        """Analyze market structure for valid setups."""
        try:
            # Use 1h timeframe for structure analysis
            candles_1h = multi_tf_data.get('1h', [])
            if len(candles_1h) < 50:
                return {'valid_setup': False}
            
            # Calculate higher highs and lower lows
            highs = [float(c['high']) for c in candles_1h[-20:]]
            lows = [float(c['low']) for c in candles_1h[-20:]]
            
            # Check for trend structure
            recent_highs = highs[-5:]
            recent_lows = lows[-5:]
            
            # Uptrend: Higher highs and higher lows
            uptrend = (max(recent_highs) > max(highs[:-5]) and 
                      min(recent_lows) > min(lows[:-5]))
            
            # Downtrend: Lower highs and lower lows
            downtrend = (max(recent_highs) < max(highs[:-5]) and 
                        min(recent_lows) < min(lows[:-5]))
            
            # Consolidation: No clear trend
            consolidation = not (uptrend or downtrend)
            
            return {
                'valid_setup': True,
                'uptrend': uptrend,
                'downtrend': downtrend,
                'consolidation': consolidation,
                'structure_type': 'uptrend' if uptrend else 'downtrend' if downtrend else 'consolidation'
            }
            
        except Exception as e:
            self.logger.error(f"Error in market structure analysis: {e}")
            return {'valid_setup': False}
    
    def _analyze_enhanced_trend(self, multi_tf_data: Dict) -> Dict:
        """Enhanced trend analysis across multiple timeframes."""
        try:
            trend_scores = {}
            total_strength = 0
            
            for tf, candles in multi_tf_data.items():
                if len(candles) < 50:
                    continue
                
                # Calculate trend indicators
                ema_20 = self.indicators.calculate_ema(candles, 20)
                ema_50 = self.indicators.calculate_ema(candles, 50)
                adx = self.indicators.calculate_adx(candles)
                macd = self.indicators.calculate_macd(candles)
                
                if any(x is None for x in [ema_20, ema_50, adx, macd]):
                    continue
                
                # Trend scoring
                trend_score = 0
                if ema_20 > ema_50:
                    trend_score += 1
                if adx['adx'] > 25:
                    trend_score += 1
                if macd > 0:
                    trend_score += 1
                
                trend_scores[tf] = trend_score
                total_strength += trend_score
            
            # Normalize strength
            max_possible = len(multi_tf_data) * 3
            normalized_strength = total_strength / max_possible if max_possible > 0 else 0
            
            return {
                'strength': normalized_strength,
                'trend_scores': trend_scores,
                'timeframe_count': len(multi_tf_data)
            }
            
        except Exception as e:
            self.logger.error(f"Error in enhanced trend analysis: {e}")
            return {'strength': 0}
    
    def _analyze_support_resistance(self, multi_tf_data: Dict) -> Dict:
        """Analyze support and resistance levels."""
        try:
            # Use 1h timeframe for S/R analysis
            candles_1h = multi_tf_data.get('1h', [])
            if len(candles_1h) < 30:
                return {'confluence': False}
            
            highs = [float(c['high']) for c in candles_1h]
            lows = [float(c['low']) for c in candles_1h]
            current_price = float(candles_1h[-1]['close'])
            
            # Find key levels
            resistance_levels = self._find_key_levels(highs, 'resistance')
            support_levels = self._find_key_levels(lows, 'support')
            
            # Check confluence with current price
            price_tolerance = current_price * 0.02  # 2% tolerance
            
            near_resistance = any(abs(level - current_price) < price_tolerance 
                                for level in resistance_levels)
            near_support = any(abs(level - current_price) < price_tolerance 
                             for level in support_levels)
            
            return {
                'confluence': near_resistance or near_support,
                'near_resistance': near_resistance,
                'near_support': near_support,
                'resistance_levels': resistance_levels,
                'support_levels': support_levels
            }
            
        except Exception as e:
            self.logger.error(f"Error in support/resistance analysis: {e}")
            return {'confluence': False}
    
    def _find_key_levels(self, prices: List[float], level_type: str) -> List[float]:
        """Find key support/resistance levels using pivot points."""
        try:
            if len(prices) < 20:
                return []
            
            # Use recent prices for level detection
            recent_prices = prices[-20:]
            
            if level_type == 'resistance':
                # Find local maxima
                levels = []
                for i in range(1, len(recent_prices) - 1):
                    if recent_prices[i] > recent_prices[i-1] and recent_prices[i] > recent_prices[i+1]:
                        levels.append(recent_prices[i])
            else:  # support
                # Find local minima
                levels = []
                for i in range(1, len(recent_prices) - 1):
                    if recent_prices[i] < recent_prices[i-1] and recent_prices[i] < recent_prices[i+1]:
                        levels.append(recent_prices[i])
            
            # Cluster nearby levels
            clustered_levels = self._cluster_levels(levels)
            return clustered_levels
            
        except Exception as e:
            self.logger.error(f"Error finding key levels: {e}")
            return []
    
    def _cluster_levels(self, levels: List[float]) -> List[float]:
        """Cluster nearby levels to find significant ones."""
        if not levels:
            return []
        
        # Sort levels
        sorted_levels = sorted(levels)
        clustered = []
        tolerance = 0.01  # 1% tolerance for clustering
        
        for level in sorted_levels:
            # Check if this level is close to any existing clustered level
            is_close = False
            for clustered_level in clustered:
                if abs(level - clustered_level) / clustered_level < tolerance:
                    is_close = True
                    break
            
            if not is_close:
                clustered.append(level)
        
        return clustered
    
    def _analyze_volume_profile(self, multi_tf_data: Dict) -> Dict:
        """Analyze volume profile for confirmation."""
        try:
            # Use 1h timeframe for volume analysis
            candles_1h = multi_tf_data.get('1h', [])
            if len(candles_1h) < 20:
                return {'high_volume': False}
            
            volumes = [float(c['volume']) for c in candles_1h[-20:]]
            avg_volume = sum(volumes) / len(volumes)
            current_volume = volumes[-1]
            
            # Volume should be above average
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            
            return {
                'high_volume': volume_ratio > 1.2,  # 20% above average
                'volume_ratio': volume_ratio,
                'current_volume': current_volume,
                'avg_volume': avg_volume
            }
            
        except Exception as e:
            self.logger.error(f"Error in volume profile analysis: {e}")
            return {'high_volume': False}
    
    def _create_enhanced_signal(self, symbol: str, multi_tf_data: Dict, 
                               structure_analysis: Dict, trend_analysis: Dict,
                               sr_analysis: Dict, volume_analysis: Dict) -> Optional[Dict]:
        """Create enhanced signal with improved risk/reward."""
        try:
            # Use 15m timeframe for main analysis
            candles_15m = multi_tf_data.get('15m', [])
            if len(candles_15m) < 50:
                return None
            
            current_price = float(candles_15m[-1]['close'])
            
            # Calculate enhanced indicators
            indicators = self._calculate_enhanced_indicators(candles_15m)
            if not indicators:
                return None
            
            # Determine signal direction based on confluence
            signal_direction = self._determine_signal_direction(
                indicators, structure_analysis, trend_analysis, sr_analysis
            )
            
            if not signal_direction:
                return None
            
            # Calculate enhanced entry, stop loss, and take profit
            entry_price = current_price
            stop_loss, take_profit = self._calculate_enhanced_exits(
                signal_direction, entry_price, indicators, sr_analysis
            )
            
            # Calculate position size and profit
            position_size = self.balance * 0.25  # 25% of balance
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
            quality_score = self._calculate_quality_score(
                indicators, structure_analysis, trend_analysis, sr_analysis, volume_analysis
            )
            
            # Only return signal if it meets high standards
            if (quality_score >= self.min_score_threshold and 
                net_profit >= self.min_profit_threshold):
                
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
                    "max_hold_time": 8,  # 8 hours for enhanced signals
                    "score": quality_score,
                    "score_max": 10,
                    "score_reasons": self._get_score_reasons(indicators, structure_analysis, trend_analysis),
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
            self.logger.error(f"Error creating enhanced signal: {e}")
            return None
    
    def _calculate_enhanced_indicators(self, candles: List[Dict]) -> Optional[Dict]:
        """Calculate enhanced technical indicators."""
        try:
            rsi = self.indicators.calculate_rsi(candles)
            ema_20 = self.indicators.calculate_ema(candles, 20)
            ema_50 = self.indicators.calculate_ema(candles, 50)
            macd = self.indicators.calculate_macd(candles)
            bb_upper, bb_lower = self.indicators.calculate_bollinger_bands(candles)
            stoch_rsi = self.indicators.calculate_stoch_rsi(candles)
            atr = self.indicators.calculate_atr(candles)
            adx = self.indicators.calculate_adx(candles)
            
            if any(x is None for x in [rsi, ema_20, ema_50, macd, bb_upper, bb_lower, stoch_rsi, atr, adx]):
                return None
            
            return {
                "rsi": rsi,
                "ema_20": ema_20,
                "ema_50": ema_50,
                "macd": macd,
                "bollinger_upper": bb_upper,
                "bollinger_lower": bb_lower,
                "stoch_rsi": stoch_rsi,
                "atr": atr,
                "adx": adx
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating enhanced indicators: {e}")
            return None
    
    def _determine_signal_direction(self, indicators: Dict, structure_analysis: Dict,
                                  trend_analysis: Dict, sr_analysis: Dict) -> Optional[str]:
        """Determine signal direction based on confluence."""
        try:
            # Long conditions
            long_score = 0
            if indicators['ema_20'] > indicators['ema_50']:
                long_score += 1
            if indicators['rsi'] > 40 and indicators['rsi'] < 70:
                long_score += 1
            if indicators['macd'] > 0:
                long_score += 1
            if structure_analysis.get('uptrend', False):
                long_score += 1
            if trend_analysis['strength'] > 0.7:
                long_score += 1
            if sr_analysis.get('near_support', False):
                long_score += 1
            
            # Short conditions
            short_score = 0
            if indicators['ema_20'] < indicators['ema_50']:
                short_score += 1
            if indicators['rsi'] < 60 and indicators['rsi'] > 30:
                short_score += 1
            if indicators['macd'] < 0:
                short_score += 1
            if structure_analysis.get('downtrend', False):
                short_score += 1
            if trend_analysis['strength'] > 0.7:
                short_score += 1
            if sr_analysis.get('near_resistance', False):
                short_score += 1
            
            # Need at least 4 out of 6 conditions for signal
            if long_score >= 4 and long_score > short_score:
                return 'long'
            elif short_score >= 4 and short_score > long_score:
                return 'short'
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error determining signal direction: {e}")
            return None
    
    def _calculate_enhanced_exits(self, direction: str, entry_price: float,
                                indicators: Dict, sr_analysis: Dict) -> Tuple[float, float]:
        """Calculate enhanced stop loss and take profit levels."""
        try:
            atr = indicators['atr']
            if atr is None:
                atr = entry_price * 0.02  # 2% default ATR
            
            # Enhanced risk/reward ratio (1:2 minimum)
            if direction == 'long':
                # Use ATR-based stops with support levels
                stop_loss = entry_price - (atr * 1.5)
                take_profit = entry_price + (atr * 3.0)  # 1:2 risk/reward
                
                # Adjust based on support levels
                if sr_analysis.get('support_levels'):
                    nearest_support = max(sr_analysis['support_levels'])
                    if nearest_support < entry_price:
                        stop_loss = max(stop_loss, nearest_support)
                
            else:  # short
                # Use ATR-based stops with resistance levels
                stop_loss = entry_price + (atr * 1.5)
                take_profit = entry_price - (atr * 3.0)  # 1:2 risk/reward
                
                # Adjust based on resistance levels
                if sr_analysis.get('resistance_levels'):
                    nearest_resistance = min(sr_analysis['resistance_levels'])
                    if nearest_resistance > entry_price:
                        stop_loss = min(stop_loss, nearest_resistance)
            
            return stop_loss, take_profit
            
        except Exception as e:
            self.logger.error(f"Error calculating enhanced exits: {e}")
            # Fallback to simple percentages
            if direction == 'long':
                return entry_price * 0.97, entry_price * 1.06  # 3% SL, 6% TP
            else:
                return entry_price * 1.03, entry_price * 0.94  # 3% SL, 6% TP
    
    def _calculate_quality_score(self, indicators: Dict, structure_analysis: Dict,
                               trend_analysis: Dict, sr_analysis: Dict, volume_analysis: Dict) -> int:
        """Calculate quality score for signal."""
        try:
            score = 0
            
            # Trend strength (0-2 points)
            if trend_analysis['strength'] > 0.8:
                score += 2
            elif trend_analysis['strength'] > 0.6:
                score += 1
            
            # Market structure (0-2 points)
            if structure_analysis.get('uptrend') or structure_analysis.get('downtrend'):
                score += 2
            
            # Support/Resistance confluence (0-2 points)
            if sr_analysis.get('confluence', False):
                score += 2
            
            # Volume confirmation (0-1 point)
            if volume_analysis.get('high_volume', False):
                score += 1
            
            # Indicator confluence (0-3 points)
            indicator_score = 0
            if indicators['rsi'] > 40 and indicators['rsi'] < 70:
                indicator_score += 1
            if indicators['macd'] > 0:
                indicator_score += 1
            if indicators['ema_20'] > indicators['ema_50']:
                indicator_score += 1
            
            score += min(indicator_score, 3)
            
            return score
            
        except Exception as e:
            self.logger.error(f"Error calculating quality score: {e}")
            return 0
    
    def _get_score_reasons(self, indicators: Dict, structure_analysis: Dict,
                          trend_analysis: Dict) -> List[str]:
        """Get reasons for the quality score."""
        reasons = []
        
        if trend_analysis['strength'] > 0.8:
            reasons.append("Strong multi-timeframe trend")
        elif trend_analysis['strength'] > 0.6:
            reasons.append("Moderate multi-timeframe trend")
        
        if structure_analysis.get('uptrend'):
            reasons.append("Clear uptrend structure")
        elif structure_analysis.get('downtrend'):
            reasons.append("Clear downtrend structure")
        
        if indicators['rsi'] > 40 and indicators['rsi'] < 70:
            reasons.append("Optimal RSI range")
        
        if indicators['macd'] > 0:
            reasons.append("Positive MACD momentum")
        
        if indicators['ema_20'] > indicators['ema_50']:
            reasons.append("EMA trend alignment")
        
        return reasons 