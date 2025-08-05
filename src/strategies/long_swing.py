from src.data.fetcher import CoinDCXFetcher
from src.data.indicators import TechnicalIndicators
from src.utils.logger import setup_logger
import yaml
import pandas as pd

class LongSwingStrategy:
    def __init__(self, balance):
        self.fetcher = CoinDCXFetcher()
        self.indicators = TechnicalIndicators()
        with open("config/config.yaml", "r") as file:
            self.config = yaml.safe_load(file)
        self.balance = balance
        self.logger = setup_logger()

    def generate_signal(self, symbol, timeframe, direction=None):
        self.logger.debug(f"[Long Swing] generate_signal called for {symbol} {timeframe} direction={direction}")
        try:
            market_data = self.fetcher.fetch_market_data(symbol)
            if not market_data or "last_price" not in market_data:
                self.logger.warning(f"No valid market data for {symbol}")
                return None
            current_price = market_data["last_price"]

            # MULTI-TIMEFRAME ANALYSIS: Fetch data for 1h, 30m, and 15m timeframes
            multi_tf_timeframes = ['1h', '30m', '15m']
            multi_tf_data = self.fetcher.fetch_multi_timeframe_data(symbol, multi_tf_timeframes, limit=120)
            
            if not multi_tf_data:
                self.logger.warning(f"No multi-timeframe data available for {symbol}")
                return None
            
            # Use the primary timeframe (usually 1h or 30m for long_swing)
            primary_tf = timeframe if timeframe in multi_tf_data else '1h'
            candles = multi_tf_data.get(primary_tf)
            
            if not candles or len(candles) < 50:
                self.logger.warning(f"Insufficient {primary_tf} candles for {symbol}: {len(candles) if candles else 0}")
                return None

            # --- Multi-timeframe high/low calculation ---
            highs = []
            lows = []
            for tf in multi_tf_timeframes:
                tf_candles = multi_tf_data.get(tf)
                if tf_candles and len(tf_candles) >= 20:
                    highs.append(max(c['high'] for c in tf_candles[-20:]))
                    lows.append(min(c['low'] for c in tf_candles[-20:]))
            if not highs or not lows:
                self.logger.warning(f"No highs/lows for {symbol}")
                return None
            nearest_resistance = min(highs)
            nearest_support = max(lows)

            # Calculate indicators
            rsi = self.indicators.calculate_rsi(candles)
            ema_20 = self.indicators.calculate_ema(candles, 20)
            ema_50 = self.indicators.calculate_ema(candles, 50)
            ema_200 = self.indicators.calculate_ema(candles, 200)
            macd_long = self.indicators.calculate_macd_long_swing(candles)
            bb_upper, bb_lower = self.indicators.calculate_bollinger_bands(candles)
            stoch_rsi = self.indicators.calculate_stoch_rsi(candles)
            vwap_daily = self.indicators.calculate_vwap_daily(candles)
            atr = self.indicators.calculate_atr(candles)

            if any(x is None for x in [rsi, ema_20, ema_50, ema_200, macd_long, bb_upper, bb_lower, stoch_rsi, vwap_daily, atr]):
                self.logger.warning(f"Indicator failure for {symbol}: rsi={rsi}, ema_20={ema_20}, ema_50={ema_50}, ema_200={ema_200}, macd_long={macd_long}, bb_upper={bb_upper}, bb_lower={bb_lower}, stoch_rsi={stoch_rsi}, vwap_daily={vwap_daily}, atr={atr}")
                return None

            # MULTI-TIMEFRAME TREND CONFIRMATION
            trend_confirmation = self._analyze_multi_timeframe_trend(multi_tf_data, current_price)
            self.logger.debug(f"[Long Swing] Multi-timeframe trend confirmation for {symbol}: {trend_confirmation}")

            # Remove VOLATILITY FILTER and add LIQUIDITY FILTER
            volumes = [float(candle["volume"]) for candle in candles]
            avg_volume = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else 1.0
            min_avg_volume = 300000  # Increased threshold for highly voluminous signals
            if avg_volume < min_avg_volume:
                self.logger.info(f"[LongSwing] Skipping {symbol} due to low liquidity (avg_volume={avg_volume:.2f} < {min_avg_volume})")
                return None

            stoch_k = stoch_rsi["%K"]
            stoch_d = stoch_rsi["%D"]
            if pd.isna(stoch_d):
                self.logger.debug(f"StochRSI %D is NaN for {symbol}, using %K for logic.")
                stoch_d = stoch_k

            # Volume confirmation (20-period SMA)
            volumes = [float(candle["volume"]) for candle in candles]
            avg_volume = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else 1.0
            current_volume = volumes[-1] if volumes else 0
            volume_ok = current_volume > avg_volume

            # Order book and spread checks
            order_book = self.fetcher.fetch_order_book(symbol)
            if not order_book or not all(k in order_book for k in ["best_ask", "best_bid", "ask_vol", "bid_vol"]):
                self.logger.warning(f"Invalid order book for {symbol}: {order_book}")
                return None
            spread = order_book.get("spread", 0)
            spread_pct = (spread / current_price) * 100 if spread and current_price else 100
            spread_limit = 1.5  # 1.5% for long swing trading
            if spread_pct > spread_limit:
                self.logger.debug(f"Spread too high for {symbol}: {spread_pct:.3f}% > {spread_limit}%")
                return None

            # --- SIMPLIFIED LONG SWING STRATEGY ENTRY LOGIC ---
            signal = None
            
            # SIMPLIFIED LONG SWING CONDITIONS - Focus on clear bullish setups
            long_conditions = []
            
            # 1. CLEAR EMA TREND: EMA20 > EMA50 > EMA200 with minimum distance
            ema_trend_strength_20_50 = ((ema_20 - ema_50) / ema_50) * 100
            ema_trend_strength_50_200 = ((ema_50 - ema_200) / ema_200) * 100
            ema_trend = ema_20 > ema_50 > ema_200 and ema_trend_strength_20_50 > 0.2 and ema_trend_strength_50_200 > 0.2
            long_conditions.append(("Strong EMA uptrend (20>50>200)", ema_trend))
            
            # 2. PRICE ABOVE EMAs: Current price > EMA20
            price_above_ema = current_price > ema_20
            long_conditions.append(("Price > EMA20", price_above_ema))
            
            # 3. RSI OPTIMAL BULLISH: RSI between 45-65 (not overbought)
            rsi_bullish = 45 <= rsi <= 65
            long_conditions.append(("RSI optimal bullish (45-65)", rsi_bullish))
            
            # 4. MACD POSITIVE AND RISING: MACD line > signal line and histogram > 0
            macd_bullish = macd_long["macd_line"] > macd_long["signal_line"] and macd_long["histogram"] > 0
            long_conditions.append(("Strong MACD momentum (36,78,18)", macd_bullish))
            
            # 5. PRICE ABOVE VWAP: Current price > daily VWAP
            above_vwap = current_price > vwap_daily
            long_conditions.append(("Price > VWAP", above_vwap))
            
            # 6. VOLUME CONFIRMATION: Current volume > average
            long_conditions.append(("Volume > 20MA", volume_ok))
            
            # 7. NOT OVERBOUGHT: StochRSI < 80 (avoid extreme overbought)
            not_overbought = stoch_k < 80 and stoch_d < 80
            long_conditions.append(("StochRSI not overbought", not_overbought))
            
            # 8. MULTI-TIMEFRAME TREND CONFIRMATION
            multi_tf_bullish = trend_confirmation.get('trend', 'neutral') in ['bullish', 'strong_bullish']
            long_conditions.append(("Multi-timeframe trend bullish", multi_tf_bullish))
            
            # --- ADVANCED INDICATORS (Pro-Trader Logic) ---
            ema_100 = self.indicators.calculate_ema(candles, 100)
            supertrend = self.indicators.calculate_supertrend(candles)
            adx = self.indicators.calculate_adx(candles)
            obv = self.indicators.calculate_obv(candles)
            cci = self.indicators.calculate_cci(candles)
            momentum_osc = self.indicators.calculate_momentum(candles)
            keltner_upper, keltner_lower = self.indicators.calculate_keltner_channels(candles)
            fib_levels = self.indicators.calculate_fibonacci_levels(candles)

            if any(x is None for x in [ema_100, supertrend, adx, obv, cci, momentum_osc, keltner_upper, keltner_lower, fib_levels]):
                self.logger.warning(f"Advanced indicator failure for {symbol}")
                return None

            # --- ENHANCED LONG CONDITIONS ---
            long_conditions.append(("Supertrend uptrend", supertrend['in_uptrend']))
            long_conditions.append(("ADX > 25 and DI+ > DI-", adx['adx'] > 25 and adx['di_plus'] > adx['di_minus']))
            long_conditions.append(("Price > EMA100", current_price > ema_100))
            long_conditions.append(("OBV rising", obv > 0))

            # SIMPLIFIED SHORT SWING CONDITIONS - Focus on clear bearish setups
            short_conditions = []
            
            # 1. CLEAR EMA TREND: EMA20 < EMA50 < EMA200 with minimum distance
            ema_trend_short = ema_20 < ema_50 < ema_200 and ema_trend_strength_20_50 < -0.2 and ema_trend_strength_50_200 < -0.2
            short_conditions.append(("Strong EMA downtrend (20<50<200)", ema_trend_short))
            
            # 2. PRICE BELOW EMAs: Current price < EMA20
            price_below_ema = current_price < ema_20
            short_conditions.append(("Price < EMA20", price_below_ema))
            
            # 3. RSI OPTIMAL BEARISH: RSI between 35-55 (not oversold)
            rsi_bearish = 35 <= rsi <= 55
            short_conditions.append(("RSI optimal bearish (35-55)", rsi_bearish))
            
            # 4. MACD NEGATIVE AND FALLING: MACD line < signal line and histogram < 0
            macd_bearish = macd_long["macd_line"] < macd_long["signal_line"] and macd_long["histogram"] < 0
            short_conditions.append(("Strong MACD momentum (36,78,18) bearish", macd_bearish))
            
            # 5. PRICE BELOW VWAP: Current price < daily VWAP
            below_vwap = current_price < vwap_daily
            short_conditions.append(("Price < VWAP", below_vwap))
            
            # 6. VOLUME CONFIRMATION: Current volume > average
            short_conditions.append(("Volume > 20MA", volume_ok))
            
            # 7. NOT OVERSOLD: StochRSI > 20 (avoid extreme oversold)
            not_oversold = stoch_k > 20 and stoch_d > 20
            short_conditions.append(("StochRSI not oversold", not_oversold))
            
            # 8. MULTI-TIMEFRAME TREND CONFIRMATION
            multi_tf_bearish = trend_confirmation.get('trend', 'neutral') in ['bearish', 'strong_bearish']
            short_conditions.append(("Multi-timeframe trend bearish", multi_tf_bearish))
            
            # --- ENHANCED SHORT CONDITIONS ---
            short_conditions.append(("Supertrend downtrend", not supertrend['in_uptrend']))
            short_conditions.append(("ADX > 25 and DI- > DI+", adx['adx'] > 25 and adx['di_minus'] > adx['di_plus']))
            short_conditions.append(("Price < EMA100", current_price < ema_100))
            short_conditions.append(("OBV falling", obv < 0))

            # --- BONUS CONFIDENCE BOOSTERS ---
            bonus_score = 0
            bonus_reasons = []
            if cci > 100:
                bonus_score += 1
                bonus_reasons.append("CCI > 100 (bullish)")
            if cci < -100:
                bonus_score += 1
                bonus_reasons.append("CCI < -100 (bearish)")
            if momentum_osc > 0:
                bonus_score += 1
                bonus_reasons.append("Momentum Oscillator positive (bullish)")
            if momentum_osc < 0:
                bonus_score += 1
                bonus_reasons.append("Momentum Oscillator negative (bearish)")
            if current_price > keltner_upper:
                bonus_score += 1
                bonus_reasons.append("Price above Keltner upper (bullish breakout)")
            if current_price < keltner_lower:
                bonus_score += 1
                bonus_reasons.append("Price below Keltner lower (bearish breakout)")
            fib_382 = fib_levels['38.2%']
            fib_618 = fib_levels['61.8%']
            if abs(current_price - fib_382) / current_price < 0.005:
                bonus_score += 1
                bonus_reasons.append("Entry near Fib 38.2% level")
            if abs(current_price - fib_618) / current_price < 0.005:
                bonus_score += 1
                bonus_reasons.append("Entry near Fib 61.8% level")

            # Check long/short conditions (REDUCED TO 5 out of 8 for better signal generation)
            long_score = sum(1 for _, condition in long_conditions if condition)
            short_score = sum(1 for _, condition in short_conditions if condition)
            self.logger.debug(f"[Long Swing] {symbol} {timeframe} Long conditions: {[(name, condition) for name, condition in long_conditions]}")
            self.logger.debug(f"[Long Swing] {symbol} {timeframe} Short conditions: {[(name, condition) for name, condition in short_conditions]}")
            
            # DETERMINE TRADE DIRECTION (REDUCED TO 5 out of 8 for better signal generation)
            if long_score >= 5 and long_score > short_score:
                side = "long"
                score = long_score
                reasons = [name for name, condition in long_conditions if condition]
                self.logger.debug(f"[Long Swing] LONG signal conditions met: {reasons}")
            elif short_score >= 5 and short_score > long_score:
                side = "short"
                score = short_score
                reasons = [name for name, condition in short_conditions if condition]
                self.logger.debug(f"[Long Swing] SHORT signal conditions met: {reasons}")
            else:
                self.logger.debug(f"[Long Swing] No signal: Long score={long_score}/8, Short score={short_score}/8 (need 5+)")
                return None
            # Force direction if specified
            if direction is not None and side != direction:
                self.logger.debug(f"[Long Swing] Skipping signal for {symbol}: forced direction {direction}, got {side}")
                return None
            # Calculate pivot points
            # pivots = self.indicators.calculate_pivot_points(candles)
            
            # Filter out problematic score 8 signals (tend to hit SL frequently)
            if side and score == 8:
                self.logger.info(f"[Long Swing] FILTERED OUT: {symbol} {timeframe} {side.upper()} signal with score {score}/8 - Score 8 signals tend to hit stop loss frequently")
                return None
            
            # Generate signal if conditions are met
            if side and score >= 6:  # Need 6 out of 8 conditions
                entry_price = current_price
                
                # IMPROVED: Better TP/SL for long swing trading with ATR
                if side == "long":
                    # Long trade: SL below entry, TP above entry
                    atr_multiplier_sl = 1.5
                    atr_multiplier_tp = 2.5
                    stop_loss = entry_price - (atr * atr_multiplier_sl)
                    tp1 = entry_price + (atr * atr_multiplier_tp)
                    tp2 = entry_price + (atr * (atr_multiplier_tp + 1.0))
                else:
                    # Short trade: SL above entry, TP below entry
                    atr_multiplier_sl = 1.5
                    atr_multiplier_tp = 2.5
                    stop_loss = entry_price + (atr * atr_multiplier_sl)
                    tp1 = entry_price - (atr * atr_multiplier_tp)
                    tp2 = entry_price - (atr * (atr_multiplier_tp + 1.0))

                # Calculate estimated profit in INR
                usdt_inr_rate = self.fetcher.fetch_usdt_inr_rate()
                if side == "long":
                    profit_pct = ((tp1 - entry_price) / entry_price) * 100
                else:
                    profit_pct = ((entry_price - tp1) / entry_price) * 100
                
                estimated_profit_inr = (self.balance * profit_pct / 100) * usdt_inr_rate

                # Calculate confidence based on score and multi-timeframe confirmation
                base_confidence = min(score / 9.0, 1.0)  # Normalize to 0-1
                trend_strength = trend_confirmation.get('strength', 0.5)
                confidence = (base_confidence + trend_strength) / 2

                # Calculate quality score
                quality_score = (score * 10) + (trend_strength * 50) + (confidence * 100)

                # Normalize quality score to 0-100
                max_possible_score = 9 * 10 + 1.0 * 50 + 1.0 * 100  # 240
                normalized_score = min((quality_score / max_possible_score) * 100, 100)
                
                # Calculate total possible score
                score_max = len(long_conditions) + 5  # 5 is the max possible bonus points
                # Add bonus_score to main score
                total_score = score + bonus_score
                total_reasons = reasons + bonus_reasons
                signal = {
                    "symbol": symbol,
                    "timeframe": primary_tf,
                    "strategy": "long_swing",
                    "direction": f"TradeDirection.{side.upper()}",
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": tp1,
                    "tp1": tp1,
                    "tp2": tp2,
                    "max_hold_time": 24,  # 24 hours for long swing
                    "score": total_score,
                    "score_max": score_max,
                    "score_reasons": total_reasons,
                    "indicators": {
                        "rsi": rsi,
                        "ema_20": ema_20,
                        "ema_50": ema_50,
                        "ema_200": ema_200,
                        "macd": macd_long["histogram"],
                        "bollinger_upper": bb_upper,
                        "bollinger_lower": bb_lower,
                        "vwap_daily": vwap_daily,
                        "volume_ok": volume_ok,
                        "stoch_rsi_k": stoch_k,
                        "stoch_rsi_d": stoch_d,
                        "atr": atr,
                        "multi_timeframe_trend": trend_confirmation.get('trend', 'neutral'),
                        "trend_strength": trend_confirmation.get('strength', 0.5),
                        "ema_100": ema_100,
                        "supertrend": supertrend,
                        "adx": adx,
                        "obv": obv,
                        "cci": cci,
                        "momentum_osc": momentum_osc,
                        "keltner_upper": keltner_upper,
                        "keltner_lower": keltner_lower,
                        "fib_levels": fib_levels
                    },
                    "timestamp": self.fetcher.get_ist_timestamp(),
                    "estimated_profit_inr": estimated_profit_inr,
                    "confidence": confidence,
                    "quality_score": quality_score,
                    "normalized_score": normalized_score,
                    "side": side,
                    "bonus_score": bonus_score,
                    "bonus_reasons": bonus_reasons
                }

                self.logger.info(f"[Long Swing] Generated {side.upper()} signal for {symbol} on {primary_tf} - Score: {score}/9, Confidence: {confidence:.3f}, TP: {tp1:.6f}, SL: {stop_loss:.6f}")
                return signal
                
        except Exception as e:
            self.logger.error(f"Error generating long swing signal for {symbol}: {e}")
            return None 

    def _analyze_multi_timeframe_trend(self, multi_tf_data, current_price):
        """Analyze trend across multiple timeframes for confirmation.
        
        Args:
            multi_tf_data: Dictionary with timeframe as key and candle data as value
            current_price: Current market price
            
        Returns:
            Dictionary with trend analysis results
        """
        try:
            trend_scores = {'bullish': 0, 'bearish': 0, 'neutral': 0}
            total_timeframes = 0
            trend_consistency = []
            momentum_scores = []
            
            for tf, candles in multi_tf_data.items():
                if not candles or len(candles) < 20:
                    continue
                    
                total_timeframes += 1
                
                # Calculate EMAs for trend analysis (more stringent for long swing)
                ema_20 = self.indicators.calculate_ema(candles, 20)
                ema_50 = self.indicators.calculate_ema(candles, 50)
                ema_200 = self.indicators.calculate_ema(candles, 200)
                
                if ema_20 is None or ema_50 is None or ema_200 is None:
                    continue
                
                # Calculate EMA slopes for momentum
                if len(candles) >= 25:
                    ema_20_prev = self.indicators.calculate_ema(candles[:-5], 20)  # 5 candles ago
                    ema_50_prev = self.indicators.calculate_ema(candles[:-5], 50)
                    ema_200_prev = self.indicators.calculate_ema(candles[:-5], 200)
                    
                    if ema_20_prev and ema_50_prev and ema_200_prev:
                        ema_20_slope = (ema_20 - ema_20_prev) / ema_20_prev
                        ema_50_slope = (ema_50 - ema_50_prev) / ema_50_prev
                        ema_200_slope = (ema_200 - ema_200_prev) / ema_200_prev
                        momentum_score = (ema_20_slope + ema_50_slope + ema_200_slope) / 3
                        momentum_scores.append(momentum_score)
                
                # Calculate price momentum
                if len(candles) >= 10:
                    price_5_ago = float(candles[-5]["close"])
                    price_10_ago = float(candles[-10]["close"])
                    price_momentum = (current_price - price_5_ago) / price_5_ago
                    momentum_scores.append(price_momentum)
                
                # Determine trend for this timeframe (more stringent for long swing)
                trend_score = 0
                if ema_20 > ema_50 > ema_200 and current_price > ema_20:
                    trend_scores['bullish'] += 1
                    trend_score = 1
                elif ema_20 < ema_50 < ema_200 and current_price < ema_20:
                    trend_scores['bearish'] += 1
                    trend_score = -1
                else:
                    trend_scores['neutral'] += 1
                    trend_score = 0
                
                trend_consistency.append(trend_score)
            
            if total_timeframes == 0:
                return {'trend': 'neutral', 'strength': 0.5}
            
            # Calculate more nuanced trend strength
            max_trend = max(trend_scores, key=trend_scores.get)
            base_strength = trend_scores[max_trend] / total_timeframes
            
            # Adjust strength based on momentum consistency
            momentum_consistency = 0
            if momentum_scores:
                # Calculate how consistent the momentum is across timeframes
                positive_momentum = sum(1 for m in momentum_scores if m > 0)
                negative_momentum = sum(1 for m in momentum_scores if m < 0)
                total_momentum = len(momentum_scores)
                
                if total_momentum > 0:
                    if max_trend == 'bullish':
                        momentum_consistency = positive_momentum / total_momentum
                    elif max_trend == 'bearish':
                        momentum_consistency = negative_momentum / total_momentum
                    else:
                        momentum_consistency = 0.5
            
            # Calculate trend consistency across timeframes
            trend_variance = 0
            if len(trend_consistency) > 1:
                # Calculate variance in trend signals
                trend_mean = sum(trend_consistency) / len(trend_consistency)
                trend_variance = sum((t - trend_mean) ** 2 for t in trend_consistency) / len(trend_consistency)
                # Convert variance to consistency (lower variance = higher consistency)
                trend_consistency_score = max(0, 1 - (trend_variance / 2))
            else:
                trend_consistency_score = 1.0
            
            # Combine factors for final strength (more stringent for long swing)
            final_strength = (base_strength * 0.5 + 
                            momentum_consistency * 0.3 + 
                            trend_consistency_score * 0.2)
            
            # Ensure strength is between 0.1 and 1.0
            final_strength = max(0.1, min(1.0, final_strength))
            
            # Enhance trend description based on strength
            if final_strength >= 0.8:
                if max_trend == 'bullish':
                    trend = 'strong_bullish'
                elif max_trend == 'bearish':
                    trend = 'strong_bearish'
                else:
                    trend = 'neutral'
            elif final_strength >= 0.6:
                if max_trend == 'bullish':
                    trend = 'bullish'
                elif max_trend == 'bearish':
                    trend = 'bearish'
                else:
                    trend = 'neutral'
            else:
                trend = 'weak_' + max_trend if max_trend != 'neutral' else 'neutral'
            
            return {
                'trend': trend,
                'strength': final_strength,
                'scores': trend_scores,
                'total_timeframes': total_timeframes,
                'momentum_consistency': momentum_consistency,
                'trend_consistency': trend_consistency_score
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing multi-timeframe trend: {e}")
            return {'trend': 'neutral', 'strength': 0.5} 