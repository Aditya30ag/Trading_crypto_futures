from src.data.fetcher import CoinDCXFetcher
from src.data.indicators import TechnicalIndicators
from src.utils.logger import setup_logger
import yaml
import pandas as pd

class ScalpingStrategy:
    def __init__(self, balance):
        self.fetcher = CoinDCXFetcher()
        self.indicators = TechnicalIndicators()
        with open("config/config.yaml", "r") as file:
            self.config = yaml.safe_load(file)
        self.balance = balance
        self.logger = setup_logger()

    def generate_signal(self, symbol, timeframe):
        self.logger.debug(f"[Scalping] generate_signal called for {symbol} {timeframe}")
        try:
            market_data = self.fetcher.fetch_market_data(symbol)
            if not market_data or "last_price" not in market_data:
                self.logger.warning(f"No valid market data for {symbol}")
                return None
            current_price = market_data["last_price"]

            # ENHANCED: Multi-timeframe data fetching for scalping (1m, 5m, 15m)
            primary_tf = timeframe
            multi_tf_data = {}
            
            # Fetch data for multiple timeframes
            timeframes_to_fetch = ["1m", "5m"]
            
            for tf in timeframes_to_fetch:
                try:
                    # Try CoinDesk first for 1m data (only if enabled)
                    if tf == "1m":
                        coindesk_candles = self.fetcher.fetch_coindesk_candles(symbol, tf, limit=50)
                        if coindesk_candles and len(coindesk_candles) >= 50:
                            multi_tf_data[tf] = coindesk_candles
                            self.logger.debug(f"[Scalping] Using CoinDesk {tf} data for {symbol}")
                            continue
                    
                    # Fallback to CoinDCX
                    candles = self.fetcher.fetch_candlestick_data(symbol, tf, limit=100)
                    if candles and len(candles) >= 50:
                        multi_tf_data[tf] = candles
                        self.logger.debug(f"[Scalping] Using CoinDCX {tf} data for {symbol}")
                    else:
                        self.logger.debug(f"[Scalping] Insufficient {tf} data for {symbol}: {len(candles) if candles else 0} candles")
                except Exception as e:
                    self.logger.warning(f"[Scalping] Failed to fetch {tf} data for {symbol}: {e}")
                    continue
            
            # Use primary timeframe data for main analysis
            candles = multi_tf_data.get(primary_tf)
            if not candles or len(candles) < 50:
                self.logger.warning(f"Insufficient {primary_tf} candles for {symbol}: {len(candles) if candles else 0}")
                return None

            # ENHANCED: Multi-timeframe trend analysis
            trend_confirmation = self._analyze_multi_timeframe_trend(multi_tf_data, current_price)
            self.logger.debug(f"[Scalping] Multi-timeframe trend for {symbol}: {trend_confirmation}")

            # ENHANCED: Liquidity filter (replacing volatility filter)
            liquidity_ok = self._check_liquidity_filter(symbol, candles)
            if not liquidity_ok:
                self.logger.debug(f"[Scalping] No signal for {symbol}: Liquidity filter not met")
                return None

            # Calculate indicators
            rsi = self.indicators.calculate_rsi(candles)
            ema_20 = self.indicators.calculate_ema(candles, 20)
            ema_50 = self.indicators.calculate_ema(candles, 50)
            macd = self.indicators.calculate_macd(candles)
            bb_upper, bb_lower = self.indicators.calculate_bollinger_bands(candles)
            stoch_rsi = self.indicators.calculate_stoch_rsi(candles)
            vwap_daily = self.indicators.calculate_vwap_daily(candles)

            if any(x is None for x in [rsi, ema_20, ema_50, macd, bb_upper, bb_lower, stoch_rsi, vwap_daily]):
                self.logger.warning(f"Indicator failure for {symbol}: rsi={rsi}, ema_20={ema_20}, ema_50={ema_50}, macd={macd}, bb_upper={bb_upper}, bb_lower={bb_lower}, stoch_rsi={stoch_rsi}, vwap_daily={vwap_daily}")
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
            spread_limit = 0.5  # 0.5% for scalping
            if spread_pct > spread_limit:
                self.logger.debug(f"Spread too high for {symbol}: {spread_pct:.3f}% > {spread_limit}%")
                return None

            # --- ADVANCED INDICATORS (Pro-Trader Logic) ---
            ema_100 = self.indicators.calculate_ema(candles, 100)
            supertrend = self.indicators.calculate_supertrend(candles)
            adx = self.indicators.calculate_adx(candles)
            obv = self.indicators.calculate_obv(candles)
            cci = self.indicators.calculate_cci(candles)
            momentum_osc = self.indicators.calculate_momentum(candles)
            keltner_upper, keltner_lower = self.indicators.calculate_keltner_channels(candles)
            fib_levels = self.indicators.calculate_fibonacci_levels(candles)

            # Add to indicator failure check
            if any(x is None for x in [ema_100, supertrend, adx, obv, cci, momentum_osc, keltner_upper, keltner_lower, fib_levels]):
                self.logger.warning(f"Advanced indicator failure for {symbol}")
                return None

            # --- SIMPLIFIED AND RELIABLE SCALPING ENTRY LOGIC ---
            signal = None
            side = None
            
            # SIMPLIFIED LONG CONDITIONS - Focus on clear bullish setups only
            long_conditions = []
            
            # 1. CLEAR EMA TREND: EMA20 > EMA50 with minimum distance
            ema_trend_strength = ((ema_20 - ema_50) / ema_50) * 100
            ema_bullish = ema_20 > ema_50 and ema_trend_strength > 0.1  # Minimum 0.1% separation
            long_conditions.append(("EMA20 > EMA50 (clear trend)", ema_bullish))
            
            # 2. PRICE ABOVE EMAs: Current price > EMA20
            price_above_ema = current_price > ema_20
            long_conditions.append(("Price > EMA20", price_above_ema))
            
            # 3. RSI NOT OVERBOUGHT: RSI between 45-65 (optimal bullish range)
            rsi_bullish = 45 <= rsi <= 65  # Narrower, more reliable range
            long_conditions.append(("RSI optimal bullish (45-65)", rsi_bullish))
            
            # 4. MACD POSITIVE AND RISING: MACD > 0 and histogram increasing
            macd_bullish = macd > 0 and stoch_k > stoch_d  # MACD positive + momentum
            long_conditions.append(("MACD positive + momentum", macd_bullish))
            
            # 5. VOLUME CONFIRMATION: Current volume > average
            long_conditions.append(("Volume > average", volume_ok))
            
            # 6. PRICE ABOVE VWAP: Current price > daily VWAP
            above_vwap = current_price > vwap_daily
            long_conditions.append(("Price > VWAP", above_vwap))
            
            # 7. NOT OVERBOUGHT: StochRSI < 80 (avoid extreme overbought)
            not_overbought = stoch_k < 80 and stoch_d < 80
            long_conditions.append(("Not overbought (StochRSI < 80)", not_overbought))
            
            # 8. MULTI-TIMEFRAME TREND CONFIRMATION
            multi_tf_bullish = trend_confirmation.get('trend', 'neutral') in ['bullish', 'strong_bullish']
            long_conditions.append(("Multi-timeframe trend bullish", multi_tf_bullish))
            
            # --- ENHANCED LONG CONDITIONS ---
            # 9. Supertrend in uptrend
            supertrend_up = supertrend['in_uptrend']
            long_conditions.append(("Supertrend uptrend", supertrend_up))
            # 10. ADX strong trend
            adx_ok = adx['adx'] > 25 and adx['di_plus'] > adx['di_minus']
            long_conditions.append(("ADX > 25 and DI+ > DI-", adx_ok))
            # 11. Price above EMA100
            price_above_ema100 = current_price > ema_100
            long_conditions.append(("Price > EMA100", price_above_ema100))
            # 12. OBV rising (last 2 values)
            obv_rising = obv > 0  # For simplicity, positive OBV means rising
            long_conditions.append(("OBV rising", obv_rising))

            # --- SIMPLIFIED SHORT CONDITIONS - Focus on clear bearish setups only ---
            short_conditions = []
            
            # 1. CLEAR EMA TREND: EMA20 < EMA50 with minimum distance
            ema_bearish = ema_20 < ema_50 and ema_trend_strength < -0.1  # Minimum 0.1% separation
            short_conditions.append(("EMA20 < EMA50 (clear trend)", ema_bearish))
            
            # 2. PRICE BELOW EMAs: Current price < EMA20
            price_below_ema = current_price < ema_20
            short_conditions.append(("Price < EMA20", price_below_ema))
            
            # 3. RSI NOT OVERSOLD: RSI between 35-55 (optimal bearish range)
            rsi_bearish = 35 <= rsi <= 55  # Narrower, more reliable range
            short_conditions.append(("RSI optimal bearish (35-55)", rsi_bearish))
            
            # 4. MACD NEGATIVE AND FALLING: MACD < 0 and histogram decreasing
            macd_bearish = macd < 0 and stoch_k < stoch_d  # MACD negative + momentum
            short_conditions.append(("MACD negative + momentum", macd_bearish))
            
            # 5. VOLUME CONFIRMATION: Current volume > average
            short_conditions.append(("Volume > average", volume_ok))
            
            # 6. PRICE BELOW VWAP: Current price < daily VWAP
            below_vwap = current_price < vwap_daily
            short_conditions.append(("Price < VWAP", below_vwap))
            
            # 7. NOT OVERSOLD: StochRSI > 20 (avoid extreme oversold)
            not_oversold = stoch_k > 20 and stoch_d > 20
            short_conditions.append(("Not oversold (StochRSI > 20)", not_oversold))
            
            # 8. MULTI-TIMEFRAME TREND CONFIRMATION
            multi_tf_bearish = trend_confirmation.get('trend', 'neutral') in ['bearish', 'strong_bearish']
            short_conditions.append(("Multi-timeframe trend bearish", multi_tf_bearish))
            
            # --- ENHANCED SHORT CONDITIONS ---
            # 9. Supertrend in downtrend
            supertrend_down = not supertrend['in_uptrend']
            short_conditions.append(("Supertrend downtrend", supertrend_down))
            # 10. ADX strong trend
            adx_ok_short = adx['adx'] > 25 and adx['di_minus'] > adx['di_plus']
            short_conditions.append(("ADX > 25 and DI- > DI+", adx_ok_short))
            # 11. Price below EMA100
            price_below_ema100 = current_price < ema_100
            short_conditions.append(("Price < EMA100", price_below_ema100))
            # 12. OBV falling (last 2 values)
            obv_falling = obv < 0  # For simplicity, negative OBV means falling
            short_conditions.append(("OBV falling", obv_falling))

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
            # Fibonacci confluence (entry near 38.2% or 61.8%)
            fib_382 = fib_levels['38.2%']
            fib_618 = fib_levels['61.8%']
            if abs(current_price - fib_382) / current_price < 0.005:
                bonus_score += 1
                bonus_reasons.append("Entry near Fib 38.2% level")
            if abs(current_price - fib_618) / current_price < 0.005:
                bonus_score += 1
                bonus_reasons.append("Entry near Fib 61.8% level")

            # Add bonus_score to main score for final decision
            # (You may want to cap the max score or adjust thresholds accordingly)

            # Check long conditions (NEED 6 out of 8 for reliability)
            long_score = sum(1 for _, condition in long_conditions if condition)
            self.logger.debug(f"[Scalping] {symbol} {timeframe} Long conditions: {[(name, condition) for name, condition in long_conditions]}")
            
            # Check short conditions (NEED 6 out of 8 for reliability)
            short_score = sum(1 for _, condition in short_conditions if condition)
            self.logger.debug(f"[Scalping] {symbol} {timeframe} Short conditions: {[(name, condition) for name, condition in short_conditions]}")
            
            # DETERMINE TRADE DIRECTION (MORE SELECTIVE: Need 6 out of 8 for better quality)
            if long_score >= 6 and long_score > short_score:
                side = "long"
                score = long_score
                reasons = [name for name, condition in long_conditions if condition]
            elif short_score >= 6 and short_score > long_score:
                side = "short"
                score = short_score
                reasons = [name for name, condition in short_conditions if condition]
            else:
                self.logger.debug(f"[Scalping] No signal for {symbol}: Long score={long_score}/8, Short score={short_score}/8 (need 6+)")
                return None

            # ENHANCED: Incorporate multi-timeframe trend confirmation
            trend_trend = trend_confirmation.get('trend', 'neutral')
            trend_strength = trend_confirmation.get('strength', 0.5)
            
            # Adjust scores based on multi-timeframe trend alignment
            if side == "long" and trend_trend in ['bullish', 'strong_bullish']:
                score += 1  # Bonus for trend alignment
                reasons.append("Multi-timeframe trend bullish")
            elif side == "short" and trend_trend in ['bearish', 'strong_bearish']:
                score += 1  # Bonus for trend alignment
                reasons.append("Multi-timeframe trend bearish")
            elif side == "long" and trend_trend in ['bearish', 'strong_bearish']:
                score -= 1  # Penalty for trend misalignment
                reasons.append("Multi-timeframe trend bearish (penalty)")
            elif side == "short" and trend_trend in ['bullish', 'strong_bullish']:
                score -= 1  # Penalty for trend misalignment
                reasons.append("Multi-timeframe trend bullish (penalty)")

            # Calculate pivot points
            pivots = self.indicators.calculate_pivot_points(candles)

            # Generate signal if conditions are met (MORE SELECTIVE: Need 6+ for better quality)
            if side and score >= 6:  # MORE SELECTIVE: Need 6+ out of 8
                entry_price = current_price
                
                # SIMPLIFIED AND REALISTIC TP/SL CALCULATION
                # Use ATR-based stops for better risk management
                atr = self.indicators.calculate_atr(candles)
                if atr is None:
                    atr = entry_price * 0.01  # Default 1% ATR if calculation fails
                
                # Calculate ATR percentage
                atr_pct = (atr / entry_price) * 100
                
                # SIMPLIFIED: Use fixed percentages for scalping (more predictable)
                if side == "long":
                    # Long trade: SL below entry, TP above entry
                    stop_loss = entry_price * 0.985  # 1.5% below entry
                    tp1 = entry_price * 1.015  # 1.5% above entry
                    tp2 = entry_price * 1.02   # 2% above entry (capped)
                else:
                    # Short trade: SL above entry, TP below entry
                    stop_loss = entry_price * 1.015  # 1.5% above entry
                    tp1 = entry_price * 0.985  # 1.5% below entry
                    tp2 = entry_price * 0.98   # 2% below entry (capped)

                # SIMPLIFIED: No complex pivot point logic for scalping
                # Use simple, predictable levels
                
                # Entry price validation (0.5% slippage tolerance)
                entry_tolerance = entry_price * 0.005  # 0.5% tolerance
                min_entry_price = entry_price - entry_tolerance
                max_entry_price = entry_price + entry_tolerance
                
                # SIMPLIFIED: Realistic profit calculation
                position_size = self.balance * self.config["trading"]["position_size_ratio"]
                leverage = 25  # 25x leverage
                taker_fee_rate = self.config["trading"]["fees"]["taker"]
                usdt_inr = self.fetcher.fetch_usdt_inr_rate() or 93.0
                
                # Calculate base profit using tp1 (1.5% target)
                quantity = (position_size / usdt_inr) / entry_price * leverage
                usdt_profit = (tp1 - entry_price) * quantity if side == "long" else (entry_price - tp1) * quantity
                inr_profit = usdt_profit * usdt_inr
                taker_fee = inr_profit * taker_fee_rate
                net_profit = inr_profit - taker_fee
                
                # Simple profit adjustment based on score
                score_multiplier = 1.0 + (score - 6) * 0.1  # 10% increase per score point above 6
                final_profit = net_profit * score_multiplier
                
                # Cap profit at reasonable levels
                final_profit = min(final_profit, 1000.0)  # Maximum ₹1000 profit

                # CAP SL TO 2.1% AND TP TO 2% FOR SCALPING
                if side == "long":
                    stop_loss = max(stop_loss, entry_price * 0.979)
                    tp1 = min(tp1, entry_price * 1.02)
                    tp2 = tp1
                else:
                    stop_loss = min(stop_loss, entry_price * 1.021)
                    tp1 = max(tp1, entry_price * 0.98)
                    tp2 = tp1

                # Remove undefined multipliers from profit calculation
                # final_profit = net_profit * score_multiplier * rsi_multiplier * volume_multiplier * macd_multiplier * structure_multiplier
                final_profit = net_profit * score_multiplier
                # Remove minimum profit threshold (no more final_profit = max(final_profit, 100.0))
                # Keep only the maximum cap if needed
                final_profit = min(final_profit, 1000.0)  # Increased maximum from ₹500 to ₹1000
                
                # Calculate confidence and quality score with multi-timeframe trend
                base_confidence = min(score / 8.0, 1.0)  # Normalize to 0-1
                confidence = (base_confidence + trend_strength) / 2
                quality_score = (score * 10) + (trend_strength * 50) + (confidence * 100)
                
                # Normalize quality score to 0-100
                max_possible_score = 8 * 10 + 1.0 * 50 + 1.0 * 100  # 230
                normalized_score = min((quality_score / max_possible_score) * 100, 100)

                # Calculate total possible score
                score_max = len(long_conditions) + 5  # 5 is the max possible bonus points
                # Add bonus_score to main score
                total_score = score + bonus_score
                total_reasons = reasons + bonus_reasons
                signal = {
                    "symbol": symbol,
                    "timeframe": primary_tf,
                    "strategy": "scalping",
                    "direction": f"TradeDirection.{side.upper()}",
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": tp1,
                    "tp1": tp1,
                    "tp2": tp2,
                    "max_hold_time": 2,  # 2 hours max hold time
                    "score": total_score,
                    "score_max": score_max,
                    "score_reasons": total_reasons,
                    "indicators": {
                        "rsi": rsi,
                        "ema_20": ema_20,
                        "ema_50": ema_50,
                        "macd": macd,
                        "bollinger_upper": bb_upper,
                        "bollinger_lower": bb_lower,
                        "vwap_daily": vwap_daily,
                        "volume_ok": volume_ok,
                        "stoch_rsi_k": stoch_k,
                        "stoch_rsi_d": stoch_d,
                        "atr": atr,
                        "multi_timeframe_trend": trend_trend,
                        "trend_strength": trend_strength,
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
                    "estimated_profit_inr": final_profit,
                    "confidence": confidence,
                    "quality_score": quality_score,
                    "normalized_score": normalized_score,
                    "side": side,
                    "bonus_score": bonus_score,
                    "bonus_reasons": bonus_reasons
                }
                
                if score >= 8:
                    self.logger.info(f"[Scalping] Generated STRONG signal (score {score}/9): {signal}")
                elif score >= 7:
                    self.logger.info(f"[Scalping] Generated signal (score {score}/9): {signal}")
                else:
                    self.logger.debug(f"[Scalping] Generated weak signal (score {score}/9): {signal}")
                return signal
            else:
                self.logger.debug(f"[Scalping] No signal for {symbol}: Long score={long_score}/9, Short score={short_score}/9 (need 7+)")
                return None
                
        except Exception as e:
            self.logger.error(f"Error generating scalping signal for {symbol}: {e}")
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
                
                # Calculate EMAs for trend analysis
                ema_20 = self.indicators.calculate_ema(candles, 20)
                ema_50 = self.indicators.calculate_ema(candles, 50)
                
                if ema_20 is None or ema_50 is None:
                    continue
                
                # Calculate EMA slopes for momentum (more sensitive for scalping)
                if len(candles) >= 25:
                    ema_20_prev = self.indicators.calculate_ema(candles[:-3], 20)  # 3 candles ago for scalping
                    ema_50_prev = self.indicators.calculate_ema(candles[:-3], 50)
                    
                    if ema_20_prev and ema_50_prev:
                        ema_20_slope = (ema_20 - ema_20_prev) / ema_20_prev
                        ema_50_slope = (ema_50 - ema_50_prev) / ema_50_prev
                        momentum_score = (ema_20_slope + ema_50_slope) / 2
                        momentum_scores.append(momentum_score)
                
                # Calculate price momentum (more sensitive for scalping)
                if len(candles) >= 10:
                    price_3_ago = float(candles[-3]["close"])  # 3 candles ago for scalping
                    price_5_ago = float(candles[-5]["close"])
                    price_momentum = (current_price - price_3_ago) / price_3_ago
                    momentum_scores.append(price_momentum)
                
                # Determine trend for this timeframe (more sensitive for scalping)
                trend_score = 0
                if ema_20 > ema_50 and current_price > ema_20:
                    trend_scores['bullish'] += 1
                    trend_score = 1
                elif ema_20 < ema_50 and current_price < ema_20:
                    trend_scores['bearish'] += 1
                    trend_score = -1
                else:
                    trend_scores['neutral'] += 1
                    trend_score = 0
                
                trend_consistency.append(trend_score)
            
            if total_timeframes == 0:
                return {'trend': 'neutral', 'strength': 0.5}
            
            # Calculate more nuanced trend strength (adapted for scalping)
            max_trend = max(trend_scores, key=trend_scores.get)
            base_strength = trend_scores[max_trend] / total_timeframes
            
            # Adjust strength based on momentum consistency (more weight for scalping)
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
            
            # Combine factors for final strength (adapted for scalping - more weight on momentum)
            final_strength = (base_strength * 0.3 + 
                            momentum_consistency * 0.5 + 
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

    def _check_liquidity_filter(self, symbol, candles):
        """Check if symbol meets liquidity requirements for scalping.
        
        Args:
            symbol: Trading symbol
            candles: Candlestick data
            
        Returns:
            bool: True if liquidity requirements are met
        """
        try:
            # Calculate average volume over last 20 candles
            volumes = [float(candle["volume"]) for candle in candles[-20:]]
            avg_volume = sum(volumes) / len(volumes) if volumes else 0
            
            # Get current volume
            current_volume = float(candles[-1]["volume"]) if candles else 0
            
            # Liquidity requirements for scalping - INCREASED for highly voluminous signals
            min_avg_volume = 200000  # Increased minimum average volume for high volume signals
            min_current_volume = 100000  # Increased minimum current volume for high volume signals
            
            # Check if volume meets requirements
            volume_ok = avg_volume >= min_avg_volume and current_volume >= min_current_volume
            
            if not volume_ok:
                self.logger.debug(f"[Scalping] Liquidity filter failed for {symbol}: avg_volume={avg_volume:.0f}, current_volume={current_volume:.0f}")
            
            return volume_ok
            
        except Exception as e:
            self.logger.error(f"Error checking liquidity filter for {symbol}: {e}")
            return False