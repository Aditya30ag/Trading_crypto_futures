from src.data.fetcher import CoinDCXFetcher
from src.data.indicators import TechnicalIndicators
from src.utils.logger import setup_logger
import yaml
import pandas as pd

class TrendStrategy:
    def __init__(self, balance):
        self.fetcher = CoinDCXFetcher()
        self.indicators = TechnicalIndicators()
        with open("config/config.yaml", "r") as file:
            self.config = yaml.safe_load(file)
        self.balance = balance
        self.logger = setup_logger()

    def generate_signal(self, symbol, timeframe):
        self.logger.debug(f"[Trend] generate_signal called for {symbol} {timeframe}")
        try:
            market_data = self.fetcher.fetch_market_data(symbol)
            if not market_data or "last_price" not in market_data:
                self.logger.warning(f"No valid market data for {symbol}")
                return None
            current_price = market_data["last_price"]

            # Get 1-day candles for primary analysis (D1)
            candles_1d = self.fetcher.fetch_candlestick_data(symbol, "1d", limit=120)
            if not candles_1d or len(candles_1d) < 50:
                self.logger.warning(f"Insufficient 1d candles for {symbol}: {len(candles_1d) if candles_1d else 0}")
                return None

            # Calculate indicators for 1-day (primary timeframe)
            rsi = self.indicators.calculate_rsi(candles_1d)
            ema_50 = self.indicators.calculate_ema(candles_1d, 50)
            ema_100 = self.indicators.calculate_ema(candles_1d, 100)
            ema_200 = self.indicators.calculate_ema(candles_1d, 200)
            adx = self.indicators.calculate_adx(candles_1d)
            macd_weekly = self.indicators.calculate_macd_weekly(candles_1d)

            if any(x is None for x in [rsi, ema_50, ema_100, ema_200, adx, macd_weekly]):
                self.logger.warning(f"Indicator failure for {symbol}: rsi={rsi}, ema_50={ema_50}, ema_100={ema_100}, ema_200={ema_200}, adx={adx}, macd_weekly={macd_weekly}")
                return None

            # Volume trend analysis
            volumes = [float(candle["volume"]) for candle in candles_1d]
            avg_volume = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else 1.0
            current_volume = volumes[-1] if volumes else 0
            volume_rising = current_volume > avg_volume

            # Order book and spread checks
            order_book = self.fetcher.fetch_order_book(symbol)
            if not order_book or not all(k in order_book for k in ["best_ask", "best_bid", "bid_vol", "ask_vol"]):
                self.logger.warning(f"Invalid order book for {symbol}: {order_book}")
                return None
            spread = order_book.get("spread", 0)
            spread_pct = (spread / current_price) * 100 if spread and current_price else 100
            spread_limit = 0.5  # 0.5% for trend trading
            if spread_pct > spread_limit:
                self.logger.debug(f"Spread too high for {symbol}: {spread_pct:.3f}% > {spread_limit}%")
                return None

            # --- TREND STRATEGY ENTRY LOGIC ---
            signal = None
            side = None
            
            # LONG TREND CONDITIONS
            long_conditions = []
            
            # 1. Price > 200 EMA (D1) - long-term uptrend
            above_200_ema = current_price > ema_200
            long_conditions.append(("Price > 200 EMA (D1)", above_200_ema))
            
            # 2. EMA 50 > EMA 100 - Golden cross
            golden_cross = ema_50 > ema_100
            long_conditions.append(("EMA 50 > EMA 100 (Golden Cross)", golden_cross))
            
            # 3. RSI > 60 - confirm trend strength
            rsi_bullish = rsi > 60
            long_conditions.append(("RSI > 60", rsi_bullish))
            
            # 4. MACD histogram green and rising
            macd_bullish = macd_weekly["histogram"] > 0
            long_conditions.append(("MACD histogram green", macd_bullish))
            
            # 5. ADX > 25 - confirm trend strength
            adx_strong = adx["adx"] > 25
            long_conditions.append(("ADX > 25", adx_strong))
            
            # 6. Volume rising with price
            long_conditions.append(("Volume rising", volume_rising))
            
            # Check if all long conditions are met
            long_score = sum(1 for _, condition in long_conditions if condition)
            self.logger.debug(f"[Trend] {symbol} {timeframe} Long conditions: {[(name, condition) for name, condition in long_conditions]}")
            
            # SHORT TREND CONDITIONS
            short_conditions = []
            
            # 1. Price < 200 EMA - long-term downtrend
            below_200_ema = current_price < ema_200
            short_conditions.append(("Price < 200 EMA (D1)", below_200_ema))
            
            # 2. EMA 50 < EMA 100 - Death cross
            death_cross = ema_50 < ema_100
            short_conditions.append(("EMA 50 < EMA 100 (Death Cross)", death_cross))
            
            # 3. RSI < 40 - confirm trend strength
            rsi_bearish = rsi < 40
            short_conditions.append(("RSI < 40", rsi_bearish))
            
            # 4. MACD histogram red and falling
            macd_bearish = macd_weekly["histogram"] < 0
            short_conditions.append(("MACD histogram red", macd_bearish))
            
            # 5. ADX > 25 - confirm trend strength
            short_conditions.append(("ADX > 25", adx_strong))
            
            # 6. Volume rising with decline
            short_conditions.append(("Volume rising", volume_rising))
            
            # Check if all short conditions are met
            short_score = sum(1 for _, condition in short_conditions if condition)
            self.logger.debug(f"[Trend] {symbol} {timeframe} Short conditions: {[(name, condition) for name, condition in short_conditions]}")
            
            # Determine trade direction based on scores
            if long_score >= 4 and long_score > short_score:
                side = "long"
                score = long_score
                reasons = [name for name, condition in long_conditions if condition]
                self.logger.debug(f"[Trend] LONG signal conditions met: {reasons}")
            elif short_score >= 4 and short_score > long_score:
                side = "short"
                score = short_score
                reasons = [name for name, condition in short_conditions if condition]
                self.logger.debug(f"[Trend] SHORT signal conditions met: {reasons}")
            else:
                self.logger.debug(f"[Trend] No signal: Long score={long_score}, Short score={short_score}")
                return None

            # Calculate pivot points
            pivots = self.indicators.calculate_pivot_points(candles_1d)

            # Generate signal if conditions are met
            if side and score >= 4:
                entry_price = current_price
                # Default percentage-based exits (increased for larger profits)
                if side == "long":
                    tp1 = entry_price * 1.25  # +25% (was 15%)
                    tp2 = entry_price * 1.50  # +50% (was 40%)
                    stop_loss = entry_price * 0.92  # -8% (was -4%)
                else:
                    tp1 = entry_price * 0.75  # -25% (was -15%)
                    tp2 = entry_price * 0.50  # -50% (was -40%)
                    stop_loss = entry_price * 1.08  # +8% (was +4%)

                # Use pivot points if available and reasonable
                if pivots:
                    if side == "long":
                        if pivots["r1"] > entry_price:
                            tp1 = pivots["r1"]
                        if pivots["r2"] > entry_price and abs(pivots["r2"] - entry_price) < abs(tp2 - entry_price):
                            tp2 = pivots["r2"]
                        if pivots["s1"] < entry_price:
                            stop_loss = pivots["s1"]
                    else:
                        if pivots["s1"] < entry_price:
                            tp1 = pivots["s1"]
                        if pivots["s2"] < entry_price and abs(pivots["s2"] - entry_price) < abs(tp2 - entry_price):
                            tp2 = pivots["s2"]
                        if pivots["r1"] > entry_price:
                            stop_loss = pivots["r1"]

                # Calculate profit using the (possibly pivot-based) tp1 and 25x leverage
                # This ensures estimated_profit_inr reflects the actual exit price logic
                position_size = self.balance * self.config["trading"]["position_size_ratio"]
                leverage = 25  # Force 25x leverage
                taker_fee_rate = self.config["trading"]["fees"]["taker"]
                usdt_inr = self.fetcher.fetch_usdt_inr_rate() or 93.0
                
                quantity = (position_size / usdt_inr) / entry_price * leverage
                usdt_profit = (tp1 - entry_price) * quantity if side == "long" else (entry_price - tp1) * quantity
                inr_profit = usdt_profit * usdt_inr
                taker_fee = inr_profit * taker_fee_rate
                net_profit = inr_profit - taker_fee
                
                signal = {
                    "symbol": symbol,
                    "timeframe": "1d",  # Primary timeframe
                    "strategy": "trend",
                    "side": side,
                    "entry_price": entry_price,
                    "exit_price": tp1,
                    "tp1": tp1,
                    "tp2": tp2,
                    "stop_loss": stop_loss,
                    "max_hold_time": 1680,  # 10 weeks max hold time (1680 hours)
                    "estimated_profit_inr": net_profit,
                    "indicators": {
                        "rsi": rsi,
                        "ema_50": ema_50,
                        "ema_100": ema_100,
                        "ema_200": ema_200,
                        "adx": adx["adx"],
                        "di_plus": adx["di_plus"],
                        "di_minus": adx["di_minus"],
                        "macd_line": macd_weekly["macd_line"],
                        "macd_signal": macd_weekly["signal_line"],
                        "macd_histogram": macd_weekly["histogram"],
                        "volume_rising": volume_rising
                    },
                    "timestamp": self.fetcher.get_ist_timestamp(),
                    "score": score,
                    "score_reasons": reasons
                }
                
                if score >= 5:
                    self.logger.info(f"[Trend] Generated signal: {signal}")
                elif score >= 4:
                    self.logger.info(f"[Trend] Generated signal (score {score}/6): {signal}")
                else:
                    self.logger.debug(f"[Trend] Generated signal (score {score}/6): {signal}")
                return signal
            else:
                self.logger.debug(f"[Trend] No signal for {symbol}: Long score={long_score}, Short score={short_score}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error generating trend signal for {symbol}: {e}")
            return None 