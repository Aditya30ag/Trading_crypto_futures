import numpy as np
import pandas as pd
from src.utils.logger import setup_logger


class TechnicalIndicators:
    def __init__(self):
        self.logger = setup_logger()

    def calculate_rsi(self, data, period=14):
        """Calculate RSI."""
        try:
            closes = [float(candle["close"]) for candle in data]
            df = pd.Series(closes)
            delta = df.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            # Handle division by zero in RSI calculation
            rs = gain / loss
            rs = rs.replace([np.inf, -np.inf], 0)  # Replace infinite values
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1]
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {e}")
            return None

    def calculate_ema(self, data, period):
        """Calculate EMA."""
        try:
            closes = [float(candle["close"]) for candle in data]
            return pd.Series(closes).ewm(span=period, adjust=False).mean().iloc[-1]
        except Exception as e:
            self.logger.error(f"Error calculating EMA: {e}")
            return None

    def calculate_macd(self, data):
        """Calculate MACD."""
        try:
            closes = [float(candle["close"]) for candle in data]
            df = pd.Series(closes)
            ema_12 = df.ewm(span=12, adjust=False).mean()
            ema_26 = df.ewm(span=26, adjust=False).mean()
            macd = ema_12 - ema_26
            return macd.iloc[-1]
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {e}")
            return None

    def calculate_bollinger_bands(self, data, period=20):
        """Calculate Bollinger Bands."""
        try:
            closes = [float(candle["close"]) for candle in data]
            df = pd.Series(closes)
            sma = df.rolling(window=period).mean()
            std = df.rolling(window=period).std()
            upper = sma + 2 * std
            lower = sma - 2 * std
            return upper.iloc[-1], lower.iloc[-1]
        except Exception as e:
            self.logger.error(f"Error calculating Bollinger Bands: {e}")
            return None, None

    def calculate_atr(self, data, period=14):
        """Calculate ATR."""
        try:
            highs = [float(candle["high"]) for candle in data]
            lows = [float(candle["low"]) for candle in data]
            closes = [float(candle["close"]) for candle in data]
            df = pd.DataFrame({"high": highs, "low": lows, "close": closes})
            tr = pd.DataFrame({
                "a": df["high"] - df["low"],
                "b": abs(df["high"] - df["close"].shift()),
                "c": abs(df["low"] - df["close"].shift())
            }).max(axis=1)
            return tr.rolling(window=period).mean().iloc[-1]
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}")
            return None

    def calculate_stoch_rsi(self, data, period=14, smooth_k=3, smooth_d=3):
        """Calculate Stochastic RSI (StochRSI)."""
        try:
            closes = [float(candle["close"]) for candle in data]
            if len(set(closes)) == 1:
                self.logger.warning(
                    "All close prices are identical (flat candles), "
                    "cannot calculate StochRSI."
                )
                return None
            df = pd.Series(closes)
            delta = df.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            # Handle division by zero in RSI calculation
            rs = gain / loss
            rs = rs.replace([np.inf, -np.inf], 0)  # Replace infinite values
            rsi = 100 - (100 / (1 + rs))
            
            # Calculate StochRSI with proper division by zero handling
            rsi_min = rsi.rolling(window=period).min()
            rsi_max = rsi.rolling(window=period).max()
            denominator = rsi_max - rsi_min
            
            # Handle division by zero in StochRSI calculation
            stoch_rsi = np.where(
                denominator != 0,
                (rsi - rsi_min) / denominator,
                0.5  # Default to 0.5 when denominator is zero
            )
            
            stoch_rsi = pd.Series(stoch_rsi, index=rsi.index)
            k = stoch_rsi.rolling(window=smooth_k).mean()
            d = k.rolling(window=smooth_d).mean()
            
            # Fix NaN for %D
            d_value = d.iloc[-1]
            if pd.isna(d_value):
                self.logger.warning(
                    "StochRSI %D is NaN, setting to 0.5 "
                    "(not enough data for smoothing window)"
                )
                d_value = 0.5
            
            k_value = k.iloc[-1]
            if pd.isna(k_value):
                self.logger.warning("StochRSI %K is NaN, setting to 0.5")
                k_value = 0.5
                
            stoch_rsi_value = stoch_rsi.iloc[-1]
            if pd.isna(stoch_rsi_value):
                stoch_rsi_value = 0.5
                
            return {
                "stoch_rsi": stoch_rsi_value,
                "%K": k_value,
                "%D": d_value
            }
        except Exception as e:
            self.logger.error(f"Error calculating Stochastic RSI: {e}")
            return None

    def calculate_vwap(self, data):
        """Calculate VWAP (Volume Weighted Average Price)."""
        try:
            if len(data) < 2:
                return None
            
            # Calculate typical price and volume-weighted sum
            typical_prices = []
            volumes = []
            
            for candle in data:
                high = float(candle["high"])
                low = float(candle["low"])
                close = float(candle["close"])
                volume = float(candle["volume"])
                
                typical_price = (high + low + close) / 3
                typical_prices.append(typical_price)
                volumes.append(volume)
            
            # Calculate VWAP
            volume_price_sum = sum(tp * vol for tp, vol in zip(typical_prices, volumes))
            total_volume = sum(volumes)
            
            if total_volume == 0:
                return None
                
            vwap = volume_price_sum / total_volume
            return vwap
            
        except Exception as e:
            self.logger.error(f"Error calculating VWAP: {e}")
            return None

    def calculate_macd_histogram(self, data):
        """Calculate MACD histogram (MACD line - Signal line)."""
        try:
            closes = [float(candle["close"]) for candle in data]
            df = pd.Series(closes)
            ema_12 = df.ewm(span=12, adjust=False).mean()
            ema_26 = df.ewm(span=26, adjust=False).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            histogram = macd_line - signal_line
            return histogram.iloc[-1]
        except Exception as e:
            self.logger.error(f"Error calculating MACD histogram: {e}")
            return None

    def calculate_macd_swing(self, data):
        """Calculate MACD for swing strategy (24, 52, 9)."""
        try:
            closes = [float(candle["close"]) for candle in data]
            df = pd.Series(closes)
            ema_24 = df.ewm(span=24, adjust=False).mean()
            ema_52 = df.ewm(span=52, adjust=False).mean()
            macd_line = ema_24 - ema_52
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            histogram = macd_line - signal_line
            return {
                "macd_line": macd_line.iloc[-1],
                "signal_line": signal_line.iloc[-1],
                "histogram": histogram.iloc[-1]
            }
        except Exception as e:
            self.logger.error(f"Error calculating MACD swing: {e}")
            return None

    def calculate_macd_long_swing(self, data):
        """Calculate MACD for long swing strategy (36, 78, 18)."""
        try:
            closes = [float(candle["close"]) for candle in data]
            df = pd.Series(closes)
            ema_36 = df.ewm(span=36, adjust=False).mean()
            ema_78 = df.ewm(span=78, adjust=False).mean()
            macd_line = ema_36 - ema_78
            signal_line = macd_line.ewm(span=18, adjust=False).mean()
            histogram = macd_line - signal_line
            return {
                "macd_line": macd_line.iloc[-1],
                "signal_line": signal_line.iloc[-1],
                "histogram": histogram.iloc[-1]
            }
        except Exception as e:
            self.logger.error(f"Error calculating MACD long swing: {e}")
            return None

    def calculate_ichimoku(self, data):
        """Calculate Ichimoku Cloud components."""
        try:
            if len(data) < 52:
                return None
                
            highs = [float(candle["high"]) for candle in data]
            lows = [float(candle["low"]) for candle in data]
            closes = [float(candle["close"]) for candle in data]
            
            df = pd.DataFrame({"high": highs, "low": lows, "close": closes})
            
            # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
            period9_high = df["high"].rolling(window=9).max()
            period9_low = df["low"].rolling(window=9).min()
            tenkan_sen = (period9_high + period9_low) / 2
            
            # Kijun-sen (Base Line): (26-period high + 26-period low)/2
            period26_high = df["high"].rolling(window=26).max()
            period26_low = df["low"].rolling(window=26).min()
            kijun_sen = (period26_high + period26_low) / 2
            
            # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
            
            # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
            period52_high = df["high"].rolling(window=52).max()
            period52_low = df["low"].rolling(window=52).min()
            senkou_span_b = ((period52_high + period52_low) / 2).shift(26)
            
            # Chikou Span (Lagging Span): Close price shifted back 26 periods
            chikou_span = df["close"].shift(-26)
            
            return {
                "tenkan_sen": tenkan_sen.iloc[-1],
                "kijun_sen": kijun_sen.iloc[-1],
                "senkou_span_a": senkou_span_a.iloc[-1],
                "senkou_span_b": senkou_span_b.iloc[-1],
                "chikou_span": chikou_span.iloc[-1]
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Ichimoku: {e}")
            return None

    def calculate_obv(self, data):
        """Calculate On Balance Volume (OBV)."""
        try:
            closes = [float(candle["close"]) for candle in data]
            volumes = [float(candle["volume"]) for candle in data]
            
            df = pd.DataFrame({"close": closes, "volume": volumes})
            
            # Calculate price changes
            df["price_change"] = df["close"].diff()
            
            # Initialize OBV
            obv = [0]
            
            for i in range(1, len(df)):
                if df["price_change"].iloc[i] > 0:
                    # Price went up, add volume
                    obv.append(obv[-1] + df["volume"].iloc[i])
                elif df["price_change"].iloc[i] < 0:
                    # Price went down, subtract volume
                    obv.append(obv[-1] - df["volume"].iloc[i])
                else:
                    # No change, keep previous OBV
                    obv.append(obv[-1])
            
            return obv[-1]
            
        except Exception as e:
            self.logger.error(f"Error calculating OBV: {e}")
            return None

    def calculate_vwap_daily(self, data):
        """Calculate Daily VWAP (Volume Weighted Average Price)."""
        try:
            if len(data) < 24:  # Need at least 24 hours of data
                return None
            
            # Since the API doesn't provide timestamps, we'll use the most recent candles
            # For daily VWAP, we'll use the last 24 candles (assuming 1-hour candles)
            # or the last 1440 candles (assuming 1-minute candles)
            # We'll use a reasonable number based on the data length
            if len(data) >= 1440:  # 1 day of 1-minute candles
                recent_data = data[-1440:]
            elif len(data) >= 24:  # 1 day of 1-hour candles
                recent_data = data[-24:]
            else:
                # Use all available data if we don't have enough
                recent_data = data
            
            return self.calculate_vwap(recent_data)
            
        except Exception as e:
            self.logger.error(f"Error calculating Daily VWAP: {e}")
            # Fallback: use last 24 candles if calculation fails
            try:
                recent_data = data[-24:] if len(data) >= 24 else data
                return self.calculate_vwap(recent_data)
            except Exception as fallback_error:
                self.logger.error(f"Fallback VWAP calculation also failed: {fallback_error}")
                return None

    def calculate_vwap_weekly(self, data):
        """Calculate Weekly VWAP (Volume Weighted Average Price)."""
        try:
            if len(data) < 7:  # Need at least 7 days of data
                return None
            
            # Since the API doesn't provide timestamps, we'll use the most recent candles
            # For weekly VWAP, we'll use the last 7 candles (assuming 1-day candles)
            # or the last 10080 candles (assuming 1-minute candles)
            # We'll use a reasonable number based on the data length
            if len(data) >= 10080:  # 1 week of 1-minute candles
                recent_data = data[-10080:]
            elif len(data) >= 168:  # 1 week of 1-hour candles
                recent_data = data[-168:]
            elif len(data) >= 7:  # 1 week of 1-day candles
                recent_data = data[-7:]
            else:
                # Use all available data if we don't have enough
                recent_data = data
            
            return self.calculate_vwap(recent_data)
            
        except Exception as e:
            self.logger.error(f"Error calculating Weekly VWAP: {e}")
            # Fallback: use last 7 candles if calculation fails
            try:
                recent_data = data[-7:] if len(data) >= 7 else data
                return self.calculate_vwap(recent_data)
            except Exception as fallback_error:
                self.logger.error(f"Fallback VWAP calculation also failed: {fallback_error}")
                return None

    def calculate_adx(self, data, period=14):
        """Calculate ADX (Average Directional Index)."""
        try:
            if len(data) < period + 1:
                return None
                
            highs = [float(candle["high"]) for candle in data]
            lows = [float(candle["low"]) for candle in data]
            closes = [float(candle["close"]) for candle in data]
            
            df = pd.DataFrame({"high": highs, "low": lows, "close": closes})
            
            # Calculate True Range (TR)
            df["tr1"] = df["high"] - df["low"]
            df["tr2"] = abs(df["high"] - df["close"].shift())
            df["tr3"] = abs(df["low"] - df["close"].shift())
            df["tr"] = df[["tr1", "tr2", "tr3"]].max(axis=1)
            
            # Calculate Directional Movement (DM)
            df["dm_plus"] = np.where(
                (df["high"] - df["high"].shift()) > (df["low"].shift() - df["low"]),
                np.maximum(df["high"] - df["high"].shift(), 0),
                0
            )
            df["dm_minus"] = np.where(
                (df["low"].shift() - df["low"]) > (df["high"] - df["high"].shift()),
                np.maximum(df["low"].shift() - df["low"], 0),
                0
            )
            
            # Smooth TR and DM using Wilder's smoothing
            df["tr_smooth"] = df["tr"].rolling(window=period).mean()
            df["dm_plus_smooth"] = df["dm_plus"].rolling(window=period).mean()
            df["dm_minus_smooth"] = df["dm_minus"].rolling(window=period).mean()
            
            # Calculate Directional Indicators (DI) with division by zero handling
            df["di_plus"] = np.where(
                df["tr_smooth"] != 0,
                100 * (df["dm_plus_smooth"] / df["tr_smooth"]),
                0
            )
            df["di_minus"] = np.where(
                df["tr_smooth"] != 0,
                100 * (df["dm_minus_smooth"] / df["tr_smooth"]),
                0
            )
            
            # Calculate Directional Index (DX) with division by zero handling
            denominator = df["di_plus"] + df["di_minus"]
            df["dx"] = np.where(
                denominator != 0,
                100 * abs(df["di_plus"] - df["di_minus"]) / denominator,
                0
            )
            
            # Calculate ADX (Average of DX)
            adx = df["dx"].rolling(window=period).mean()
            
            return {
                "adx": adx.iloc[-1],
                "di_plus": df["di_plus"].iloc[-1],
                "di_minus": df["di_minus"].iloc[-1]
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating ADX: {e}")
            return None

    def calculate_macd_weekly(self, data):
        """Calculate MACD for weekly timeframe."""
        try:
            closes = [float(candle["close"]) for candle in data]
            df = pd.Series(closes)
            ema_12 = df.ewm(span=12, adjust=False).mean()
            ema_26 = df.ewm(span=26, adjust=False).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            histogram = macd_line - signal_line
            return {
                "macd_line": macd_line.iloc[-1],
                "signal_line": signal_line.iloc[-1],
                "histogram": histogram.iloc[-1]
            }
        except Exception as e:
            self.logger.error(f"Error calculating MACD weekly: {e}")
            return None

    def calculate_pivot_points(self, data):
        """Calculate classic pivot points and support/resistance levels from the previous period's candle(s)."""
        try:
            # Use the last completed candle (previous period)
            if len(data) < 2:
                return None
            prev_candle = data[-2]
            high = float(prev_candle["high"])
            low = float(prev_candle["low"])
            close = float(prev_candle["close"])

            pivot = (high + low + close) / 3
            r1 = 2 * pivot - low
            s1 = 2 * pivot - high
            r2 = pivot + (high - low)
            s2 = pivot - (high - low)

            return {
                "pivot": pivot,
                "r1": r1,
                "s1": s1,
                "r2": r2,
                "s2": s2
            }
        except Exception as e:
            self.logger.error(f"Error calculating pivot points: {e}")
            return None

    def calculate_ema100(self, data):
        """Calculate EMA100."""
        return self.calculate_ema(data, period=100)

    def calculate_supertrend(self, data, period=10, multiplier=3):
        """Calculate Supertrend indicator."""
        try:
            highs = [float(candle["high"]) for candle in data]
            lows = [float(candle["low"]) for candle in data]
            closes = [float(candle["close"]) for candle in data]
            df = pd.DataFrame({"high": highs, "low": lows, "close": closes})
            atr = df['high'].rolling(window=period).max() - df['low'].rolling(window=period).min()
            atr = atr.rolling(window=period).mean().fillna(0)
            hl2 = (df['high'] + df['low']) / 2
            basic_upperband = hl2 + (multiplier * atr)
            basic_lowerband = hl2 - (multiplier * atr)
            final_upperband = basic_upperband.copy()
            final_lowerband = basic_lowerband.copy()
            for i in range(1, len(df)):
                if closes[i-1] > final_upperband[i-1]:
                    final_upperband[i] = min(basic_upperband[i], final_upperband[i-1])
                else:
                    final_upperband[i] = basic_upperband[i]
                if closes[i-1] < final_lowerband[i-1]:
                    final_lowerband[i] = max(basic_lowerband[i], final_lowerband[i-1])
                else:
                    final_lowerband[i] = basic_lowerband[i]
            supertrend = [np.nan] * len(df)
            in_uptrend = True
            for i in range(1, len(df)):
                if closes[i] > final_upperband[i-1]:
                    in_uptrend = True
                elif closes[i] < final_lowerband[i-1]:
                    in_uptrend = False
                supertrend[i] = final_lowerband[i] if in_uptrend else final_upperband[i]
            return {
                "supertrend": supertrend[-1],
                "in_uptrend": in_uptrend
            }
        except Exception as e:
            self.logger.error(f"Error calculating Supertrend: {e}")
            return None

    def calculate_cci(self, data, period=20):
        """Calculate Commodity Channel Index (CCI)."""
        try:
            highs = [float(candle["high"]) for candle in data]
            lows = [float(candle["low"]) for candle in data]
            closes = [float(candle["close"]) for candle in data]
            typical_price = (np.array(highs) + np.array(lows) + np.array(closes)) / 3
            tp_series = pd.Series(typical_price)
            sma = tp_series.rolling(window=period).mean()
            mad = tp_series.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
            
            # Handle division by zero in CCI calculation
            cci = np.where(
                mad != 0,
                (tp_series - sma) / (0.015 * mad),
                0  # Default to 0 when MAD is zero
            )
            
            # Convert to pandas Series to use iloc, then get the last value
            cci_series = pd.Series(cci)
            return cci_series.iloc[-1]
        except Exception as e:
            self.logger.error(f"Error calculating CCI: {e}")
            return None

    def calculate_momentum(self, data, period=10):
        """Calculate Momentum Oscillator (rate of change)."""
        try:
            closes = [float(candle["close"]) for candle in data]
            if len(closes) < period + 1:
                return None
            momentum = closes[-1] - closes[-1 - period]
            return momentum
        except Exception as e:
            self.logger.error(f"Error calculating Momentum Oscillator: {e}")
            return None

    def calculate_keltner_channels(self, data, period=20, multiplier=2):
        """Calculate Keltner Channels."""
        try:
            highs = [float(candle["high"]) for candle in data]
            lows = [float(candle["low"]) for candle in data]
            closes = [float(candle["close"]) for candle in data]
            df = pd.DataFrame({"high": highs, "low": lows, "close": closes})
            typical_price = (df["high"] + df["low"] + df["close"]) / 3
            ema = typical_price.ewm(span=period, adjust=False).mean()
            tr = pd.DataFrame({
                "a": df["high"] - df["low"],
                "b": abs(df["high"] - df["close"].shift()),
                "c": abs(df["low"] - df["close"].shift())
            }).max(axis=1)
            atr = tr.rolling(window=period).mean()
            upper = ema + multiplier * atr
            lower = ema - multiplier * atr
            return upper.iloc[-1], lower.iloc[-1]
        except Exception as e:
            self.logger.error(f"Error calculating Keltner Channels: {e}")
            return None, None

    def calculate_fibonacci_levels(self, data):
        """Calculate Fibonacci retracement levels from the most recent swing high/low."""
        try:
            highs = [float(candle["high"]) for candle in data]
            lows = [float(candle["low"]) for candle in data]
            recent_high = max(highs[-20:])
            recent_low = min(lows[-20:])
            diff = recent_high - recent_low
            levels = {
                "0.0%": recent_high,
                "23.6%": recent_high - 0.236 * diff,
                "38.2%": recent_high - 0.382 * diff,
                "50.0%": recent_high - 0.5 * diff,
                "61.8%": recent_high - 0.618 * diff,
                "78.6%": recent_high - 0.786 * diff,
                "100.0%": recent_low
            }
            return levels
        except Exception as e:
            self.logger.error(f"Error calculating Fibonacci levels: {e}")
            return None