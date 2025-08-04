#!/usr/bin/env python3
"""
Dynamic Risk Manager for Optimized Hold Times and Stop Loss Management
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from src.utils.logger import setup_logger
from src.data.indicators import TechnicalIndicators


class DynamicRiskManager:
    """
    Advanced risk management system with dynamic hold times and ATR-based stops.
    Prevents signals with excessive hold times and optimizes stop loss placement.
    """
    
    def __init__(self):
        self.logger = setup_logger()
        self.indicators = TechnicalIndicators()
    
    def calculate_optimal_hold_time(self, symbol: str, candles: List[Dict], strategy: str, timeframe: str) -> Dict:
        """
        Calculate optimal hold time based on market conditions and volatility.
        
        Args:
            symbol: Trading symbol
            candles: Candlestick data
            strategy: Trading strategy (scalping, swing, long_swing)
            timeframe: Chart timeframe
            
        Returns:
            Dict with optimal hold time and reasoning
        """
        try:
            self.logger.info(f"⏰ Calculating optimal hold time for {symbol} ({strategy}, {timeframe})")
            
            if len(candles) < 50:
                return {"hold_time_hours": 2, "reason": "Insufficient data - using default", "confidence": 0.3}
            
            # Calculate market volatility
            atr = self.indicators.calculate_atr(candles)
            current_price = float(candles[-1]["close"])
            atr_pct = (atr / current_price) * 100 if atr and current_price > 0 else 2.0
            
            # Calculate price momentum and trend strength
            closes = [float(candle["close"]) for candle in candles[-20:]]
            price_momentum = (closes[-1] - closes[-10]) / closes[-10] * 100 if len(closes) >= 10 else 0
            
            # Calculate EMA trend strength
            ema_20 = self.indicators.calculate_ema(candles, 20)
            ema_50 = self.indicators.calculate_ema(candles, 50)
            trend_strength = abs((ema_20 - ema_50) / ema_50) * 100 if ema_20 and ema_50 and ema_50 > 0 else 0
            
            # Calculate volume consistency
            volumes = [float(candle["volume"]) for candle in candles[-20:]]
            volume_cv = np.std(volumes) / np.mean(volumes) if volumes and np.mean(volumes) > 0 else 2.0
            
            # Calculate RSI for overbought/oversold conditions
            rsi = self.indicators.calculate_rsi(candles)
            rsi_extreme = abs(rsi - 50) if rsi else 0  # Distance from neutral
            
            # Base hold times by strategy (in hours)
            base_hold_times = {
                "scalping": {"5m": 0.5, "15m": 1.0, "30m": 1.5},
                "swing": {"5m": 2.0, "15m": 3.0, "30m": 4.0, "1h": 6.0},
                "long_swing": {"15m": 4.0, "30m": 6.0, "1h": 8.0, "4h": 12.0}
            }
            
            base_time = base_hold_times.get(strategy, {}).get(timeframe, 2.0)
            
            # Dynamic adjustments based on market conditions
            adjustments = []
            multiplier = 1.0
            
            # 1. Volatility adjustment
            if atr_pct > 5.0:  # High volatility
                multiplier *= 0.7  # Reduce hold time
                adjustments.append(f"High volatility ({atr_pct:.1f}%): -30% hold time")
            elif atr_pct < 1.5:  # Low volatility
                multiplier *= 1.3  # Increase hold time
                adjustments.append(f"Low volatility ({atr_pct:.1f}%): +30% hold time")
            
            # 2. Trend strength adjustment
            if trend_strength > 2.0:  # Strong trend
                multiplier *= 1.2  # Increase hold time for trend following
                adjustments.append(f"Strong trend ({trend_strength:.1f}%): +20% hold time")
            elif trend_strength < 0.5:  # Weak trend
                multiplier *= 0.8  # Reduce hold time
                adjustments.append(f"Weak trend ({trend_strength:.1f}%): -20% hold time")
            
            # 3. Volume consistency adjustment
            if volume_cv > 1.5:  # Inconsistent volume
                multiplier *= 0.8  # Reduce hold time
                adjustments.append(f"Volume inconsistent (CV={volume_cv:.2f}): -20% hold time")
            elif volume_cv < 0.5:  # Very consistent volume
                multiplier *= 1.1  # Slight increase
                adjustments.append(f"Volume consistent (CV={volume_cv:.2f}): +10% hold time")
            
            # 4. RSI extreme conditions
            if rsi_extreme > 30:  # Very overbought/oversold
                multiplier *= 0.6  # Significantly reduce hold time
                adjustments.append(f"RSI extreme ({rsi:.1f}): -40% hold time")
            elif rsi_extreme > 20:  # Moderately extreme
                multiplier *= 0.8  # Moderately reduce
                adjustments.append(f"RSI moderate extreme ({rsi:.1f}): -20% hold time")
            
            # 5. Momentum adjustment
            if abs(price_momentum) > 10:  # Strong momentum
                multiplier *= 0.75  # Reduce hold time (momentum can reverse quickly)
                adjustments.append(f"Strong momentum ({price_momentum:.1f}%): -25% hold time")
            
            # Calculate final hold time
            optimal_hold_time = base_time * multiplier
            
            # Apply strategy-specific limits
            limits = {
                "scalping": {"min": 0.25, "max": 2.0},      # 15 min to 2 hours
                "swing": {"min": 1.0, "max": 8.0},          # 1 to 8 hours
                "long_swing": {"min": 2.0, "max": 16.0}     # 2 to 16 hours (reduced from 24)
            }
            
            strategy_limits = limits.get(strategy, limits["swing"])
            optimal_hold_time = max(strategy_limits["min"], min(optimal_hold_time, strategy_limits["max"]))
            
            # Calculate confidence based on data quality
            confidence = 0.5  # Base confidence
            if len(candles) >= 100:
                confidence += 0.2
            if atr and ema_20 and ema_50:
                confidence += 0.2
            if volume_cv < 1.0:
                confidence += 0.1
            
            confidence = min(confidence, 1.0)
            
            result = {
                "hold_time_hours": round(optimal_hold_time, 2),
                "base_time": base_time,
                "multiplier": round(multiplier, 2),
                "adjustments": adjustments,
                "market_conditions": {
                    "atr_pct": round(atr_pct, 2),
                    "trend_strength": round(trend_strength, 2),
                    "volume_cv": round(volume_cv, 2),
                    "rsi": round(rsi, 1) if rsi else None,
                    "price_momentum": round(price_momentum, 2)
                },
                "confidence": round(confidence, 2),
                "reason": f"Dynamic calculation: {base_time}h * {multiplier:.2f} = {optimal_hold_time:.2f}h"
            }
            
            self.logger.info(f"⏰ {symbol} Optimal hold time: {optimal_hold_time:.2f}h (confidence: {confidence:.1%})")
            for adj in adjustments:
                self.logger.info(f"   📊 {adj}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Hold time calculation error for {symbol}: {e}")
            return {"hold_time_hours": 2, "reason": f"Error: {e}", "confidence": 0.1}
    
    def calculate_dynamic_stop_loss(self, symbol: str, candles: List[Dict], entry_price: float, 
                                   side: str, strategy: str) -> Dict:
        """
        Calculate dynamic ATR-based stop loss instead of fixed percentage.
        
        Args:
            symbol: Trading symbol
            candles: Candlestick data
            entry_price: Signal entry price
            side: Trade direction (long/short)
            strategy: Trading strategy
            
        Returns:
            Dict with stop loss analysis and levels
        """
        try:
            self.logger.info(f"🛡️ Calculating dynamic stop loss for {symbol} ({side} {strategy})")
            
            if len(candles) < 20:
                # Fallback to conservative fixed stops
                fallback_sl = entry_price * (0.985 if side == "long" else 1.015)
                return {
                    "stop_loss": fallback_sl,
                    "method": "Fixed fallback",
                    "risk_pct": 1.5,
                    "confidence": 0.3,
                    "reason": "Insufficient data for ATR calculation"
                }
            
            # Calculate ATR for volatility-based stops
            atr = self.indicators.calculate_atr(candles)
            if not atr:
                # Alternative volatility calculation
                closes = [float(candle["close"]) for candle in candles[-20:]]
                price_std = np.std(closes)
                atr = price_std  # Use price standard deviation as proxy
            
            # Calculate support/resistance levels
            highs = [float(candle["high"]) for candle in candles[-20:]]
            lows = [float(candle["low"]) for candle in candles[-20:]]
            
            recent_high = max(highs)
            recent_low = min(lows)
            
            # Strategy-specific ATR multipliers
            atr_multipliers = {
                "scalping": {"conservative": 1.0, "normal": 1.5, "aggressive": 2.0},
                "swing": {"conservative": 1.5, "normal": 2.0, "aggressive": 2.5},
                "long_swing": {"conservative": 2.0, "normal": 2.5, "aggressive": 3.0}
            }
            
            multipliers = atr_multipliers.get(strategy, atr_multipliers["swing"])
            
            # Calculate market regime (trending vs ranging)
            ema_20 = self.indicators.calculate_ema(candles, 20)
            ema_50 = self.indicators.calculate_ema(candles, 50)
            
            if ema_20 and ema_50:
                trend_strength = abs((ema_20 - ema_50) / ema_50) * 100
                is_trending = trend_strength > 1.0
            else:
                is_trending = False
            
            # Choose multiplier based on market conditions
            if is_trending:
                atr_mult = multipliers["normal"]  # Normal stops in trending market
                regime = "Trending"
            else:
                atr_mult = multipliers["conservative"]  # Tighter stops in ranging market
                regime = "Ranging"
            
            # Calculate ATR-based stop loss
            if side == "long":
                atr_stop = entry_price - (atr * atr_mult)
                # Don't place stop above recent low
                support_stop = recent_low * 0.998  # 0.2% below recent low
                stop_loss = max(atr_stop, support_stop)
            else:  # short
                atr_stop = entry_price + (atr * atr_mult)
                # Don't place stop below recent high
                resistance_stop = recent_high * 1.002  # 0.2% above recent high
                stop_loss = min(atr_stop, resistance_stop)
            
            # Calculate risk percentage
            risk_pct = abs((stop_loss - entry_price) / entry_price) * 100
            
            # Risk validation and adjustment
            max_risk_limits = {
                "scalping": 2.5,    # 2.5% max risk
                "swing": 3.5,       # 3.5% max risk
                "long_swing": 4.5   # 4.5% max risk
            }
            
            max_risk = max_risk_limits.get(strategy, 3.5)
            
            if risk_pct > max_risk:
                # Adjust stop loss to respect max risk
                if side == "long":
                    stop_loss = entry_price * (1 - max_risk / 100)
                else:
                    stop_loss = entry_price * (1 + max_risk / 100)
                risk_pct = max_risk
                adjustment = f"Adjusted to max risk {max_risk}%"
            else:
                adjustment = "No adjustment needed"
            
            # Calculate confidence based on data quality
            confidence = 0.6  # Base confidence for ATR method
            if len(candles) >= 50:
                confidence += 0.2
            if is_trending:
                confidence += 0.1  # More confident in trending markets
            if atr and atr > 0:
                confidence += 0.1
            
            confidence = min(confidence, 1.0)
            
            result = {
                "stop_loss": round(stop_loss, 6),
                "method": f"ATR-based ({atr_mult}x)",
                "risk_pct": round(risk_pct, 2),
                "atr_value": round(atr, 6) if atr else None,
                "atr_multiplier": atr_mult,
                "market_regime": regime,
                "is_trending": is_trending,
                "support_resistance": {
                    "recent_high": recent_high,
                    "recent_low": recent_low,
                },
                "adjustment": adjustment,
                "confidence": round(confidence, 2),
                "reason": f"ATR ({atr:.6f}) * {atr_mult} = {risk_pct:.2f}% risk"
            }
            
            self.logger.info(f"🛡️ {symbol} Dynamic SL: {stop_loss:.6f} ({risk_pct:.2f}% risk)")
            self.logger.info(f"   📊 Method: {result['method']} in {regime} market")
            self.logger.info(f"   🎯 Confidence: {confidence:.1%}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Dynamic stop loss error for {symbol}: {e}")
            # Safe fallback
            fallback_sl = entry_price * (0.985 if side == "long" else 1.015)
            return {
                "stop_loss": fallback_sl,
                "method": "Error fallback",
                "risk_pct": 1.5,
                "confidence": 0.1,
                "reason": f"Error: {e}"
            }
    
    def calculate_dynamic_take_profit(self, symbol: str, candles: List[Dict], entry_price: float,
                                    stop_loss: float, side: str, strategy: str) -> Dict:
        """
        Calculate dynamic take profit levels based on ATR and risk-reward ratios.
        
        Args:
            symbol: Trading symbol
            candles: Candlestick data
            entry_price: Signal entry price
            stop_loss: Calculated stop loss level
            side: Trade direction
            strategy: Trading strategy
            
        Returns:
            Dict with take profit analysis and levels
        """
        try:
            self.logger.info(f"🎯 Calculating dynamic take profit for {symbol}")
            
            # Calculate risk amount
            risk_amount = abs(entry_price - stop_loss)
            
            # Strategy-specific risk-reward ratios
            risk_reward_ratios = {
                "scalping": {"tp1": 1.5, "tp2": 2.0},
                "swing": {"tp1": 2.0, "tp2": 3.0},
                "long_swing": {"tp1": 2.5, "tp2": 4.0}
            }
            
            ratios = risk_reward_ratios.get(strategy, risk_reward_ratios["swing"])
            
            # Calculate ATR for volatility-based adjustments
            atr = self.indicators.calculate_atr(candles) if len(candles) >= 20 else None
            
            # Calculate resistance/support levels
            highs = [float(candle["high"]) for candle in candles[-20:]]
            lows = [float(candle["low"]) for candle in candles[-20:]]
            
            # Base take profit levels
            if side == "long":
                tp1_base = entry_price + (risk_amount * ratios["tp1"])
                tp2_base = entry_price + (risk_amount * ratios["tp2"])
                
                # Adjust for resistance levels
                recent_resistance = max(highs)
                if tp1_base > recent_resistance and recent_resistance > entry_price:
                    tp1_adjusted = recent_resistance * 0.998  # Slightly below resistance
                else:
                    tp1_adjusted = tp1_base
                    
                if tp2_base > recent_resistance and recent_resistance > tp1_adjusted:
                    tp2_adjusted = recent_resistance * 0.995
                else:
                    tp2_adjusted = tp2_base
                    
            else:  # short
                tp1_base = entry_price - (risk_amount * ratios["tp1"])
                tp2_base = entry_price - (risk_amount * ratios["tp2"])
                
                # Adjust for support levels
                recent_support = min(lows)
                if tp1_base < recent_support and recent_support < entry_price:
                    tp1_adjusted = recent_support * 1.002  # Slightly above support
                else:
                    tp1_adjusted = tp1_base
                    
                if tp2_base < recent_support and recent_support > tp1_adjusted:
                    tp2_adjusted = recent_support * 1.005
                else:
                    tp2_adjusted = tp2_base
            
            # Calculate final risk-reward ratios
            actual_rr_tp1 = abs(tp1_adjusted - entry_price) / risk_amount
            actual_rr_tp2 = abs(tp2_adjusted - entry_price) / risk_amount
            
            result = {
                "tp1": round(tp1_adjusted, 6),
                "tp2": round(tp2_adjusted, 6),
                "risk_reward_tp1": round(actual_rr_tp1, 2),
                "risk_reward_tp2": round(actual_rr_tp2, 2),
                "risk_amount": round(risk_amount, 6),
                "adjustments": {
                    "tp1_adjusted": tp1_adjusted != tp1_base,
                    "tp2_adjusted": tp2_adjusted != tp2_base,
                    "resistance_level": max(highs) if side == "long" else None,
                    "support_level": min(lows) if side == "short" else None
                },
                "method": "ATR-based with S/R adjustment",
                "confidence": 0.8
            }
            
            self.logger.info(f"🎯 {symbol} TP levels:")
            self.logger.info(f"   📍 TP1: {tp1_adjusted:.6f} (R:R {actual_rr_tp1:.1f}:1)")
            self.logger.info(f"   📍 TP2: {tp2_adjusted:.6f} (R:R {actual_rr_tp2:.1f}:1)")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Dynamic take profit error for {symbol}: {e}")
            # Fallback calculation
            risk_amount = abs(entry_price - stop_loss)
            tp1 = entry_price + (risk_amount * 2) if side == "long" else entry_price - (risk_amount * 2)
            tp2 = entry_price + (risk_amount * 3) if side == "long" else entry_price - (risk_amount * 3)
            
            return {
                "tp1": round(tp1, 6),
                "tp2": round(tp2, 6),
                "risk_reward_tp1": 2.0,
                "risk_reward_tp2": 3.0,
                "method": "Fallback 2:1 and 3:1",
                "confidence": 0.3
            }