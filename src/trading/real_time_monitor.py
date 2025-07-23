import time
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd

from src.data.fetcher import CoinDCXFetcher
from src.data.indicators import TechnicalIndicators
from src.utils.logger import setup_logger


class TradeStatus(Enum):
    ACTIVE = "active"
    EXITED = "exited"
    STOPPED = "stopped"
    TARGET_HIT = "target_hit"


class ExitReason(Enum):
    STOP_LOSS = "stop_loss"
    TARGET_HIT = "target_hit"
    SIGNAL_WEAKENING = "signal_weakening"
    TREND_REVERSAL = "trend_reversal"
    MANUAL = "manual"
    TIME_EXIT = "time_exit"
    VOLATILITY_EXIT = "volatility_exit"


@dataclass
class IndicatorState:
    """Track the state and journey of technical indicators"""
    rsi: float
    macd_line: float
    macd_signal: float
    macd_histogram: float
    stoch_rsi_k: float
    stoch_rsi_d: float
    atr: float
    supertrend: float
    supertrend_direction: str  # "bullish" or "bearish"
    ema_20: float
    ema_50: float
    bb_upper: float
    bb_lower: float
    volume: float
    timestamp: datetime
    
    def to_dict(self):
        return asdict(self)


@dataclass
class TradeMetrics:
    """Track trade performance metrics"""
    entry_price: float
    current_price: float
    highest_price: float
    lowest_price: float
    current_stop_loss: float
    current_target: float
    profit_loss: float
    profit_loss_percent: float
    max_profit_percent: float
    max_drawdown_percent: float
    time_in_trade: timedelta
    volatility_score: float
    signal_strength: float
    trend_alignment: float
    
    def to_dict(self):
        return asdict(self)


class RealTimeTradeMonitor:
    """
    Advanced real-time monitoring system for active trades with intelligent
    stop-loss and exit management based on indicator journeys.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = setup_logger()
        self.fetcher = CoinDCXFetcher()
        self.indicators = TechnicalIndicators()
        
        # Active trades tracking
        self.active_trades: Dict[str, Dict] = {}
        self.trade_history: List[Dict] = []
        
        # Monitoring settings
        self.monitoring_interval = 5  # seconds - more frequent for real-time
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Risk management parameters
        self.max_active_trades = 10
        self.max_drawdown_percent = 3.0
        self.max_time_in_trade = timedelta(hours=24)  # Max 24 hours per trade
        
        # Dynamic stop-loss settings
        self.trailing_stop_enabled = True
        self.atr_multiplier = 2.0
        self.profit_lock_threshold = 1.0  # Start locking profits at 1%
        self.breakeven_threshold = 0.5  # Move to breakeven at 0.5%
        
        # Signal strength thresholds
        self.signal_weakening_threshold = 0.3
        self.trend_reversal_threshold = 0.2
        
        # Indicator journey tracking
        self.indicator_history_length = 50  # Keep last 50 indicator readings
        
        self.logger.info("RealTimeTradeMonitor initialized")

    def start_trade_monitoring(self, trade_signal: Dict) -> str:
        """
        Start monitoring a new trade based on the entry signal.
        
        Args:
            trade_signal: Dictionary containing trade entry information
            
        Returns:
            trade_id: Unique identifier for the trade
        """
        try:
            trade_id = f"{trade_signal['symbol']}_{trade_signal['strategy']}_{int(time.time())}"
            
            if len(self.active_trades) >= self.max_active_trades:
                self.logger.warning(f"Maximum active trades reached ({self.max_active_trades})")
                return None
            
            # Initialize trade monitoring data
            trade_data = {
                "trade_id": trade_id,
                "signal": trade_signal,
                "status": TradeStatus.ACTIVE,
                "entry_time": datetime.now(),
                "last_update": datetime.now(),
                
                # Price tracking
                "entry_price": trade_signal["entry_price"],
                "current_price": trade_signal["entry_price"],
                "highest_price": trade_signal["entry_price"],
                "lowest_price": trade_signal["entry_price"],
                
                # Risk management
                "original_stop_loss": trade_signal["stop_loss"],
                "current_stop_loss": trade_signal["stop_loss"],
                "original_target": trade_signal["exit_price"],
                "current_target": trade_signal["exit_price"],
                
                # Performance metrics
                "profit_loss": 0.0,
                "profit_loss_percent": 0.0,
                "max_profit_percent": 0.0,
                "max_drawdown_percent": 0.0,
                
                # Indicator tracking
                "indicator_history": [],
                "current_indicators": None,
                "signal_strength": 1.0,
                "trend_alignment": 1.0,
                
                # Exit management
                "exit_reason": None,
                "exit_time": None,
                "exit_price": None,
                
                # Dynamic adjustments
                "trailing_stop_activated": False,
                "breakeven_stop_activated": False,
                "profit_lock_activated": False,
                "dynamic_exit_suggestions": [],
                
                # Volatility tracking
                "volatility_score": 0.0,
                "atr_values": [],
                
                # Time management
                "time_in_trade": timedelta(0),
                "last_volatility_check": datetime.now()
            }
            
            self.active_trades[trade_id] = trade_data
            self.logger.info(f"Started monitoring trade: {trade_id} for {trade_signal['symbol']}")
            
            # Start monitoring if not already active
            if not self.monitoring_active:
                self.start_monitoring()
            
            return trade_id
            
        except Exception as e:
            self.logger.error(f"Error starting trade monitoring: {e}")
            return None

    def start_monitoring(self):
        """Start the real-time monitoring loop in a separate thread."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Real-time monitoring started")

    def stop_monitoring(self):
        """Stop the monitoring loop."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("Real-time monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop that runs continuously."""
        self.logger.info("Starting real-time monitoring loop...")
        
        while self.monitoring_active:
            try:
                # Process each active trade
                for trade_id in list(self.active_trades.keys()):
                    if trade_id in self.active_trades:
                        self._process_trade(trade_id)
                
                # Check if any trades need to be closed
                self._check_exit_conditions()
                
                # Update trade metrics
                self._update_trade_metrics()
                
                # Save monitoring data periodically
                if self.active_trades:
                    self._save_monitoring_data()
                
                # Wait before next cycle
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)
        
        self.logger.info("Real-time monitoring loop ended")

    def _process_trade(self, trade_id: str):
        """Process a single trade with real-time data and indicators."""
        trade_data = self.active_trades[trade_id]
        symbol = trade_data["signal"]["symbol"]
        
        try:
            # Fetch current market data
            market_data = self.fetcher.fetch_market_data(symbol)
            if not market_data:
                self.logger.warning(f"Could not fetch market data for {symbol}")
                return
            
            current_price = market_data["last_price"]
            trade_data["current_price"] = current_price
            trade_data["last_update"] = datetime.now()
            
            # Update price tracking
            self._update_price_tracking(trade_data, current_price)
            
            # Calculate current indicators
            indicators = self._calculate_current_indicators(symbol)
            if indicators:
                trade_data["current_indicators"] = indicators
                self._update_indicator_history(trade_data, indicators)
                
                # Analyze indicator journey
                self._analyze_indicator_journey(trade_data)
                
                # Update dynamic stop-loss and targets
                self._update_dynamic_risk_management(trade_data, indicators)
                
                # Generate exit suggestions
                suggestions = self._generate_exit_suggestions(trade_data, indicators)
                trade_data["dynamic_exit_suggestions"] = suggestions
            
            # Update time in trade
            trade_data["time_in_trade"] = datetime.now() - trade_data["entry_time"]
            
        except Exception as e:
            self.logger.error(f"Error processing trade {trade_id}: {e}")

    def _update_price_tracking(self, trade_data: Dict, current_price: float):
        """Update price tracking metrics."""
        side = trade_data["signal"]["side"]
        entry_price = trade_data["entry_price"]
        
        # Update highest/lowest prices
        if side == "long":
            trade_data["highest_price"] = max(trade_data["highest_price"], current_price)
            trade_data["lowest_price"] = min(trade_data["lowest_price"], current_price)
            profit_loss = current_price - entry_price
        else:
            trade_data["highest_price"] = max(trade_data["highest_price"], current_price)
            trade_data["lowest_price"] = min(trade_data["lowest_price"], current_price)
            profit_loss = entry_price - current_price
        
        profit_loss_percent = (profit_loss / entry_price) * 100
        
        trade_data["profit_loss"] = profit_loss
        trade_data["profit_loss_percent"] = profit_loss_percent
        
        # Update max profit and drawdown
        if profit_loss_percent > trade_data["max_profit_percent"]:
            trade_data["max_profit_percent"] = profit_loss_percent
        
        if side == "long":
            current_drawdown = ((trade_data["highest_price"] - current_price) / trade_data["highest_price"]) * 100
        else:
            current_drawdown = ((current_price - trade_data["lowest_price"]) / trade_data["lowest_price"]) * 100
        
        if current_drawdown > trade_data["max_drawdown_percent"]:
            trade_data["max_drawdown_percent"] = current_drawdown

    def _calculate_current_indicators(self, symbol: str) -> Optional[IndicatorState]:
        """Calculate current technical indicators for the symbol."""
        try:
            # Fetch recent candle data
            candles = self.fetcher.fetch_candles(symbol, "5m", 100)
            if not candles or len(candles) < 50:
                return None
            
            # Calculate indicators
            rsi = self.indicators.calculate_rsi(candles)
            macd_data = self.indicators.calculate_macd_swing(candles)
            stoch_rsi = self.indicators.calculate_stoch_rsi(candles)
            atr = self.indicators.calculate_atr(candles)
            ema_20 = self.indicators.calculate_ema(candles, 20)
            ema_50 = self.indicators.calculate_ema(candles, 50)
            bb_upper, bb_lower = self.indicators.calculate_bollinger_bands(candles)
            
            # Calculate Supertrend (simplified version)
            supertrend, supertrend_direction = self._calculate_supertrend(candles)
            
            # Get volume
            volume = float(candles[-1]["volume"]) if candles else 0
            
            return IndicatorState(
                rsi=rsi or 50,
                macd_line=macd_data["macd_line"] if macd_data else 0,
                macd_signal=macd_data["signal_line"] if macd_data else 0,
                macd_histogram=macd_data["histogram"] if macd_data else 0,
                stoch_rsi_k=stoch_rsi["%K"] if stoch_rsi else 50,
                stoch_rsi_d=stoch_rsi["%D"] if stoch_rsi else 50,
                atr=atr or 0,
                supertrend=supertrend or 0,
                supertrend_direction=supertrend_direction or "neutral",
                ema_20=ema_20 or 0,
                ema_50=ema_50 or 0,
                bb_upper=bb_upper or 0,
                bb_lower=bb_lower or 0,
                volume=volume,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators for {symbol}: {e}")
            return None

    def _calculate_supertrend(self, candles: List[Dict]) -> Tuple[float, str]:
        """Calculate Supertrend indicator."""
        try:
            if len(candles) < 10:
                return 0, "neutral"
            
            # Simplified Supertrend calculation
            highs = [float(c["high"]) for c in candles]
            lows = [float(c["low"]) for c in candles]
            closes = [float(c["close"]) for c in candles]
            
            # Calculate ATR
            atr = self.indicators.calculate_atr(candles) or 0.001
            
            # Basic Supertrend logic
            current_close = closes[-1]
            current_high = highs[-1]
            current_low = lows[-1]
            
            # Simple trend direction based on EMA
            ema_10 = self.indicators.calculate_ema(candles, 10) or current_close
            
            if current_close > ema_10:
                supertrend = current_low - (atr * 2)
                direction = "bullish"
            else:
                supertrend = current_high + (atr * 2)
                direction = "bearish"
            
            return supertrend, direction
            
        except Exception as e:
            self.logger.error(f"Error calculating Supertrend: {e}")
            return 0, "neutral"

    def _update_indicator_history(self, trade_data: Dict, indicators: IndicatorState):
        """Update indicator history for journey tracking."""
        trade_data["indicator_history"].append(indicators.to_dict())
        
        # Keep only recent history
        if len(trade_data["indicator_history"]) > self.indicator_history_length:
            trade_data["indicator_history"] = trade_data["indicator_history"][-self.indicator_history_length:]

    def _analyze_indicator_journey(self, trade_data: Dict):
        """Analyze the journey of indicators to assess signal strength and trend alignment."""
        if len(trade_data["indicator_history"]) < 5:
            return
        
        try:
            history = trade_data["indicator_history"]
            current = history[-1]
            previous = history[-5] if len(history) >= 5 else history[0]
            
            # Calculate signal strength based on indicator consistency
            signal_strength = 1.0
            trend_alignment = 1.0
            
            # RSI analysis
            rsi_trend = current["rsi"] - previous["rsi"]
            if abs(rsi_trend) > 10:
                signal_strength *= 0.8  # RSI moving too fast
            
            # MACD analysis
            macd_strength = abs(current["macd_histogram"]) / max(abs(previous["macd_histogram"]), 0.001)
            if macd_strength < 0.5:
                signal_strength *= 0.7  # MACD weakening
            
            # Supertrend alignment
            side = trade_data["signal"]["side"]
            if side == "long" and current["supertrend_direction"] == "bearish":
                trend_alignment *= 0.6
            elif side == "short" and current["supertrend_direction"] == "bullish":
                trend_alignment *= 0.6
            
            # Volume analysis
            volume_change = current["volume"] / max(previous["volume"], 1)
            if volume_change < 0.5:
                signal_strength *= 0.9  # Low volume
            
            # Stoch RSI analysis
            if current["stoch_rsi_k"] > 80 or current["stoch_rsi_k"] < 20:
                signal_strength *= 0.8  # Overbought/oversold
            
            trade_data["signal_strength"] = max(0.1, min(1.0, signal_strength))
            trade_data["trend_alignment"] = max(0.1, min(1.0, trend_alignment))
            
        except Exception as e:
            self.logger.error(f"Error analyzing indicator journey: {e}")

    def _update_dynamic_risk_management(self, trade_data: Dict, indicators: IndicatorState):
        """Update dynamic stop-loss and target levels based on indicators and price action."""
        side = trade_data["signal"]["side"]
        current_price = trade_data["current_price"]
        entry_price = trade_data["entry_price"]
        profit_percent = trade_data["profit_loss_percent"]
        
        # Update ATR tracking
        trade_data["atr_values"].append(indicators.atr)
        if len(trade_data["atr_values"]) > 20:
            trade_data["atr_values"] = trade_data["atr_values"][-20:]
        
        current_atr = np.mean(trade_data["atr_values"]) if trade_data["atr_values"] else 0.001
        
        # Dynamic stop-loss management
        if self.trailing_stop_enabled:
            self._update_trailing_stop(trade_data, current_price, current_atr, indicators)
        
        # Dynamic target adjustment
        self._update_dynamic_target(trade_data, indicators, profit_percent)
        
        # Breakeven stop activation
        if not trade_data["breakeven_stop_activated"] and profit_percent >= self.breakeven_threshold:
            self._activate_breakeven_stop(trade_data, side, entry_price)
        
        # Profit lock activation
        if not trade_data["profit_lock_activated"] and profit_percent >= self.profit_lock_threshold:
            self._activate_profit_lock(trade_data, current_price, side)

    def _update_trailing_stop(self, trade_data: Dict, current_price: float, atr: float, indicators: IndicatorState):
        """Update trailing stop-loss based on price movement and volatility."""
        side = trade_data["signal"]["side"]
        entry_price = trade_data["entry_price"]
        profit_percent = trade_data["profit_loss_percent"]
        
        # Only activate trailing stop after minimum profit
        if profit_percent < 0.3:  # 0.3% minimum profit
            return
        
        # Calculate dynamic trailing distance based on ATR and volatility
        strategy = trade_data["signal"]["strategy"]
        
        if strategy == "scalping":
            base_trail_distance = 0.002  # 0.2% for scalping
        elif strategy == "swing":
            base_trail_distance = 0.005  # 0.5% for swing
        else:
            base_trail_distance = 0.008  # 0.8% for longer-term strategies
        
        # Adjust based on ATR
        atr_factor = min(2.0, max(0.5, atr / entry_price * 100))
        trail_distance = base_trail_distance * atr_factor
        
        # Adjust based on signal strength
        signal_factor = trade_data["signal_strength"]
        trail_distance *= (2.0 - signal_factor)  # Tighter trail for stronger signals
        
        if side == "long":
            new_stop = current_price * (1 - trail_distance)
            
            # Only move stop up (never down)
            if new_stop > trade_data["current_stop_loss"]:
                trade_data["current_stop_loss"] = new_stop
                trade_data["trailing_stop_activated"] = True
                self.logger.info(f"Updated trailing stop for {trade_data['signal']['symbol']}: {new_stop:.6f}")
        
        else:  # short
            new_stop = current_price * (1 + trail_distance)
            
            # Only move stop down (never up)
            if new_stop < trade_data["current_stop_loss"]:
                trade_data["current_stop_loss"] = new_stop
                trade_data["trailing_stop_activated"] = True
                self.logger.info(f"Updated trailing stop for {trade_data['signal']['symbol']}: {new_stop:.6f}")

    def _update_dynamic_target(self, trade_data: Dict, indicators: IndicatorState, profit_percent: float):
        """Recalculate exit target based on signal strength and indicator momentum."""
        original_target = trade_data["original_target"]
        current_target = trade_data["current_target"]
        signal_strength = trade_data["signal_strength"]
        trend_alignment = trade_data["trend_alignment"]
        
        # Calculate target adjustment factor
        strength_factor = signal_strength * trend_alignment
        
        # Adjust target based on profit and signal strength
        if profit_percent > 1.0 and strength_factor > 0.7:
            # Increase target if signal is strong and profitable
            target_multiplier = 1.0 + (strength_factor - 0.7) * 0.5
            new_target = original_target * target_multiplier
            
            if trade_data["signal"]["side"] == "long":
                if new_target > current_target:
                    trade_data["current_target"] = new_target
                    self.logger.info(f"Raised target for {trade_data['signal']['symbol']}: {new_target:.6f}")
            else:
                if new_target < current_target:
                    trade_data["current_target"] = new_target
                    self.logger.info(f"Lowered target for {trade_data['signal']['symbol']}: {new_target:.6f}")
        
        elif profit_percent < -0.5 and strength_factor < 0.4:
            # Lower target if signal is weak and losing
            target_multiplier = 1.0 - (0.4 - strength_factor) * 0.3
            new_target = original_target * target_multiplier
            
            if trade_data["signal"]["side"] == "long":
                if new_target < current_target:
                    trade_data["current_target"] = new_target
                    self.logger.info(f"Lowered target for {trade_data['signal']['symbol']}: {new_target:.6f}")
            else:
                if new_target > current_target:
                    trade_data["current_target"] = new_target
                    self.logger.info(f"Raised target for {trade_data['signal']['symbol']}: {new_target:.6f}")

    def _activate_breakeven_stop(self, trade_data: Dict, side: str, entry_price: float):
        """Activate breakeven stop-loss."""
        trade_data["breakeven_stop_activated"] = True
        trade_data["current_stop_loss"] = entry_price
        self.logger.info(f"Activated breakeven stop for {trade_data['signal']['symbol']}")

    def _activate_profit_lock(self, trade_data: Dict, current_price: float, side: str):
        """Activate profit lock mechanism."""
        trade_data["profit_lock_activated"] = True
        
        # Lock in 50% of current profit
        profit_percent = trade_data["profit_loss_percent"]
        lock_percent = profit_percent * 0.5
        
        if side == "long":
            new_stop = current_price * (1 - lock_percent / 100)
        else:
            new_stop = current_price * (1 + lock_percent / 100)
        
        trade_data["current_stop_loss"] = new_stop
        self.logger.info(f"Activated profit lock for {trade_data['signal']['symbol']}: {new_stop:.6f}")

    def _generate_exit_suggestions(self, trade_data: Dict, indicators: IndicatorState) -> List[Dict]:
        """Generate intelligent exit suggestions based on current market conditions."""
        suggestions = []
        current_price = trade_data["current_price"]
        side = trade_data["signal"]["side"]
        profit_percent = trade_data["profit_loss_percent"]
        signal_strength = trade_data["signal_strength"]
        
        # 1. Signal weakening exit
        if signal_strength < self.signal_weakening_threshold and profit_percent > 0.2:
            suggestions.append({
                "type": "signal_weakening",
                "priority": "high",
                "reason": f"Signal strength dropped to {signal_strength:.2f}",
                "suggested_action": "exit",
                "confidence": 0.8
            })
        
        # 2. Trend reversal exit
        if trade_data["trend_alignment"] < self.trend_reversal_threshold:
            suggestions.append({
                "type": "trend_reversal",
                "priority": "high",
                "reason": "Trend alignment weakened significantly",
                "suggested_action": "exit",
                "confidence": 0.9
            })
        
        # 3. Overbought/Oversold exit
        if side == "long" and indicators.rsi > 75 and indicators.stoch_rsi_k > 80:
            suggestions.append({
                "type": "overbought",
                "priority": "medium",
                "reason": "RSI and Stoch RSI indicate overbought conditions",
                "suggested_action": "exit",
                "confidence": 0.7
            })
        elif side == "short" and indicators.rsi < 25 and indicators.stoch_rsi_k < 20:
            suggestions.append({
                "type": "oversold",
                "priority": "medium",
                "reason": "RSI and Stoch RSI indicate oversold conditions",
                "suggested_action": "exit",
                "confidence": 0.7
            })
        
        # 4. MACD bearish crossover (for long positions)
        if side == "long" and indicators.macd_histogram < 0 and indicators.macd_line < indicators.macd_signal:
            suggestions.append({
                "type": "macd_bearish",
                "priority": "medium",
                "reason": "MACD showing bearish crossover",
                "suggested_action": "exit",
                "confidence": 0.6
            })
        
        # 5. Supertrend flip
        if side == "long" and indicators.supertrend_direction == "bearish":
            suggestions.append({
                "type": "supertrend_flip",
                "priority": "high",
                "reason": "Supertrend flipped to bearish",
                "suggested_action": "exit",
                "confidence": 0.8
            })
        elif side == "short" and indicators.supertrend_direction == "bullish":
            suggestions.append({
                "type": "supertrend_flip",
                "priority": "high",
                "reason": "Supertrend flipped to bullish",
                "suggested_action": "exit",
                "confidence": 0.8
            })
        
        # 6. Volatility exit
        if trade_data["volatility_score"] > 0.8 and profit_percent > 0.5:
            suggestions.append({
                "type": "high_volatility",
                "priority": "medium",
                "reason": "High volatility detected, consider taking profits",
                "suggested_action": "partial_exit",
                "confidence": 0.6
            })
        
        return suggestions

    def _check_exit_conditions(self):
        """Check all exit conditions for active trades."""
        for trade_id in list(self.active_trades.keys()):
            trade_data = self.active_trades[trade_id]
            
            if trade_data["status"] != TradeStatus.ACTIVE:
                continue
            
            current_price = trade_data["current_price"]
            side = trade_data["signal"]["side"]
            
            # Check stop-loss hit
            if self._is_stop_loss_hit(trade_data, current_price):
                self._exit_trade(trade_id, ExitReason.STOP_LOSS, current_price)
                continue
            
            # Check target hit
            if self._is_target_hit(trade_data, current_price):
                self._exit_trade(trade_id, ExitReason.TARGET_HIT, current_price)
                continue
            
            # Check time-based exit
            if trade_data["time_in_trade"] > self.max_time_in_trade:
                self._exit_trade(trade_id, ExitReason.TIME_EXIT, current_price)
                continue
            
            # Check high-priority exit suggestions
            high_priority_suggestions = [
                s for s in trade_data["dynamic_exit_suggestions"]
                if s["priority"] == "high" and s["suggested_action"] == "exit"
            ]
            
            if high_priority_suggestions:
                reason = ExitReason.SIGNAL_WEAKENING
                self._exit_trade(trade_id, reason, current_price)
                continue

    def _is_stop_loss_hit(self, trade_data: Dict, current_price: float) -> bool:
        """Check if stop-loss has been hit."""
        side = trade_data["signal"]["side"]
        stop_loss = trade_data["current_stop_loss"]
        
        if side == "long":
            return current_price <= stop_loss
        else:
            return current_price >= stop_loss

    def _is_target_hit(self, trade_data: Dict, current_price: float) -> bool:
        """Check if target has been hit."""
        side = trade_data["signal"]["side"]
        target = trade_data["current_target"]
        
        if side == "long":
            return current_price >= target
        else:
            return current_price <= target

    def _exit_trade(self, trade_id: str, reason: ExitReason, exit_price: float):
        """Exit a trade with the specified reason."""
        trade_data = self.active_trades[trade_id]
        
        trade_data["status"] = TradeStatus.EXITED
        trade_data["exit_reason"] = reason.value
        trade_data["exit_time"] = datetime.now()
        trade_data["exit_price"] = exit_price
        
        # Calculate final P&L
        entry_price = trade_data["entry_price"]
        side = trade_data["signal"]["side"]
        
        if side == "long":
            final_pl = exit_price - entry_price
        else:
            final_pl = entry_price - exit_price
        
        final_pl_percent = (final_pl / entry_price) * 100
        
        trade_data["profit_loss"] = final_pl
        trade_data["profit_loss_percent"] = final_pl_percent
        
        # Move to history
        self.trade_history.append(trade_data.copy())
        del self.active_trades[trade_id]
        
        self.logger.info(f"Exited trade {trade_id}: {reason.value} at {exit_price:.6f}, P&L: {final_pl_percent:.2f}%")

    def _update_trade_metrics(self):
        """Update overall trade metrics."""
        for trade_data in self.active_trades.values():
            if trade_data["status"] == TradeStatus.ACTIVE:
                # Update time in trade
                trade_data["time_in_trade"] = datetime.now() - trade_data["entry_time"]
                
                # Update volatility score
                if trade_data["atr_values"]:
                    current_atr = trade_data["atr_values"][-1]
                    avg_atr = np.mean(trade_data["atr_values"])
                    trade_data["volatility_score"] = current_atr / max(avg_atr, 0.001)

    def _save_monitoring_data(self):
        """Save current monitoring data to file."""
        try:
            data = {
                "timestamp": datetime.now().isoformat(),
                "active_trades": len(self.active_trades),
                "trades": {}
            }
            
            for trade_id, trade_data in self.active_trades.items():
                # Convert datetime objects to strings for JSON serialization
                trade_copy = trade_data.copy()
                trade_copy["entry_time"] = trade_data["entry_time"].isoformat()
                trade_copy["last_update"] = trade_data["last_update"].isoformat()
                trade_copy["time_in_trade"] = str(trade_data["time_in_trade"])
                
                if trade_data["exit_time"]:
                    trade_copy["exit_time"] = trade_data["exit_time"].isoformat()
                
                data["trades"][trade_id] = trade_copy
            
            with open("logs/real_time_monitoring.json", "w") as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Error saving monitoring data: {e}")

    def get_active_trades(self) -> Dict[str, Dict]:
        """Get all active trades."""
        return self.active_trades

    def get_trade_summary(self) -> Dict:
        """Get summary of all trades."""
        active_count = len(self.active_trades)
        total_pl = sum(t["profit_loss_percent"] for t in self.active_trades.values())
        avg_pl = total_pl / active_count if active_count > 0 else 0
        
        return {
            "active_trades": active_count,
            "total_pnl_percent": total_pl,
            "average_pnl_percent": avg_pl,
            "max_drawdown": max((t["max_drawdown_percent"] for t in self.active_trades.values()), default=0)
        }

    def manual_exit_trade(self, trade_id: str, exit_price: float = None) -> bool:
        """Manually exit a trade."""
        if trade_id not in self.active_trades:
            return False
        
        trade_data = self.active_trades[trade_id]
        
        if exit_price is None:
            exit_price = trade_data["current_price"]
        
        self._exit_trade(trade_id, ExitReason.MANUAL, exit_price)
        return True

    def get_trade_details(self, trade_id: str) -> Optional[Dict]:
        """Get detailed information about a specific trade."""
        return self.active_trades.get(trade_id) 