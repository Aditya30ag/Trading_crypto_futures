import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from src.data.fetcher import CoinDCXFetcher
from src.data.indicators import TechnicalIndicators
from src.utils.logger import setup_logger


class SignalMonitor:
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger()
        self.fetcher = CoinDCXFetcher()
        self.indicators = TechnicalIndicators()
        
        # Active signals tracking
        self.active_signals: Dict[str, Dict] = {}
        self.signal_history: List[Dict] = []
        
        # Monitoring settings
        self.monitoring_interval = 10  # seconds
        self.trailing_stop_enabled = True
        self.dynamic_exit_enabled = True
        
        # Risk management
        signal_config = config.get("trading", {}).get("signal_generation", {})
        self.max_active_signals = signal_config.get("max_active_signals", 20)  # Monitor only top 20
        self.max_drawdown_percent = 2.0  # 2% max drawdown per signal
        
        self.logger.info("SignalMonitor initialized")

    def add_signal(self, signal: Dict) -> bool:
        """Add a new signal to active monitoring."""
        try:
            signal_id = f"{signal['symbol']}_{signal['strategy']}_{signal['timestamp']}"
            
            if len(self.active_signals) >= self.max_active_signals:
                self.logger.warning(f"Maximum active signals reached ({self.max_active_signals})")
                return False
            
            # Initialize monitoring data
            # Handle both exit_price and take_profit keys for compatibility
            exit_price = signal.get("exit_price", signal.get("take_profit"))
            if exit_price is None:
                self.logger.error(f"Signal missing exit_price/take_profit: {signal}")
                return False
                
            monitoring_data = {
                "signal": signal,
                "signal_id": signal_id,
                "entry_time": datetime.now(),
                "current_price": signal["entry_price"],
                "highest_price": signal["entry_price"],
                "lowest_price": signal["entry_price"],
                "current_stop_loss": signal["stop_loss"],
                "current_exit_price": exit_price,
                "status": "active",
                "exit_reason": None,
                "exit_time": None,
                "profit_loss": 0.0,
                "profit_loss_percent": 0.0,
                "trailing_stop_activated": False,
                "dynamic_exit_suggestions": [],
                "price_history": [],
                "last_update": datetime.now()
            }
            
            self.active_signals[signal_id] = monitoring_data
            self.logger.info(f"Added signal to monitoring: {signal_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding signal to monitoring: {e}")
            return False

    def add_high_profit_signal(self, signal: Dict) -> bool:
        """Add a high-profit signal with priority monitoring."""
        try:
            signal_id = f"{signal['symbol']}_{signal['strategy']}_{signal['timestamp']}"
            
            # Check if we have space for high-profit signals
            if len(self.active_signals) >= self.max_active_signals:
                # Remove lowest profit signal to make room for high-profit signal
                lowest_profit_signal = min(self.active_signals.items(), 
                                         key=lambda x: x[1]["signal"].get("estimated_profit_inr", 0))
                if signal.get("estimated_profit_inr", 0) > lowest_profit_signal[1]["signal"].get("estimated_profit_inr", 0):
                    del self.active_signals[lowest_profit_signal[0]]
                    self.logger.info(f"Replaced low-profit signal with high-profit signal: {signal_id}")
                else:
                    self.logger.warning(f"High-profit signal {signal_id} has lower profit than existing signals")
                    return False
            
            # Initialize monitoring data with high-profit priority
            exit_price = signal.get("exit_price", signal.get("take_profit"))
            if exit_price is None:
                self.logger.error(f"Signal missing exit_price/take_profit: {signal}")
                return False
                
            monitoring_data = {
                "signal": signal,
                "signal_id": signal_id,
                "entry_time": datetime.now(),
                "current_price": signal["entry_price"],
                "highest_price": signal["entry_price"],
                "lowest_price": signal["entry_price"],
                "current_stop_loss": signal["stop_loss"],
                "current_exit_price": exit_price,
                "status": "active",
                "exit_reason": None,
                "exit_time": None,
                "profit_loss": 0.0,
                "profit_loss_percent": 0.0,
                "trailing_stop_activated": False,
                "dynamic_exit_suggestions": [],
                "price_history": [],
                "last_update": datetime.now(),
                "is_high_profit": True,  # Mark as high-profit signal
                "priority_level": "high"  # Set high priority
            }
            
            self.active_signals[signal_id] = monitoring_data
            self.logger.info(f"Added high-profit signal to monitoring: {signal_id} (Est. Profit: ₹{signal.get('estimated_profit_inr', 0):.2f})")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding high-profit signal to monitoring: {e}")
            return False

    def get_high_profit_signals(self) -> List[Dict]:
        """Get all high-profit signals currently being monitored."""
        high_profit_signals = []
        for signal_id, signal_data in self.active_signals.items():
            if signal_data.get("is_high_profit", False):
                high_profit_signals.append({
                    "signal_id": signal_id,
                    "signal": signal_data["signal"],
                    "profit_loss": signal_data["profit_loss"],
                    "profit_loss_percent": signal_data["profit_loss_percent"],
                    "status": signal_data["status"]
                })
        return high_profit_signals

    def get_signal_priority_summary(self) -> Dict:
        """Get summary of signal priorities and profit potential."""
        high_profit_count = 0
        total_estimated_profit = 0.0
        total_current_profit = 0.0
        
        for signal_data in self.active_signals.values():
            if signal_data.get("is_high_profit", False):
                high_profit_count += 1
                total_estimated_profit += signal_data["signal"].get("estimated_profit_inr", 0)
                total_current_profit += signal_data["profit_loss"]
        
        return {
            "total_signals": len(self.active_signals),
            "high_profit_signals": high_profit_count,
            "total_estimated_profit": total_estimated_profit,
            "total_current_profit": total_current_profit,
            "average_profit_per_signal": total_current_profit / len(self.active_signals) if self.active_signals else 0
        }

    def update_signal_status(self, signal_id: str, current_price: float) -> Dict:
        """Update signal status and calculate new exit suggestions."""
        if signal_id not in self.active_signals:
            return {}
        
        signal_data = self.active_signals[signal_id]
        signal = signal_data["signal"]
        side = signal["side"]
        entry_price = signal["entry_price"]
        
        # Update price tracking
        signal_data["current_price"] = current_price
        signal_data["last_update"] = datetime.now()
        
        # Track highest/lowest prices
        if side == "long":
            signal_data["highest_price"] = max(signal_data["highest_price"], current_price)
            signal_data["lowest_price"] = min(signal_data["lowest_price"], current_price)
        else:
            signal_data["highest_price"] = max(signal_data["highest_price"], current_price)
            signal_data["lowest_price"] = min(signal_data["lowest_price"], current_price)
        
        # Calculate current P&L
        if side == "long":
            profit_loss = current_price - entry_price
        else:
            profit_loss = entry_price - current_price
        
        profit_loss_percent = (profit_loss / entry_price) * 100
        signal_data["profit_loss"] = profit_loss
        signal_data["profit_loss_percent"] = profit_loss_percent
        
        # Calculate INR P&L for monitoring
        position_size_ratio = self.config["trading"]["position_size_ratio"]
        initial_balance = self.config["trading"]["initial_balance"]
        
        # DYNAMIC: Calculate position size based on signal characteristics
        score = signal.get("score", 6)
        strategy = signal.get("strategy", "scalping")
        estimated_profit = signal.get("estimated_profit_inr", 100)
        
        # Adjust position size based on signal score
        score_multiplier = 1.0
        if score >= 8:
            score_multiplier = 1.3  # 30% more for very strong signals
        elif score >= 7:
            score_multiplier = 1.2  # 20% more for strong signals
        elif score >= 6:
            score_multiplier = 1.1  # 10% more for good signals
        elif score <= 5:
            score_multiplier = 0.8  # 20% less for weak signals
        
        # Adjust based on strategy
        strategy_multiplier = 1.0
        if strategy == "scalping":
            strategy_multiplier = 0.9  # Smaller positions for scalping
        elif strategy == "long_swing":
            strategy_multiplier = 1.1  # Larger positions for long swing
        
        # Adjust based on estimated profit
        profit_multiplier = 1.0
        if estimated_profit >= 500:
            profit_multiplier = 1.2  # Larger position for high-profit signals
        elif estimated_profit <= 100:
            profit_multiplier = 0.8  # Smaller position for low-profit signals
        
        # Calculate final position size
        final_position_size_ratio = position_size_ratio * score_multiplier * strategy_multiplier * profit_multiplier
        position_size_inr = initial_balance * final_position_size_ratio
        
        usdt_inr_rate = 93.0  # Default rate
        position_size_usdt = position_size_inr / usdt_inr_rate
        leverage = 25
        quantity = (position_size_usdt / entry_price) * leverage if entry_price > 0 else 0
        notional_value = quantity * entry_price
        current_pnl_inr = (profit_loss_percent / 100) * notional_value * usdt_inr_rate
        signal_data["profit_loss_inr"] = current_pnl_inr
        
        # Check for stop loss hit
        if self._check_stop_loss_hit(signal_data, current_price):
            return self._handle_exit(signal_id, "stop_loss", current_price)
        
        # Check for target hit
        if self._check_target_hit(signal_data, current_price):
            return self._handle_exit(signal_id, "target", current_price)
        
        # NEW: Check for automatic exit based on INR loss threshold
        if self._check_inr_loss_threshold(signal_data, current_price):
            return self._handle_exit(signal_id, "inr_loss_threshold", current_price)
        
        # NEW: Check for quick profit exit (0.5%+ profit for scalping)
        if self._check_quick_profit_exit(signal_data, current_price):
            return self._handle_exit(signal_id, "quick_profit", current_price)
        
        # Update trailing stop loss
        if self.trailing_stop_enabled:
            self._update_trailing_stop(signal_data, current_price)
        
        # Generate dynamic exit suggestions
        if self.dynamic_exit_enabled:
            suggestions = self._generate_exit_suggestions(signal_data, current_price)
            
            # Add INR loss warning to suggestions
            if current_pnl_inr < -150:
                suggestions.append({
                    "type": "inr_loss_warning",
                    "reason": f"High INR loss: ₹{current_pnl_inr:.2f} (approaching -₹200 threshold)",
                    "priority": "high",
                    "suggested_action": "exit"
                })
            elif current_pnl_inr < -100:
                suggestions.append({
                    "type": "inr_loss_warning",
                    "reason": f"Moderate INR loss: ₹{current_pnl_inr:.2f} (monitor closely)",
                    "priority": "medium",
                    "suggested_action": "monitor"
                })
            
            signal_data["dynamic_exit_suggestions"] = suggestions
        
        # Add to price history (keep last 100 points)
        signal_data["price_history"].append({
            "timestamp": datetime.now(),
            "price": current_price,
            "profit_loss_percent": profit_loss_percent
        })
        if len(signal_data["price_history"]) > 100:
            signal_data["price_history"] = signal_data["price_history"][-100:]
        
        return signal_data

    def _check_stop_loss_hit(self, signal_data: Dict, current_price: float) -> bool:
        """Check if stop loss has been hit."""
        signal = signal_data["signal"]
        side = signal["side"]
        stop_loss = signal_data["current_stop_loss"]
        
        if side == "long":
            return current_price <= stop_loss
        else:
            return current_price >= stop_loss

    def _check_target_hit(self, signal_data: Dict, current_price: float) -> bool:
        """Check if target has been hit."""
        signal = signal_data["signal"]
        side = signal["side"]
        target = signal_data["current_exit_price"]
        
        if side == "long":
            return current_price >= target
        else:
            return current_price <= target

    def _check_inr_loss_threshold(self, signal_data: Dict, current_price: float) -> bool:
        """Check if INR loss exceeds the threshold (-₹200)."""
        signal = signal_data["signal"]
        side = signal["side"]
        entry_price = signal["entry_price"]
        
        # Calculate current P&L percentage
        if side == "long":
            profit_loss_percent = ((current_price - entry_price) / entry_price) * 100
        else:
            profit_loss_percent = ((entry_price - current_price) / entry_price) * 100
        
        # Calculate position size and notional value
        position_size_ratio = self.config["trading"]["position_size_ratio"]
        initial_balance = self.config["trading"]["initial_balance"]
        position_size_inr = initial_balance * position_size_ratio
        
        # Calculate USDT position size
        usdt_inr_rate = 93.0  # Default rate
        position_size_usdt = position_size_inr / usdt_inr_rate
        
        # Calculate quantity and notional value
        leverage = 25  # 25x leverage
        quantity = (position_size_usdt / entry_price) * leverage if entry_price > 0 else 0
        notional_value = quantity * entry_price
        
        # Calculate current P&L in INR
        current_pnl_inr = (profit_loss_percent / 100) * notional_value * usdt_inr_rate
        
        # Check if loss exceeds -₹200 threshold
        if current_pnl_inr < -150:  # More aggressive: -₹150 instead of -₹200
            self.logger.warning(f"🚨 INR Loss Threshold Hit: {signal['symbol']} - Loss: ₹{current_pnl_inr:.2f} (Threshold: -₹150)")
            return True
        
        # Warning when approaching threshold
        if current_pnl_inr < -100:  # Warning at -₹100 instead of -₹150
            self.logger.warning(f"⚠️ Approaching INR Loss Threshold: {signal['symbol']} - Loss: ₹{current_pnl_inr:.2f} (Threshold: -₹150)")
        
        return False

    def _check_quick_profit_exit(self, signal_data: Dict, current_price: float) -> bool:
        """Check if quick profit exit should be triggered."""
        signal = signal_data["signal"]
        side = signal["side"]
        entry_price = signal["entry_price"]
        strategy = signal["strategy"]
        
        # Calculate current P&L percentage
        if side == "long":
            profit_loss_percent = ((current_price - entry_price) / entry_price) * 100
        else:
            profit_loss_percent = ((entry_price - current_price) / entry_price) * 100
        
        # Quick profit exit conditions
        if strategy == "scalping":
            # For scalping: exit at 0.5% profit or after 30 minutes
            if profit_loss_percent >= 0.5:
                self.logger.info(f"🎯 Quick profit exit for {signal['symbol']}: {profit_loss_percent:.2f}% profit")
                return True
            
            # Time-based exit for scalping (30 minutes max)
            entry_time = signal_data["entry_time"]
            time_elapsed = datetime.now() - entry_time
            if time_elapsed.total_seconds() / 60 > 30:  # 30 minutes
                self.logger.info(f"⏰ Time-based exit for {signal['symbol']}: {time_elapsed.total_seconds()/60:.1f} minutes")
                return True
        
        elif strategy == "swing":
            # For swing: exit at 1% profit or after 2 hours
            if profit_loss_percent >= 1.0:
                self.logger.info(f"🎯 Quick profit exit for {signal['symbol']}: {profit_loss_percent:.2f}% profit")
                return True
            
            # Time-based exit for swing (2 hours max)
            entry_time = signal_data["entry_time"]
            time_elapsed = datetime.now() - entry_time
            if time_elapsed.total_seconds() / 3600 > 2:  # 2 hours
                self.logger.info(f"⏰ Time-based exit for {signal['symbol']}: {time_elapsed.total_seconds()/3600:.1f} hours")
                return True
        
        return False

    def _update_trailing_stop(self, signal_data: Dict, current_price: float):
        """Update trailing stop loss based on price movement."""
        signal = signal_data["signal"]
        side = signal["side"]
        entry_price = signal["entry_price"]
        original_stop = signal["stop_loss"]
        
        # Get ATR for dynamic trailing stop
        atr = signal.get("atr", 0)
        volatility_factor = signal.get("volatility_factor", 0.01)
        
        # Calculate trailing stop based on profit
        if side == "long":
            profit_percent = ((current_price - entry_price) / entry_price) * 100
            
            # Activate trailing stop after 0.5% profit for scalping, 1% for others
            activation_threshold = 0.5 if signal["strategy"] == "scalping" else 1.0
            
            if profit_percent > activation_threshold:
                signal_data["trailing_stop_activated"] = True
                
                # Dynamic trailing distance based on volatility
                if signal["strategy"] == "scalping":
                    # For scalping: tighter trailing (0.3-0.5% based on volatility)
                    trail_distance = max(0.003, min(0.005, volatility_factor * 10))
                else:
                    # For other strategies: standard 0.5% trailing
                    trail_distance = 0.005
                
                new_stop = current_price * (1 - trail_distance)
                
                # Don't move stop loss below original stop
                if new_stop > original_stop:
                    signal_data["current_stop_loss"] = new_stop
                    self.logger.info(f"Updated trailing stop for {signal['symbol']}: {new_stop:.6f} (trail: {trail_distance*100:.1f}%)")
        
        else:  # short
            profit_percent = ((entry_price - current_price) / entry_price) * 100
            
            # Activate trailing stop after 0.5% profit for scalping, 1% for others
            activation_threshold = 0.5 if signal["strategy"] == "scalping" else 1.0
            
            if profit_percent > activation_threshold:
                signal_data["trailing_stop_activated"] = True
                
                # Dynamic trailing distance based on volatility
                if signal["strategy"] == "scalping":
                    # For scalping: tighter trailing (0.3-0.5% based on volatility)
                    trail_distance = max(0.003, min(0.005, volatility_factor * 10))
                else:
                    # For other strategies: standard 0.5% trailing
                    trail_distance = 0.005
                
                new_stop = current_price * (1 + trail_distance)
                
                # Don't move stop loss above original stop
                if new_stop < original_stop:
                    signal_data["current_stop_loss"] = new_stop
                    self.logger.info(f"Updated trailing stop for {signal['symbol']}: {new_stop:.6f} (trail: {trail_distance*100:.1f}%)")

    def _generate_exit_suggestions(self, signal_data: Dict, current_price: float) -> List[Dict]:
        """Generate dynamic exit suggestions based on market conditions."""
        signal = signal_data["signal"]
        symbol = signal["symbol"]
        side = signal["side"]
        entry_price = signal["entry_price"]
        current_profit_percent = signal_data["profit_loss_percent"]
        strategy = signal["strategy"]
        
        suggestions = []
        
        try:
            # Get current market data for analysis
            market_data = self.fetcher.fetch_market_data(symbol)
            if not market_data:
                return suggestions
            
            # Get recent candles for technical analysis
            timeframe = signal.get("timeframe", "5m")
            candles = self.fetcher.fetch_candlestick_data(symbol, timeframe, limit=50)
            if not candles or len(candles) < 20:
                return suggestions
            
            # Calculate indicators
            rsi = self.indicators.calculate_rsi(candles)
            ema_20 = self.indicators.calculate_ema(candles, 20)
            bb_upper, bb_lower = self.indicators.calculate_bollinger_bands(candles)
            
            if any(x is None for x in [rsi, ema_20, bb_upper, bb_lower]):
                return suggestions
            
            # Strategy-specific suggestions
            if strategy == "scalping":
                suggestions.extend(self._generate_scalping_suggestions(
                    signal_data, current_price, rsi, ema_20, bb_upper, bb_lower, current_profit_percent
                ))
            else:
                suggestions.extend(self._generate_general_suggestions(
                    signal_data, current_price, rsi, ema_20, bb_upper, bb_lower, current_profit_percent
                ))
            
        except Exception as e:
            self.logger.error(f"Error generating exit suggestions: {e}")
        
        return suggestions

    def _generate_scalping_suggestions(self, signal_data: Dict, current_price: float, 
                                     rsi: float, ema_20: float, bb_upper: float, bb_lower: float, 
                                     current_profit_percent: float) -> List[Dict]:
        """Generate scalping-specific exit suggestions."""
        signal = signal_data["signal"]
        side = signal["side"]
        entry_price = signal["entry_price"]
        suggestions = []
        
        # 1. Quick profit taking for scalping (0.5%+ profit)
        if current_profit_percent >= 0.5:
            suggestions.append({
                "type": "quick_profit",
                "reason": f"Quick profit achieved ({current_profit_percent:.2f}%), consider exit",
                "priority": "high",
                "suggested_action": "exit"
            })
        
        # 2. RSI-based exits for scalping
        if side == "long":
            if rsi > 75 and current_profit_percent > 0.2:
                suggestions.append({
                    "type": "rsi_overbought",
                    "reason": f"RSI overbought ({rsi:.1f}), consider taking profits",
                    "priority": "high",
                    "suggested_action": "exit"
                })
        else:  # short
            if rsi < 25 and current_profit_percent > 0.2:
                suggestions.append({
                    "type": "rsi_oversold",
                    "reason": f"RSI oversold ({rsi:.1f}), consider taking profits",
                    "priority": "high",
                    "suggested_action": "exit"
                })
        
        # 3. Bollinger Band exits for scalping
        if side == "long" and current_price >= bb_upper and current_profit_percent > 0.2:
            suggestions.append({
                "type": "bollinger_upper",
                "reason": f"Price at upper Bollinger Band, consider exit",
                "priority": "medium",
                "suggested_action": "exit"
            })
        elif side == "short" and current_price <= bb_lower and current_profit_percent > 0.2:
            suggestions.append({
                "type": "bollinger_lower",
                "reason": f"Price at lower Bollinger Band, consider exit",
                "priority": "medium",
                "suggested_action": "exit"
            })
        
        # 4. Trend reversal warning for scalping
        if side == "long" and current_price < ema_20 and current_profit_percent > 0.1:
            suggestions.append({
                "type": "trend_reversal",
                "reason": f"Price below EMA20, trend may be reversing",
                "priority": "medium",
                "suggested_action": "exit"
            })
        elif side == "short" and current_price > ema_20 and current_profit_percent > 0.1:
            suggestions.append({
                "type": "trend_reversal",
                "reason": f"Price above EMA20, trend may be reversing",
                "priority": "medium",
                "suggested_action": "exit"
            })
        
        # 5. Time-based exit for scalping (more aggressive)
        entry_time = signal_data["entry_time"]
        time_elapsed = datetime.now() - entry_time
        max_hold_time = signal.get("max_hold_time", 10)  # Default 10 candles
        
        if time_elapsed.total_seconds() / 60 > max_hold_time * 0.6:  # 60% of max time
            suggestions.append({
                "type": "time_based",
                "reason": f"Approaching max hold time ({time_elapsed.total_seconds()/60:.1f}min)",
                "priority": "medium",
                "suggested_action": "exit"
            })
        
        return suggestions

    def _generate_general_suggestions(self, signal_data: Dict, current_price: float, 
                                    rsi: float, ema_20: float, bb_upper: float, bb_lower: float, 
                                    current_profit_percent: float) -> List[Dict]:
        """Generate general exit suggestions for non-scalping strategies."""
        signal = signal_data["signal"]
        side = signal["side"]
        suggestions = []
        
        # 1. RSI-based exit
        if side == "long":
            if rsi > 70 and current_profit_percent > 0.5:
                suggestions.append({
                    "type": "rsi_overbought",
                    "reason": f"RSI overbought ({rsi:.1f}), consider taking profits",
                    "priority": "high" if rsi > 75 else "medium",
                    "suggested_action": "partial_exit"
                })
        else:  # short
            if rsi < 30 and current_profit_percent > 0.5:
                suggestions.append({
                    "type": "rsi_oversold",
                    "reason": f"RSI oversold ({rsi:.1f}), consider taking profits",
                    "priority": "high" if rsi < 25 else "medium",
                    "suggested_action": "partial_exit"
                })
        
        # 2. Bollinger Band exit
        if side == "long" and current_price >= bb_upper and current_profit_percent > 0.3:
            suggestions.append({
                "type": "bollinger_upper",
                "reason": f"Price at upper Bollinger Band, consider exit",
                "priority": "medium",
                "suggested_action": "exit"
            })
        elif side == "short" and current_price <= bb_lower and current_profit_percent > 0.3:
            suggestions.append({
                "type": "bollinger_lower",
                "reason": f"Price at lower Bollinger Band, consider exit",
                "priority": "medium",
                "suggested_action": "exit"
            })
        
        # 3. Trend reversal warning
        if side == "long" and current_price < ema_20 and current_profit_percent > 0.2:
            suggestions.append({
                "type": "trend_reversal",
                "reason": f"Price below EMA20, trend may be reversing",
                "priority": "medium",
                "suggested_action": "tighten_stop"
            })
        elif side == "short" and current_price > ema_20 and current_profit_percent > 0.2:
            suggestions.append({
                "type": "trend_reversal",
                "reason": f"Price above EMA20, trend may be reversing",
                "priority": "medium",
                "suggested_action": "tighten_stop"
            })
        
        # 4. Profit taking levels
        if current_profit_percent >= 2.0:
            suggestions.append({
                "type": "profit_taking",
                "reason": f"Good profit achieved ({current_profit_percent:.2f}%), consider exit",
                "priority": "high",
                "suggested_action": "exit"
            })
        elif current_profit_percent >= 1.0:
            suggestions.append({
                "type": "profit_taking",
                "reason": f"Moderate profit achieved ({current_profit_percent:.2f}%), consider partial exit",
                "priority": "medium",
                "suggested_action": "partial_exit"
            })
        
        # 5. Time-based exit
        entry_time = signal_data["entry_time"]
        time_elapsed = datetime.now() - entry_time
        max_hold_time = signal.get("max_hold_time", 60)  # Default 60 minutes
        
        if time_elapsed.total_seconds() / 3600 > max_hold_time * 0.8:  # 80% of max time
            suggestions.append({
                "type": "time_based",
                "reason": f"Approaching max hold time ({time_elapsed.total_seconds()/3600:.1f}h)",
                "priority": "medium",
                "suggested_action": "exit"
            })
        
        return suggestions

    def _handle_exit(self, signal_id: str, exit_reason: str, exit_price: float) -> Dict:
        """Handle signal exit and move to history."""
        signal_data = self.active_signals[signal_id]
        signal = signal_data["signal"]
        
        # Update exit information
        signal_data["status"] = "closed"
        signal_data["exit_reason"] = exit_reason
        signal_data["exit_time"] = datetime.now()
        signal_data["current_price"] = exit_price
        
        # Calculate final P&L
        entry_price = signal["entry_price"]
        side = signal["side"]
        
        if side == "long":
            final_profit = exit_price - entry_price
        else:
            final_profit = entry_price - exit_price
        
        final_profit_percent = (final_profit / entry_price) * 100
        signal_data["profit_loss"] = final_profit
        signal_data["profit_loss_percent"] = final_profit_percent
        
        # Log exit
        self.logger.info(f"Signal {signal_id} exited: {exit_reason} at {exit_price:.6f} "
                        f"(P&L: {final_profit_percent:.2f}%)")
        
        # Move to history
        self.signal_history.append(signal_data)
        del self.active_signals[signal_id]
        
        return signal_data

    def monitor_all_signals(self) -> Dict:
        """Monitor all active signals and return status summary."""
        summary = {
            "active_signals": len(self.active_signals),
            "total_profit_loss": 0.0,
            "signals": [],
            "alerts": []
        }
        
        # Test connection first
        if not self.fetcher.test_connection():
            self.logger.warning("API connection test failed, continuing with cached data")
        
        for signal_id, signal_data in list(self.active_signals.items()):
            try:
                symbol = signal_data["signal"]["symbol"]
                current_price_data = self.fetcher.fetch_market_data(symbol)
                
                if not current_price_data or current_price_data["last_price"] == 0:
                    self.logger.warning(f"Could not fetch current price for {symbol}, skipping update")
                    continue
                
                current_price = current_price_data["last_price"]
                updated_data = self.update_signal_status(signal_id, current_price)
                
                if updated_data:
                    summary["signals"].append({
                        "signal_id": signal_id,
                        "symbol": symbol,
                        "strategy": signal_data["signal"]["strategy"],
                        "side": signal_data["signal"]["side"],
                        "entry_price": signal_data["signal"]["entry_price"],
                        "current_price": current_price,
                        "profit_loss_percent": updated_data["profit_loss_percent"],
                        "status": updated_data["status"],
                        "current_stop_loss": updated_data["current_stop_loss"],
                        "current_exit_price": updated_data["current_exit_price"],
                        "exit_suggestions": updated_data.get("dynamic_exit_suggestions", [])
                    })
                    
                    summary["total_profit_loss"] += updated_data["profit_loss_percent"]
                    
                    # Generate alerts for high-priority suggestions
                    high_priority_suggestions = [
                        s for s in updated_data.get("dynamic_exit_suggestions", [])
                        if s.get("priority") == "high"
                    ]
                    
                    for suggestion in high_priority_suggestions:
                        summary["alerts"].append({
                            "signal_id": signal_id,
                            "symbol": symbol,
                            "type": suggestion["type"],
                            "message": suggestion["reason"],
                            "action": suggestion["suggested_action"],
                            "timestamp": datetime.now()
                        })
                
            except Exception as e:
                self.logger.error(f"Error monitoring signal {signal_id}: {e}")
                # Don't remove the signal, just skip this update cycle
        
        return summary

    def get_signal_summary(self) -> Dict:
        """Get summary of all signals (active and historical)."""
        active_count = len(self.active_signals)
        historical_count = len(self.signal_history)
        
        # Calculate historical performance
        total_trades = historical_count
        profitable_trades = sum(1 for s in self.signal_history if s["profit_loss_percent"] > 0)
        win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
        
        total_profit = sum(s["profit_loss_percent"] for s in self.signal_history)
        avg_profit = total_profit / total_trades if total_trades > 0 else 0
        
        return {
            "active_signals": active_count,
            "historical_signals": historical_count,
            "total_trades": total_trades,
            "profitable_trades": profitable_trades,
            "win_rate": win_rate,
            "total_profit_percent": total_profit,
            "average_profit_percent": avg_profit,
            "active_signals_details": list(self.active_signals.keys()),
            "last_update": datetime.now()
        }

    def save_monitoring_data(self, filename: str = "monitoring_data.json"):
        """Save monitoring data to file."""
        try:
            # Convert datetime objects to strings for JSON serialization
            def convert_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return obj
            
            # Deep copy the data and convert datetime objects
            import copy
            data_to_save = copy.deepcopy({
                "active_signals": self.active_signals,
                "signal_history": self.signal_history,
                "summary": self.get_signal_summary(),
                "timestamp": datetime.now().isoformat()
            })
            
            # Convert all datetime objects and enum objects in the data
            def convert_objects_recursive(obj):
                if isinstance(obj, dict):
                    converted = {}
                    for k, v in obj.items():
                        if isinstance(v, datetime):
                            converted[k] = v.isoformat()
                        elif hasattr(v, 'value') and hasattr(v, '__class__'):  # Enum objects
                            converted[k] = v.value
                        else:
                            converted[k] = convert_objects_recursive(v)
                    return converted
                elif isinstance(obj, list):
                    return [convert_objects_recursive(item) for item in obj]
                elif isinstance(obj, datetime):
                    return obj.isoformat()
                elif hasattr(obj, 'value') and hasattr(obj, '__class__'):  # Enum objects
                    return obj.value
                else:
                    return obj
            
            data_to_save = convert_objects_recursive(data_to_save)
            
            with open(filename, 'w') as f:
                json.dump(data_to_save, f, indent=2)
            
            self.logger.info(f"Monitoring data saved to {filename} with {len(self.active_signals)} active signals")
            
        except Exception as e:
            self.logger.error(f"Error saving monitoring data: {e}")

    def load_monitoring_data(self, filename: str = "monitoring_data.json"):
        """Load monitoring data from file."""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # Convert string timestamps back to datetime objects
            def convert_timestamps_recursive(obj):
                if isinstance(obj, dict):
                    converted = {}
                    for k, v in obj.items():
                        if k in ["entry_time", "exit_time", "last_update", "timestamp"] and isinstance(v, str):
                            try:
                                converted[k] = datetime.fromisoformat(v)
                            except:
                                converted[k] = v
                        else:
                            converted[k] = convert_timestamps_recursive(v)
                    return converted
                elif isinstance(obj, list):
                    return [convert_timestamps_recursive(item) for item in obj]
                else:
                    return obj
            
            converted_data = convert_timestamps_recursive(data)
            
            self.active_signals = converted_data.get("active_signals", {})
            self.signal_history = converted_data.get("signal_history", [])
            
            self.logger.info(f"Monitoring data loaded from {filename} with {len(self.active_signals)} active signals")
            
        except FileNotFoundError:
            self.logger.info(f"No existing monitoring data found at {filename}")
        except Exception as e:
            self.logger.error(f"Error loading monitoring data: {e}") 