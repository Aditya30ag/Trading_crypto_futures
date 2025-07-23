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


class TradePhase(Enum):
    """Trade lifecycle phases"""
    ENTRY = "entry"
    MONITORING = "monitoring"
    ADJUSTING = "adjusting"
    EXITING = "exiting"
    COMPLETED = "completed"


class SignalStrength(Enum):
    """Signal strength levels"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


@dataclass
class SignalJourney:
    """Track the evolution of trading signals over time"""
    timestamp: datetime
    phase: TradePhase
    price: float
    rsi: float
    macd_line: float
    macd_signal: float
    macd_histogram: float
    ema_20: float
    ema_50: float
    bb_upper: float
    bb_lower: float
    atr: float
    volume: float
    signal_strength: SignalStrength
    trend_alignment: float
    volatility_score: float
    momentum_score: float
    
    def to_dict(self):
        return asdict(self)


@dataclass
class DynamicLevels:
    """Dynamic stop-loss and take-profit levels"""
    current_stop_loss: float
    current_take_profit: float
    trailing_stop: float
    breakeven_stop: float
    profit_lock_level: float
    risk_adjusted_target: float
    
    def to_dict(self):
        return asdict(self)


class IntelligentTradeMonitor:
    """
    Intelligent trading system that focuses on single high-quality entries
    with dynamic post-entry management based on signal journey evolution.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = setup_logger()
        self.fetcher = CoinDCXFetcher()
        self.indicators = TechnicalIndicators()
        
        # Active trades tracking (one per symbol-strategy pair)
        self.active_trades: Dict[str, Dict] = {}
        self.trade_history: List[Dict] = []
        
        # Monitoring settings
        self.monitoring_interval = 3  # seconds - more frequent for intelligent monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Signal quality thresholds
        self.min_signal_strength = 0.5  # Reduced from 0.7
        self.min_trend_alignment = 0.4  # Reduced from 0.6
        self.min_risk_reward = 1.5      # Reduced from 2.0
        
        # Dynamic adjustment parameters
        self.profit_lock_threshold = 0.8  # Lock profits at 0.8% gain
        self.breakeven_threshold = 0.4    # Move to breakeven at 0.4% gain
        self.trailing_activation = 0.6    # Activate trailing at 0.6% gain
        self.signal_weakening_threshold = 0.4
        
        # Strategy selection parameters
        self.strategy_weights = {
            "scalping": {"volatility": 0.3, "momentum": 0.4, "trend": 0.3},
            "swing": {"volatility": 0.2, "momentum": 0.3, "trend": 0.5},
            "long_swing": {"volatility": 0.1, "momentum": 0.2, "trend": 0.7},
            "trend": {"volatility": 0.1, "momentum": 0.1, "trend": 0.8}
        }
        
        self.logger.info("IntelligentTradeMonitor initialized")

    def evaluate_and_enter_trade(self, symbol: str, market_conditions: Dict) -> Optional[str]:
        """
        Evaluate market conditions and enter a single high-quality trade
        with the best-fit strategy and timeframe.
        
        Args:
            symbol: Trading symbol
            market_conditions: Current market analysis
            
        Returns:
            trade_id: Unique identifier for the trade, or None if no entry
        """
        try:
            # Check if we already have an active trade for this symbol
            if self._has_active_trade(symbol):
                self.logger.info(f"Already have active trade for {symbol}, skipping entry")
                return None
            
            # Analyze market conditions and select best strategy
            strategy_analysis = self._analyze_market_conditions(symbol, market_conditions)
            
            if not strategy_analysis:
                return None
            
            # Select the best strategy based on current conditions
            best_strategy = self._select_best_strategy(strategy_analysis)
            
            if not best_strategy:
                self.logger.info(f"No suitable strategy found for {symbol}")
                return None
            
            # Generate entry signal for the selected strategy
            entry_signal = self._generate_entry_signal(symbol, best_strategy, market_conditions)
            
            if not entry_signal:
                return None
            
            # Validate signal quality
            if not self._validate_signal_quality(entry_signal):
                self.logger.info(f"Signal quality insufficient for {symbol}")
                return None
            
            # Enter the trade
            trade_id = self._enter_trade(entry_signal)
            
            if trade_id:
                self.logger.info(f"Entered intelligent trade {trade_id} for {symbol} using {best_strategy['name']} strategy")
            
            return trade_id
            
        except Exception as e:
            self.logger.error(f"Error evaluating trade entry for {symbol}: {e}")
            return None

    def _analyze_market_conditions(self, symbol: str, market_conditions: Dict) -> Optional[Dict]:
        """Analyze current market conditions to determine strategy suitability."""
        try:
            # Fetch recent market data
            candles = self.fetcher.fetch_candlestick_data(symbol, "5m", 100)
            if not candles or len(candles) < 50:
                return None
            
            # Calculate key indicators
            rsi = self.indicators.calculate_rsi(candles)
            macd_data = self.indicators.calculate_macd_swing(candles)
            ema_20 = self.indicators.calculate_ema(candles, 20)
            ema_50 = self.indicators.calculate_ema(candles, 50)
            bb_upper, bb_lower = self.indicators.calculate_bollinger_bands(candles)
            atr = self.indicators.calculate_atr(candles)
            
            # Calculate market condition scores
            volatility_score = self._calculate_volatility_score(atr, market_conditions.get("current_price", 0))
            momentum_score = self._calculate_momentum_score(rsi, macd_data)
            trend_score = self._calculate_trend_score(ema_20, ema_50, market_conditions)
            
            return {
                "symbol": symbol,
                "volatility_score": volatility_score,
                "momentum_score": momentum_score,
                "trend_score": trend_score,
                "indicators": {
                    "rsi": rsi,
                    "macd": macd_data,
                    "ema_20": ema_20,
                    "ema_50": ema_50,
                    "bb_upper": bb_upper,
                    "bb_lower": bb_lower,
                    "atr": atr
                },
                "current_price": market_conditions.get("current_price", 0)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing market conditions for {symbol}: {e}")
            return None

    def _select_best_strategy(self, analysis: Dict) -> Optional[Dict]:
        """Select the best strategy based on current market conditions."""
        strategies = []
        
        for strategy_name, weights in self.strategy_weights.items():
            # Calculate strategy score based on market conditions
            score = (
                analysis["volatility_score"] * weights["volatility"] +
                analysis["momentum_score"] * weights["momentum"] +
                analysis["trend_score"] * weights["trend"]
            )
            
            strategies.append({
                "name": strategy_name,
                "score": score,
                "weights": weights
            })
        
        # Sort by score and select the best
        strategies.sort(key=lambda x: x["score"], reverse=True)
        
        if strategies and strategies[0]["score"] > 0.6:  # Minimum threshold
            return strategies[0]
        
        return None

    def _generate_entry_signal(self, symbol: str, strategy: Dict, market_conditions: Dict) -> Optional[Dict]:
        """Generate entry signal for the selected strategy."""
        try:
            # Get strategy-specific parameters
            strategy_params = self._get_strategy_parameters(strategy["name"])
            
            # Calculate entry levels based on current market conditions
            current_price = market_conditions["current_price"]
            atr = market_conditions["analysis"]["indicators"]["atr"]
            
            # Calculate dynamic entry levels
            if strategy["name"] == "scalping":
                stop_distance = atr * 1.5
                target_distance = atr * 3.0
            elif strategy["name"] == "swing":
                stop_distance = atr * 2.0
                target_distance = atr * 4.0
            elif strategy["name"] == "long_swing":
                stop_distance = atr * 2.5
                target_distance = atr * 5.0
            else:  # trend
                stop_distance = atr * 3.0
                target_distance = atr * 6.0
            
            # Determine trade direction based on indicators
            direction = self._determine_trade_direction(market_conditions["analysis"]["indicators"])
            
            if direction == "long":
                entry_price = current_price
                stop_loss = entry_price - stop_distance
                take_profit = entry_price + target_distance
            else:
                entry_price = current_price
                stop_loss = entry_price + stop_distance
                take_profit = entry_price - target_distance
            
            # Calculate risk-reward ratio
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            risk_reward_ratio = reward / risk if risk > 0 else 0
            
            return {
                "symbol": symbol,
                "strategy": strategy["name"],
                "direction": direction,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "risk_reward_ratio": risk_reward_ratio,
                "strategy_score": strategy["score"],
                "timestamp": datetime.now(),
                "indicators": market_conditions["analysis"]["indicators"]
            }
            
        except Exception as e:
            self.logger.error(f"Error generating entry signal: {e}")
            return None

    def _validate_signal_quality(self, signal: Dict) -> bool:
        """Validate signal quality before entry."""
        # Check minimum risk-reward ratio
        if signal["risk_reward_ratio"] < self.min_risk_reward:
            return False
        
        # Check strategy score
        if signal["strategy_score"] < 0.6:
            return False
        
        # Check signal strength
        signal_strength = self._calculate_signal_strength(signal)
        if signal_strength < self.min_signal_strength:
            return False
        
        return True

    def _enter_trade(self, signal: Dict) -> Optional[str]:
        """Enter the trade and start intelligent monitoring."""
        try:
            trade_id = f"{signal['symbol']}_{signal['strategy']}_{int(time.time())}"
            
            # Initialize trade data
            trade_data = {
                "trade_id": trade_id,
                "signal": signal,
                "phase": TradePhase.ENTRY,
                "entry_time": datetime.now(),
                "last_update": datetime.now(),
                
                # Price tracking
                "entry_price": signal["entry_price"],
                "current_price": signal["entry_price"],
                "highest_price": signal["entry_price"],
                "lowest_price": signal["entry_price"],
                
                # Dynamic levels
                "dynamic_levels": DynamicLevels(
                    current_stop_loss=signal["stop_loss"],
                    current_take_profit=signal["take_profit"],
                    trailing_stop=signal["stop_loss"],
                    breakeven_stop=signal["entry_price"],
                    profit_lock_level=signal["entry_price"],
                    risk_adjusted_target=signal["take_profit"]
                ),
                
                # Performance tracking
                "profit_loss": 0.0,
                "profit_loss_percent": 0.0,
                "max_profit_percent": 0.0,
                "max_drawdown_percent": 0.0,
                
                # Signal journey tracking
                "signal_journey": [],
                "current_signal_strength": SignalStrength.STRONG,
                "trend_alignment": 1.0,
                
                # Exit management
                "exit_reason": None,
                "exit_time": None,
                "exit_price": None,
                
                # Dynamic adjustments
                "trailing_activated": False,
                "breakeven_activated": False,
                "profit_lock_activated": False,
                "last_adjustment": datetime.now()
            }
            
            self.active_trades[trade_id] = trade_data
            
            # Start monitoring if not already active
            if not self.monitoring_active:
                self.start_monitoring()
            
            return trade_id
            
        except Exception as e:
            self.logger.error(f"Error entering trade: {e}")
            return None

    def start_monitoring(self):
        """Start the intelligent monitoring loop."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Intelligent monitoring started")

    def stop_monitoring(self):
        """Stop the monitoring loop."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("Intelligent monitoring stopped")

    def _monitoring_loop(self):
        """Main intelligent monitoring loop."""
        self.logger.info("Starting intelligent monitoring loop...")
        
        while self.monitoring_active:
            try:
                # Process each active trade
                for trade_id in list(self.active_trades.keys()):
                    if trade_id in self.active_trades:
                        self._process_trade_intelligently(trade_id)
                
                # Check exit conditions
                self._check_intelligent_exits()
                
                # Update trade metrics
                self._update_trade_metrics()
                
                # Save monitoring data
                if self.active_trades:
                    self._save_monitoring_data()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)
        
        self.logger.info("Intelligent monitoring loop ended")

    def _process_trade_intelligently(self, trade_id: str):
        """Process a trade with intelligent signal journey tracking."""
        trade_data = self.active_trades[trade_id]
        symbol = trade_data["signal"]["symbol"]
        
        try:
            # Fetch current market data
            market_data = self.fetcher.fetch_market_data(symbol)
            if not market_data:
                return
            
            current_price = market_data["last_price"]
            trade_data["current_price"] = current_price
            trade_data["last_update"] = datetime.now()
            
            # Update price tracking
            self._update_price_tracking(trade_data, current_price)
            
            # Calculate current indicators
            indicators = self._calculate_current_indicators(symbol)
            if not indicators:
                return
            
            # Update signal journey
            self._update_signal_journey(trade_data, indicators, current_price)
            
            # Analyze signal evolution
            signal_evolution = self._analyze_signal_evolution(trade_data)
            
            # Update dynamic levels based on signal journey
            self._update_dynamic_levels(trade_data, signal_evolution)
            
            # Check for phase transitions
            self._check_phase_transitions(trade_data, signal_evolution)
            
        except Exception as e:
            self.logger.error(f"Error processing trade {trade_id}: {e}")

    def _update_signal_journey(self, trade_data: Dict, indicators: Dict, current_price: float):
        """Update the signal journey with current market conditions."""
        # Calculate current signal strength
        signal_strength = self._calculate_current_signal_strength(indicators, trade_data["signal"])
        
        # Calculate trend alignment
        trend_alignment = self._calculate_trend_alignment(indicators, trade_data["signal"])
        
        # Calculate volatility and momentum scores
        volatility_score = self._calculate_volatility_score(indicators.get("atr", 0), current_price)
        momentum_score = self._calculate_momentum_score(indicators.get("rsi", 50), indicators.get("macd", {}))
        
        # Create signal journey entry
        journey_entry = SignalJourney(
            timestamp=datetime.now(),
            phase=trade_data["phase"],
            price=current_price,
            rsi=indicators.get("rsi", 50),
            macd_line=indicators.get("macd", {}).get("macd_line", 0),
            macd_signal=indicators.get("macd", {}).get("signal_line", 0),
            macd_histogram=indicators.get("macd", {}).get("histogram", 0),
            ema_20=indicators.get("ema_20", 0),
            ema_50=indicators.get("ema_50", 0),
            bb_upper=indicators.get("bb_upper", 0),
            bb_lower=indicators.get("bb_lower", 0),
            atr=indicators.get("atr", 0),
            volume=indicators.get("volume", 0),
            signal_strength=signal_strength,
            trend_alignment=trend_alignment,
            volatility_score=volatility_score,
            momentum_score=momentum_score
        )
        
        trade_data["signal_journey"].append(journey_entry.to_dict())
        
        # Keep only recent journey entries
        if len(trade_data["signal_journey"]) > 100:
            trade_data["signal_journey"] = trade_data["signal_journey"][-100:]
        
        # Update current signal strength
        trade_data["current_signal_strength"] = signal_strength
        trade_data["trend_alignment"] = trend_alignment

    def _analyze_signal_evolution(self, trade_data: Dict) -> Dict:
        """Analyze how the signal has evolved since entry."""
        if len(trade_data["signal_journey"]) < 5:
            return {"evolution": "insufficient_data"}
        
        journey = trade_data["signal_journey"]
        entry = journey[0]
        current = journey[-1]
        
        # Calculate evolution metrics
        price_change = ((current["price"] - entry["price"]) / entry["price"]) * 100
        signal_strength_change = self._compare_signal_strengths(entry["signal_strength"], current["signal_strength"])
        trend_alignment_change = current["trend_alignment"] - entry["trend_alignment"]
        
        # Determine evolution type
        if signal_strength_change > 0.2 and trend_alignment_change > 0.1:
            evolution = "strengthening"
        elif signal_strength_change < -0.2 or trend_alignment_change < -0.1:
            evolution = "weakening"
        else:
            evolution = "stable"
        
        return {
            "evolution": evolution,
            "price_change": price_change,
            "signal_strength_change": signal_strength_change,
            "trend_alignment_change": trend_alignment_change,
            "journey_length": len(journey)
        }

    def _update_dynamic_levels(self, trade_data: Dict, signal_evolution: Dict):
        """Update dynamic levels based on signal evolution."""
        signal = trade_data["signal"]
        current_price = trade_data["current_price"]
        profit_percent = trade_data["profit_loss_percent"]
        evolution = signal_evolution["evolution"]
        
        # Update stop-loss based on signal evolution
        if evolution == "strengthening" and profit_percent > self.trailing_activation:
            # Tighten stop-loss as signal strengthens
            if signal["direction"] == "long":
                new_stop = current_price * 0.995  # 0.5% below current price
                if new_stop > trade_data["dynamic_levels"]["current_stop_loss"]:
                    trade_data["dynamic_levels"]["current_stop_loss"] = new_stop
                    trade_data["trailing_activated"] = True
            else:
                new_stop = current_price * 1.005  # 0.5% above current price
                if new_stop < trade_data["dynamic_levels"]["current_stop_loss"]:
                    trade_data["dynamic_levels"]["current_stop_loss"] = new_stop
                    trade_data["trailing_activated"] = True
        
        elif evolution == "weakening" and profit_percent > 0:
            # Widen stop-loss to give more room if signal weakens but profitable
            if signal["direction"] == "long":
                new_stop = current_price * 0.985  # 1.5% below current price
                trade_data["dynamic_levels"]["current_stop_loss"] = max(
                    new_stop, trade_data["dynamic_levels"]["current_stop_loss"]
                )
            else:
                new_stop = current_price * 1.015  # 1.5% above current price
                trade_data["dynamic_levels"]["current_stop_loss"] = min(
                    new_stop, trade_data["dynamic_levels"]["current_stop_loss"]
                )
        
        # Update take-profit based on signal strength
        if evolution == "strengthening" and profit_percent > 0.5:
            # Extend target if signal is strengthening
            if signal["direction"] == "long":
                new_target = current_price * 1.02  # 2% above current price
                trade_data["dynamic_levels"]["current_take_profit"] = max(
                    new_target, trade_data["dynamic_levels"]["current_take_profit"]
                )
            else:
                new_target = current_price * 0.98  # 2% below current price
                trade_data["dynamic_levels"]["current_take_profit"] = min(
                    new_target, trade_data["dynamic_levels"]["current_take_profit"]
                )

    def _check_phase_transitions(self, trade_data: Dict, signal_evolution: Dict):
        """Check for phase transitions based on signal evolution."""
        current_phase = trade_data["phase"]
        evolution = signal_evolution["evolution"]
        profit_percent = trade_data["profit_loss_percent"]
        
        # Phase transition logic
        if current_phase == TradePhase.ENTRY and profit_percent > 0.2:
            trade_data["phase"] = TradePhase.MONITORING
            self.logger.info(f"Trade {trade_data['trade_id']} entered monitoring phase")
        
        elif current_phase == TradePhase.MONITORING:
            if evolution == "strengthening" and profit_percent > 0.5:
                trade_data["phase"] = TradePhase.ADJUSTING
                self.logger.info(f"Trade {trade_data['trade_id']} entered adjusting phase")
            elif evolution == "weakening" and profit_percent < -0.3:
                trade_data["phase"] = TradePhase.EXITING
                self.logger.info(f"Trade {trade_data['trade_id']} entered exiting phase")
        
        elif current_phase == TradePhase.ADJUSTING:
            if evolution == "weakening" and profit_percent > 0.8:
                trade_data["phase"] = TradePhase.EXITING
                self.logger.info(f"Trade {trade_data['trade_id']} entering exit phase (profit taking)")

    def _check_intelligent_exits(self):
        """Check for intelligent exit conditions based on signal journey."""
        for trade_id in list(self.active_trades.keys()):
            trade_data = self.active_trades[trade_id]
            
            if trade_data["phase"] == TradePhase.EXITING:
                self._execute_intelligent_exit(trade_id, "phase_transition")
                continue
            
            # Check signal weakening exit
            if (trade_data["current_signal_strength"] == SignalStrength.WEAK and 
                trade_data["profit_loss_percent"] > 0.2):
                self._execute_intelligent_exit(trade_id, "signal_weakening")
                continue
            
            # Check trend reversal exit
            if trade_data["trend_alignment"] < 0.3:
                self._execute_intelligent_exit(trade_id, "trend_reversal")
                continue

    def _execute_intelligent_exit(self, trade_id: str, reason: str):
        """Execute intelligent exit based on signal journey analysis."""
        trade_data = self.active_trades[trade_id]
        
        # Get current market price
        market_data = self.fetcher.fetch_market_data(trade_data["signal"]["symbol"])
        exit_price = market_data["last_price"] if market_data else trade_data["current_price"]
        
        # Update trade data
        trade_data["phase"] = TradePhase.COMPLETED
        trade_data["exit_reason"] = reason
        trade_data["exit_time"] = datetime.now()
        trade_data["exit_price"] = exit_price
        
        # Calculate final P&L
        entry_price = trade_data["entry_price"]
        direction = trade_data["signal"]["direction"]
        
        if direction == "long":
            final_pl = exit_price - entry_price
        else:
            final_pl = entry_price - exit_price
        
        final_pl_percent = (final_pl / entry_price) * 100
        
        trade_data["profit_loss"] = final_pl
        trade_data["profit_loss_percent"] = final_pl_percent
        
        # Move to history
        self.trade_history.append(trade_data.copy())
        del self.active_trades[trade_id]
        
        self.logger.info(f"Intelligent exit for {trade_id}: {reason} at {exit_price:.6f}, P&L: {final_pl_percent:.2f}%")

    # Helper methods
    def _has_active_trade(self, symbol: str) -> bool:
        """Check if we already have an active trade for this symbol."""
        return any(trade["signal"]["symbol"] == symbol for trade in self.active_trades.values())

    def _calculate_volatility_score(self, atr: float, price: float) -> float:
        """Calculate volatility score."""
        if not atr or not price:
            return 0.5
        volatility_percent = (atr / price) * 100
        return min(1.0, max(0.0, volatility_percent / 5.0))  # Normalize to 0-1

    def _calculate_momentum_score(self, rsi: float, macd_data: Dict) -> float:
        """Calculate momentum score."""
        rsi_score = 1.0 - abs(rsi - 50) / 50 if rsi else 0.5
        macd_score = 0.5
        if macd_data and "histogram" in macd_data:
            macd_score = min(1.0, abs(macd_data["histogram"]) / 0.01)
        return (rsi_score + macd_score) / 2

    def _calculate_trend_score(self, ema_20: float, ema_50: float, market_conditions: Dict) -> float:
        """Calculate trend score."""
        if not ema_20 or not ema_50:
            return 0.5
        current_price = market_conditions.get("current_price", 0)
        if not current_price:
            return 0.5
        
        # Check if price is above both EMAs
        if current_price > ema_20 > ema_50:
            return 1.0
        elif current_price > ema_20:
            return 0.7
        else:
            return 0.3

    def _get_strategy_parameters(self, strategy_name: str) -> Dict:
        """Get strategy-specific parameters."""
        return self.config["strategies"].get(strategy_name, {})

    def _determine_trade_direction(self, indicators: Dict) -> str:
        """Determine trade direction based on indicators."""
        rsi = indicators.get("rsi", 50)
        ema_20 = indicators.get("ema_20", 0)
        ema_50 = indicators.get("ema_50", 0)
        
        if rsi < 40 and ema_20 < ema_50:
            return "short"
        elif rsi > 60 and ema_20 > ema_50:
            return "long"
        else:
            return "long"  # Default to long

    def _calculate_signal_strength(self, signal: Dict) -> float:
        """Calculate overall signal strength."""
        return signal.get("strategy_score", 0.5)

    def _calculate_current_indicators(self, symbol: str) -> Optional[Dict]:
        """Calculate current indicators for a symbol."""
        try:
            candles = self.fetcher.fetch_candlestick_data(symbol, "5m", 50)
            if not candles or len(candles) < 20:
                return None
            
            return {
                "rsi": self.indicators.calculate_rsi(candles),
                "macd": self.indicators.calculate_macd_swing(candles),
                "ema_20": self.indicators.calculate_ema(candles, 20),
                "ema_50": self.indicators.calculate_ema(candles, 50),
                "bb_upper": self.indicators.calculate_bollinger_bands(candles)[0],
                "bb_lower": self.indicators.calculate_bollinger_bands(candles)[1],
                "atr": self.indicators.calculate_atr(candles),
                "volume": float(candles[-1]["volume"]) if candles else 0
            }
        except Exception as e:
            self.logger.error(f"Error calculating indicators for {symbol}: {e}")
            return None

    def _calculate_current_signal_strength(self, indicators: Dict, original_signal: Dict) -> SignalStrength:
        """Calculate current signal strength."""
        # Compare current indicators with entry conditions
        rsi = indicators.get("rsi", 50)
        macd_data = indicators.get("macd", {})
        
        # Simple strength calculation
        if rsi > 70 or rsi < 30:
            return SignalStrength.WEAK
        elif 40 <= rsi <= 60:
            return SignalStrength.STRONG
        else:
            return SignalStrength.MODERATE

    def _calculate_trend_alignment(self, indicators: Dict, original_signal: Dict) -> float:
        """Calculate current trend alignment."""
        ema_20 = indicators.get("ema_20", 0)
        ema_50 = indicators.get("ema_50", 0)
        current_price = indicators.get("price", 0)
        
        if not all([ema_20, ema_50, current_price]):
            return 0.5
        
        direction = original_signal["direction"]
        
        if direction == "long":
            if current_price > ema_20 > ema_50:
                return 1.0
            elif current_price > ema_20:
                return 0.7
            else:
                return 0.3
        else:
            if current_price < ema_20 < ema_50:
                return 1.0
            elif current_price < ema_20:
                return 0.7
            else:
                return 0.3

    def _compare_signal_strengths(self, entry_strength: str, current_strength: str) -> float:
        """Compare signal strengths."""
        strength_values = {
            "weak": 0.25,
            "moderate": 0.5,
            "strong": 0.75,
            "very_strong": 1.0
        }
        
        entry_val = strength_values.get(entry_strength, 0.5)
        current_val = strength_values.get(current_strength, 0.5)
        
        return current_val - entry_val

    def _update_price_tracking(self, trade_data: Dict, current_price: float):
        """Update price tracking metrics."""
        direction = trade_data["signal"]["direction"]
        entry_price = trade_data["entry_price"]
        
        # Update highest/lowest prices
        if direction == "long":
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

    def _update_trade_metrics(self):
        """Update overall trade metrics."""
        for trade_data in self.active_trades.values():
            if trade_data["phase"] != TradePhase.COMPLETED:
                trade_data["time_in_trade"] = datetime.now() - trade_data["entry_time"]

    def _save_monitoring_data(self):
        """Save current monitoring data."""
        try:
            data = {
                "timestamp": datetime.now().isoformat(),
                "active_trades": len(self.active_trades),
                "trades": {}
            }
            
            for trade_id, trade_data in self.active_trades.items():
                trade_copy = trade_data.copy()
                trade_copy["entry_time"] = trade_data["entry_time"].isoformat()
                trade_copy["last_update"] = trade_data["last_update"].isoformat()
                
                if trade_data["exit_time"]:
                    trade_copy["exit_time"] = trade_data["exit_time"].isoformat()
                
                data["trades"][trade_id] = trade_copy
            
            with open("logs/intelligent_monitoring.json", "w") as f:
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