import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass

from src.strategies.strategy_manager import StrategyManager
from src.trading.real_time_monitor import RealTimeTradeMonitor, ExitReason
from src.utils.logger import setup_logger
from src.data.fetcher import CoinDCXFetcher


@dataclass
class TradeSignal:
    """Enhanced trade signal with additional metadata"""
    symbol: str
    strategy: str
    side: str
    entry_price: float
    stop_loss: float
    exit_price: float
    score: float
    timestamp: datetime
    indicators: Dict
    signal_strength: float
    risk_reward_ratio: float
    volatility_factor: float
    trend_alignment: float


class EnhancedTradeExecutor:
    """
    Enhanced trade executor with intelligent real-time monitoring and
    dynamic risk management capabilities.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = setup_logger()
        self.fetcher = CoinDCXFetcher()
        self.strategy_manager = StrategyManager(config)
        self.real_time_monitor = RealTimeTradeMonitor(config)
        
        # Trading state
        self.active_trades: Dict[str, Dict] = {}
        self.trade_history: List[Dict] = []
        self.total_pnl = 0.0
        self.win_rate = 0.0
        self.max_drawdown = 0.0
        
        # Risk management
        self.max_concurrent_trades = 5
        self.max_daily_loss = 5.0  # 5% max daily loss
        self.max_position_size = 0.2  # 20% max position size
        
        # Performance tracking
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_reset_date = datetime.now().date()
        
        self.logger.info("EnhancedTradeExecutor initialized")

    def run(self) -> List[Dict]:
        """
        Main execution loop that generates signals and manages trades
        with real-time monitoring.
        """
        self.logger.info("Starting enhanced trade execution...")
        
        try:
            # Generate trading signals
            signals = self.strategy_manager.generate_signals()
            self.logger.info(f"Generated {len(signals)} signals")
            
            if not signals:
                self.logger.warning("No signals generated")
                return []
            
            # Process and filter signals
            enhanced_signals = self._enhance_signals(signals)
            filtered_signals = self._filter_signals(enhanced_signals)
            
            self.logger.info(f"Filtered to {len(filtered_signals)} high-quality signals")
            
            # Execute trades for qualified signals
            executed_trades = []
            for signal in filtered_signals:
                if self._can_execute_trade(signal):
                    trade_id = self._execute_trade(signal)
                    if trade_id:
                        executed_trades.append({
                            "trade_id": trade_id,
                            "signal": signal
                        })
                        self.logger.info(f"Executed trade {trade_id} for {signal.symbol}")
            
            # Start real-time monitoring if we have active trades
            if executed_trades:
                self.logger.info(f"Started monitoring {len(executed_trades)} trades")
                self._start_performance_monitoring()
            
            return executed_trades
            
        except Exception as e:
            self.logger.error(f"Error in trade execution: {e}")
            return []

    def _enhance_signals(self, signals: List[Dict]) -> List[TradeSignal]:
        """Enhance raw signals with additional analysis and metadata."""
        enhanced_signals = []
        
        for signal in signals:
            try:
                # Calculate additional indicators
                symbol = signal["symbol"]
                market_data = self.fetcher.fetch_market_data(symbol)
                
                if not market_data:
                    continue
                
                # Fetch recent candles for advanced analysis
                candles = self.fetcher.fetch_candles(symbol, "5m", 100)
                if not candles or len(candles) < 50:
                    continue
                
                # Calculate advanced indicators
                from src.data.indicators import TechnicalIndicators
                indicators = TechnicalIndicators()
                
                rsi = indicators.calculate_rsi(candles)
                macd_data = indicators.calculate_macd_swing(candles)
                stoch_rsi = indicators.calculate_stoch_rsi(candles)
                atr = indicators.calculate_atr(candles)
                ema_20 = indicators.calculate_ema(candles, 20)
                ema_50 = indicators.calculate_ema(candles, 50)
                
                # Calculate signal strength
                signal_strength = self._calculate_signal_strength(signal, {
                    "rsi": rsi,
                    "macd": macd_data,
                    "stoch_rsi": stoch_rsi,
                    "atr": atr,
                    "ema_20": ema_20,
                    "ema_50": ema_50
                })
                
                # Calculate volatility factor
                volatility_factor = self._calculate_volatility_factor(atr, signal["entry_price"])
                
                # Calculate trend alignment
                trend_alignment = self._calculate_trend_alignment(signal, ema_20, ema_50)
                
                # Calculate risk-reward ratio
                risk_reward_ratio = self._calculate_risk_reward_ratio(signal)
                
                # Create enhanced signal
                enhanced_signal = TradeSignal(
                    symbol=signal["symbol"],
                    strategy=signal["strategy"],
                    side=signal["side"],
                    entry_price=signal["entry_price"],
                    stop_loss=signal["stop_loss"],
                    exit_price=signal["exit_price"],
                    score=signal.get("score", 0),
                    timestamp=datetime.fromtimestamp(signal["timestamp"]),
                    indicators={
                        "rsi": rsi,
                        "macd": macd_data,
                        "stoch_rsi": stoch_rsi,
                        "atr": atr,
                        "ema_20": ema_20,
                        "ema_50": ema_50
                    },
                    signal_strength=signal_strength,
                    risk_reward_ratio=risk_reward_ratio,
                    volatility_factor=volatility_factor,
                    trend_alignment=trend_alignment
                )
                
                enhanced_signals.append(enhanced_signal)
                
            except Exception as e:
                self.logger.error(f"Error enhancing signal for {signal.get('symbol', 'unknown')}: {e}")
                continue
        
        return enhanced_signals

    def _calculate_signal_strength(self, signal: Dict, indicators: Dict) -> float:
        """Calculate overall signal strength based on multiple factors."""
        strength = 1.0
        
        # RSI factor
        rsi = indicators.get("rsi", 50)
        if signal["side"] == "long":
            if 30 <= rsi <= 70:
                strength *= 1.2
            elif rsi < 20 or rsi > 80:
                strength *= 0.7
        else:  # short
            if 30 <= rsi <= 70:
                strength *= 1.2
            elif rsi < 20 or rsi > 80:
                strength *= 0.7
        
        # MACD factor
        macd_data = indicators.get("macd")
        if macd_data:
            macd_histogram = macd_data.get("histogram", 0)
            if abs(macd_histogram) > 0.001:
                strength *= 1.1
        
        # Stoch RSI factor
        stoch_rsi = indicators.get("stoch_rsi")
        if stoch_rsi:
            stoch_k = stoch_rsi.get("%K", 50)
            if signal["side"] == "long" and stoch_k < 80:
                strength *= 1.1
            elif signal["side"] == "short" and stoch_k > 20:
                strength *= 1.1
        
        # Strategy-specific adjustments
        strategy = signal["strategy"]
        if strategy == "scalping":
            strength *= 0.9  # Slightly lower confidence for scalping
        elif strategy == "trend":
            strength *= 1.1  # Higher confidence for trend following
        
        return max(0.1, min(1.0, strength))

    def _calculate_volatility_factor(self, atr: float, price: float) -> float:
        """Calculate volatility factor based on ATR."""
        if not atr or not price:
            return 0.01
        
        volatility_percent = (atr / price) * 100
        return min(0.05, max(0.001, volatility_percent))

    def _calculate_trend_alignment(self, signal: Dict, ema_20: float, ema_50: float) -> float:
        """Calculate trend alignment score."""
        if not ema_20 or not ema_50:
            return 0.5
        
        entry_price = signal["entry_price"]
        
        # Check if price is above/below EMAs
        if signal["side"] == "long":
            if entry_price > ema_20 > ema_50:
                return 1.0
            elif entry_price > ema_20:
                return 0.7
            else:
                return 0.3
        else:  # short
            if entry_price < ema_20 < ema_50:
                return 1.0
            elif entry_price < ema_20:
                return 0.7
            else:
                return 0.3

    def _calculate_risk_reward_ratio(self, signal: Dict) -> float:
        """Calculate risk-reward ratio."""
        entry_price = signal["entry_price"]
        stop_loss = signal["stop_loss"]
        exit_price = signal["exit_price"]
        
        if signal["side"] == "long":
            risk = entry_price - stop_loss
            reward = exit_price - entry_price
        else:
            risk = stop_loss - entry_price
            reward = entry_price - exit_price
        
        if risk <= 0:
            return 0.0
        
        return reward / risk

    def _filter_signals(self, signals: List[TradeSignal]) -> List[TradeSignal]:
        """Filter signals based on quality criteria."""
        filtered_signals = []
        
        for signal in signals:
            # Minimum score requirement
            if signal.score < 6:
                continue
            
            # Minimum signal strength
            if signal.signal_strength < 0.6:
                continue
            
            # Minimum risk-reward ratio
            if signal.risk_reward_ratio < 1.5:
                continue
            
            # Minimum trend alignment
            if signal.trend_alignment < 0.5:
                continue
            
            # Maximum volatility (avoid extremely volatile assets)
            if signal.volatility_factor > 0.03:
                continue
            
            filtered_signals.append(signal)
        
        # Sort by overall quality score
        filtered_signals.sort(
            key=lambda x: x.score * x.signal_strength * x.trend_alignment,
            reverse=True
        )
        
        return filtered_signals

    def _can_execute_trade(self, signal: TradeSignal) -> bool:
        """Check if we can execute a new trade based on risk management rules."""
        # Check maximum concurrent trades
        if len(self.active_trades) >= self.max_concurrent_trades:
            return False
        
        # Check daily loss limit
        if self.daily_pnl < -self.max_daily_loss:
            self.logger.warning(f"Daily loss limit reached: {self.daily_pnl:.2f}%")
            return False
        
        # Check if we already have a position in this symbol
        for trade_id, trade_data in self.active_trades.items():
            if trade_data["signal"]["symbol"] == signal.symbol:
                return False
        
        # Check position size limit
        position_value = signal.entry_price * self.config["trading"]["position_size_ratio"]
        account_value = self.config["trading"]["initial_balance"]
        position_size_percent = (position_value / account_value) * 100
        
        if position_size_percent > self.max_position_size * 100:
            return False
        
        return True

    def _execute_trade(self, signal: TradeSignal) -> Optional[str]:
        """Execute a trade and start monitoring."""
        try:
            # Convert TradeSignal to dictionary format
            signal_dict = {
                "symbol": signal.symbol,
                "strategy": signal.strategy,
                "side": signal.side,
                "entry_price": signal.entry_price,
                "stop_loss": signal.stop_loss,
                "exit_price": signal.exit_price,
                "score": signal.score,
                "timestamp": int(signal.timestamp.timestamp()),
                "signal_strength": signal.signal_strength,
                "risk_reward_ratio": signal.risk_reward_ratio,
                "volatility_factor": signal.volatility_factor,
                "trend_alignment": signal.trend_alignment,
                "indicators": signal.indicators
            }
            
            # Start real-time monitoring
            trade_id = self.real_time_monitor.start_trade_monitoring(signal_dict)
            
            if trade_id:
                # Add to active trades
                self.active_trades[trade_id] = {
                    "signal": signal_dict,
                    "entry_time": datetime.now(),
                    "status": "active"
                }
                
                self.logger.info(f"Executed trade {trade_id} for {signal.symbol}")
                return trade_id
            
        except Exception as e:
            self.logger.error(f"Error executing trade for {signal.symbol}: {e}")
        
        return None

    def _start_performance_monitoring(self):
        """Start performance monitoring in a separate thread."""
        def monitor_performance():
            while self.active_trades or self.real_time_monitor.active_trades:
                try:
                    # Update daily metrics
                    self._update_daily_metrics()
                    
                    # Check for completed trades
                    self._process_completed_trades()
                    
                    # Log performance summary
                    self._log_performance_summary()
                    
                    time.sleep(30)  # Check every 30 seconds
                    
                except Exception as e:
                    self.logger.error(f"Error in performance monitoring: {e}")
                    time.sleep(60)
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=monitor_performance, daemon=True)
        monitor_thread.start()

    def _update_daily_metrics(self):
        """Update daily performance metrics."""
        current_date = datetime.now().date()
        
        # Reset daily metrics if it's a new day
        if current_date != self.last_reset_date:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.last_reset_date = current_date

    def _process_completed_trades(self):
        """Process completed trades and update statistics."""
        # Get completed trades from real-time monitor
        monitor_trades = self.real_time_monitor.active_trades
        
        # Check for trades that are no longer active
        for trade_id in list(self.active_trades.keys()):
            if trade_id not in monitor_trades:
                # Trade was completed
                if trade_id in self.real_time_monitor.trade_history:
                    completed_trade = self.real_time_monitor.trade_history[-1]
                    
                    # Update statistics
                    pnl_percent = completed_trade["profit_loss_percent"]
                    self.total_pnl += pnl_percent
                    self.daily_pnl += pnl_percent
                    self.daily_trades += 1
                    
                    # Update win rate
                    if pnl_percent > 0:
                        self.win_rate = (self.win_rate * (self.daily_trades - 1) + 1) / self.daily_trades
                    else:
                        self.win_rate = (self.win_rate * (self.daily_trades - 1)) / self.daily_trades
                    
                    # Update max drawdown
                    if pnl_percent < self.max_drawdown:
                        self.max_drawdown = pnl_percent
                    
                    # Move to history
                    self.trade_history.append(completed_trade)
                    del self.active_trades[trade_id]
                    
                    self.logger.info(f"Trade {trade_id} completed: {pnl_percent:.2f}% P&L")

    def _log_performance_summary(self):
        """Log current performance summary."""
        if self.active_trades:
            summary = self.real_time_monitor.get_trade_summary()
            
            self.logger.info(
                f"Performance Summary - "
                f"Active: {summary['active_trades']}, "
                f"Daily P&L: {self.daily_pnl:.2f}%, "
                f"Total P&L: {self.total_pnl:.2f}%, "
                f"Win Rate: {self.win_rate:.1%}, "
                f"Max DD: {self.max_drawdown:.2f}%"
            )

    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary."""
        return {
            "total_pnl_percent": self.total_pnl,
            "daily_pnl_percent": self.daily_pnl,
            "win_rate": self.win_rate,
            "max_drawdown_percent": self.max_drawdown,
            "total_trades": len(self.trade_history),
            "daily_trades": self.daily_trades,
            "active_trades": len(self.active_trades),
            "max_daily_loss": self.max_daily_loss
        }

    def get_active_trades(self) -> Dict[str, Dict]:
        """Get all active trades with detailed information."""
        return self.real_time_monitor.get_active_trades()

    def manual_exit_trade(self, trade_id: str, exit_price: float = None) -> bool:
        """Manually exit a specific trade."""
        return self.real_time_monitor.manual_exit_trade(trade_id, exit_price)

    def get_trade_details(self, trade_id: str) -> Optional[Dict]:
        """Get detailed information about a specific trade."""
        return self.real_time_monitor.get_trade_details(trade_id)

    def stop_all_trades(self):
        """Stop all active trades."""
        active_trades = list(self.real_time_monitor.active_trades.keys())
        
        for trade_id in active_trades:
            self.real_time_monitor.manual_exit_trade(trade_id)
        
        self.logger.info(f"Stopped {len(active_trades)} active trades")

    def get_exit_suggestions(self, trade_id: str) -> List[Dict]:
        """Get exit suggestions for a specific trade."""
        trade_data = self.real_time_monitor.get_trade_details(trade_id)
        if trade_data:
            return trade_data.get("dynamic_exit_suggestions", [])
        return [] 