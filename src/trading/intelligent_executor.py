import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass

from src.strategies.strategy_manager import StrategyManager
from src.trading.intelligent_monitor import IntelligentTradeMonitor, TradePhase, SignalStrength
from src.utils.logger import setup_logger
from src.data.fetcher import CoinDCXFetcher


@dataclass
class MarketCondition:
    """Market condition analysis for intelligent trade selection"""
    symbol: str
    current_price: float
    volatility_score: float
    momentum_score: float
    trend_score: float
    volume_score: float
    overall_score: float
    best_strategy: str
    signal_quality: float
    timestamp: datetime


class IntelligentTradeExecutor:
    """
    Intelligent trade executor that focuses on single high-quality entries
    with dynamic post-entry management based on signal journey evolution.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = setup_logger()
        self.fetcher = CoinDCXFetcher()
        self.strategy_manager = StrategyManager(config)
        self.intelligent_monitor = IntelligentTradeMonitor(config)
        
        # Trading state
        self.active_trades: Dict[str, Dict] = {}
        self.trade_history: List[Dict] = []
        self.market_conditions: Dict[str, MarketCondition] = {}
        
        # Performance tracking
        self.total_pnl = 0.0
        self.win_rate = 0.0
        self.max_drawdown = 0.0
        self.daily_pnl = 0.0
        self.daily_trades = 0
        
        # Risk management
        self.max_concurrent_trades = 3  # Reduced for focused trading
        self.max_daily_loss = 3.0  # 3% max daily loss
        self.max_position_size = 0.15  # 15% max position size
        
        # Market analysis settings
        self.analysis_interval = 30  # seconds
        self.min_market_score = 0.6
        self.min_signal_quality = 0.7
        
        self.logger.info("IntelligentTradeExecutor initialized")

    def run(self) -> List[Dict]:
        """
        Main execution loop that focuses on single high-quality trades
        with intelligent market analysis and dynamic management.
        """
        self.logger.info("Starting intelligent trade execution...")
        
        try:
            # Analyze market conditions for all instruments
            market_analysis = self._analyze_all_markets()
            
            if not market_analysis:
                self.logger.info("No suitable market conditions found")
                return []
            
            # Select the best trading opportunities
            best_opportunities = self._select_best_opportunities(market_analysis)
            
            self.logger.info(f"Found {len(best_opportunities)} high-quality opportunities")
            
            # Execute trades for the best opportunities
            executed_trades = []
            for opportunity in best_opportunities:
                if self._can_execute_trade(opportunity):
                    trade_id = self._execute_intelligent_trade(opportunity)
                    if trade_id:
                        executed_trades.append({
                            "trade_id": trade_id,
                            "opportunity": opportunity
                        })
                        self.logger.info(f"Executed intelligent trade {trade_id} for {opportunity.symbol}")
            
            # Start intelligent monitoring if we have active trades
            if executed_trades:
                self.logger.info(f"Started intelligent monitoring for {len(executed_trades)} trades")
                self._start_performance_monitoring()
            
            return executed_trades
            
        except Exception as e:
            self.logger.error(f"Error in intelligent trade execution: {e}")
            return []

    def _analyze_all_markets(self) -> List[MarketCondition]:
        """Analyze market conditions for all instruments."""
        market_conditions = []
        
        try:
            # Get top instruments
            instruments = self._get_top_instruments()
            
            self.logger.info(f"Analyzing market conditions for {len(instruments)} instruments")
            
            for symbol in instruments:
                try:
                    # Fetch market data
                    market_data = self.fetcher.fetch_market_data(symbol)
                    if not market_data:
                        continue
                    
                    # Analyze market conditions
                    condition = self._analyze_market_condition(symbol, market_data)
                    if condition and condition.overall_score >= self.min_market_score:
                        market_conditions.append(condition)
                        self.market_conditions[symbol] = condition
                    
                    # Small delay to avoid API rate limits
                    time.sleep(0.1)
                    
                except Exception as e:
                    self.logger.error(f"Error analyzing {symbol}: {e}")
                    continue
            
            # Sort by overall score
            market_conditions.sort(key=lambda x: x.overall_score, reverse=True)
            
            self.logger.info(f"Found {len(market_conditions)} instruments with good market conditions")
            
        except Exception as e:
            self.logger.error(f"Error in market analysis: {e}")
        
        return market_conditions

    def _get_top_instruments(self) -> List[str]:
        """Get top instruments for analysis."""
        try:
            # Use configured instruments or fetch from API
            instruments = self.config["trading"]["instruments"]
            
            # Limit to top 20 for focused analysis
            return instruments[:20]
            
        except Exception as e:
            self.logger.error(f"Error getting instruments: {e}")
            return []

    def _analyze_market_condition(self, symbol: str, market_data: Dict) -> Optional[MarketCondition]:
        """Analyze market condition for a specific symbol."""
        try:
            current_price = market_data["last_price"]
            
            # Fetch recent candles for technical analysis
            candles = self.fetcher.fetch_candlestick_data(symbol, "5m", 100)
            if not candles or len(candles) < 50:
                return None
            
            # Calculate technical indicators
            from src.data.indicators import TechnicalIndicators
            indicators = TechnicalIndicators()
            
            rsi = indicators.calculate_rsi(candles)
            macd_data = indicators.calculate_macd_swing(candles)
            ema_20 = indicators.calculate_ema(candles, 20)
            ema_50 = indicators.calculate_ema(candles, 50)
            atr = indicators.calculate_atr(candles)
            bb_upper, bb_lower = indicators.calculate_bollinger_bands(candles)
            
            # Calculate market condition scores
            volatility_score = self._calculate_volatility_score(atr, current_price)
            momentum_score = self._calculate_momentum_score(rsi, macd_data)
            trend_score = self._calculate_trend_score(ema_20, ema_50, current_price)
            volume_score = self._calculate_volume_score(candles)
            
            # Calculate overall score
            overall_score = (
                volatility_score * 0.2 +
                momentum_score * 0.3 +
                trend_score * 0.3 +
                volume_score * 0.2
            )
            
            # Select best strategy for current conditions
            best_strategy = self._select_best_strategy_for_conditions(
                volatility_score, momentum_score, trend_score
            )
            
            # Calculate signal quality
            signal_quality = self._calculate_signal_quality(
                rsi, macd_data, ema_20, ema_50, current_price
            )
            
            return MarketCondition(
                symbol=symbol,
                current_price=current_price,
                volatility_score=volatility_score,
                momentum_score=momentum_score,
                trend_score=trend_score,
                volume_score=volume_score,
                overall_score=overall_score,
                best_strategy=best_strategy,
                signal_quality=signal_quality,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing market condition for {symbol}: {e}")
            return None

    def _select_best_opportunities(self, market_conditions: List[MarketCondition]) -> List[MarketCondition]:
        """Select the best trading opportunities."""
        # Filter by minimum requirements
        qualified_conditions = [
            condition for condition in market_conditions
            if (condition.overall_score >= self.min_market_score and
                condition.signal_quality >= self.min_signal_quality)
        ]
        
        # Sort by overall score and signal quality
        qualified_conditions.sort(
            key=lambda x: (x.overall_score * 0.7 + x.signal_quality * 0.3),
            reverse=True
        )
        
        # Return top opportunities (limit to max concurrent trades)
        return qualified_conditions[:self.max_concurrent_trades]

    def _can_execute_trade(self, opportunity: MarketCondition) -> bool:
        """Check if we can execute a trade for this opportunity."""
        # Check maximum concurrent trades
        if len(self.active_trades) >= self.max_concurrent_trades:
            return False
        
        # Check if we already have a position in this symbol
        if opportunity.symbol in [trade["symbol"] for trade in self.active_trades.values()]:
            return False
        
        # Check daily loss limit
        if self.daily_pnl < -self.max_daily_loss:
            self.logger.warning(f"Daily loss limit reached: {self.daily_pnl:.2f}%")
            return False
        
        # Check position size limit
        position_value = opportunity.current_price * self.config["trading"]["position_size_ratio"]
        account_value = self.config["trading"]["initial_balance"]
        position_size_percent = (position_value / account_value) * 100
        
        if position_size_percent > self.max_position_size * 100:
            return False
        
        return True

    def _execute_intelligent_trade(self, opportunity: MarketCondition) -> Optional[str]:
        """Execute an intelligent trade based on market opportunity."""
        try:
            # Create market conditions dict for the monitor
            market_conditions = {
                "current_price": opportunity.current_price,
                "analysis": {
                    "volatility_score": opportunity.volatility_score,
                    "momentum_score": opportunity.momentum_score,
                    "trend_score": opportunity.trend_score,
                    "indicators": {
                        "rsi": 50,  # Will be calculated by monitor
                        "macd": {},
                        "ema_20": 0,
                        "ema_50": 0,
                        "bb_upper": 0,
                        "bb_lower": 0,
                        "atr": 0
                    }
                }
            }
            
            # Use intelligent monitor to evaluate and enter trade
            trade_id = self.intelligent_monitor.evaluate_and_enter_trade(
                opportunity.symbol, market_conditions
            )
            
            if trade_id:
                # Add to active trades tracking
                self.active_trades[trade_id] = {
                    "symbol": opportunity.symbol,
                    "strategy": opportunity.best_strategy,
                    "entry_price": opportunity.current_price,
                    "entry_time": datetime.now(),
                    "market_score": opportunity.overall_score,
                    "signal_quality": opportunity.signal_quality,
                    "status": "active"
                }
                
                self.logger.info(f"Executed intelligent trade {trade_id} for {opportunity.symbol}")
                return trade_id
            
        except Exception as e:
            self.logger.error(f"Error executing intelligent trade for {opportunity.symbol}: {e}")
        
        return None

    def _start_performance_monitoring(self):
        """Start performance monitoring in a separate thread."""
        def monitor_performance():
            while self.active_trades or self.intelligent_monitor.active_trades:
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
        if hasattr(self, 'last_reset_date') and current_date != self.last_reset_date:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.last_reset_date = current_date
        elif not hasattr(self, 'last_reset_date'):
            self.last_reset_date = current_date

    def _process_completed_trades(self):
        """Process completed trades and update statistics."""
        # Get completed trades from intelligent monitor
        monitor_trades = self.intelligent_monitor.active_trades
        
        # Check for trades that are no longer active
        for trade_id in list(self.active_trades.keys()):
            if trade_id not in monitor_trades:
                # Trade was completed
                if trade_id in [t.get("trade_id") for t in self.intelligent_monitor.trade_history]:
                    completed_trade = next(
                        t for t in self.intelligent_monitor.trade_history 
                        if t.get("trade_id") == trade_id
                    )
                    
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
                    
                    self.logger.info(f"Intelligent trade {trade_id} completed: {pnl_percent:.2f}% P&L")

    def _log_performance_summary(self):
        """Log current performance summary."""
        if self.active_trades:
            summary = self.intelligent_monitor.get_trade_summary()
            
            self.logger.info(
                f"Intelligent Performance Summary - "
                f"Active: {summary['active_trades']}, "
                f"Daily P&L: {self.daily_pnl:.2f}%, "
                f"Total P&L: {self.total_pnl:.2f}%, "
                f"Win Rate: {self.win_rate:.1%}, "
                f"Max DD: {self.max_drawdown:.2f}%"
            )

    # Helper methods for market analysis
    def _calculate_volatility_score(self, atr: float, price: float) -> float:
        """Calculate volatility score."""
        if not atr or not price:
            return 0.5
        volatility_percent = (atr / price) * 100
        # Normalize: 0-2% = good, 2-5% = moderate, >5% = high
        if volatility_percent <= 2:
            return 0.8
        elif volatility_percent <= 5:
            return 0.5
        else:
            return 0.2

    def _calculate_momentum_score(self, rsi: float, macd_data: Dict) -> float:
        """Calculate momentum score."""
        rsi_score = 0.5
        if rsi:
            # RSI between 40-60 is neutral, 30-40 or 60-70 is good momentum
            if 30 <= rsi <= 40 or 60 <= rsi <= 70:
                rsi_score = 0.8
            elif 40 < rsi < 60:
                rsi_score = 0.5
            else:
                rsi_score = 0.2
        
        macd_score = 0.5
        if macd_data and "histogram" in macd_data:
            histogram = abs(macd_data["histogram"])
            if histogram > 0.01:
                macd_score = 0.8
            elif histogram > 0.005:
                macd_score = 0.6
            else:
                macd_score = 0.3
        
        return (rsi_score + macd_score) / 2

    def _calculate_trend_score(self, ema_20: float, ema_50: float, current_price: float) -> float:
        """Calculate trend score."""
        if not all([ema_20, ema_50, current_price]):
            return 0.5
        
        # Check if price is above both EMAs (uptrend)
        if current_price > ema_20 > ema_50:
            return 0.9
        elif current_price > ema_20:
            return 0.7
        elif current_price < ema_20 < ema_50:
            return 0.1  # Downtrend
        else:
            return 0.4  # Mixed signals

    def _calculate_volume_score(self, candles: List[Dict]) -> float:
        """Calculate volume score."""
        if not candles or len(candles) < 20:
            return 0.5
        
        # Calculate average volume
        volumes = [float(candle["volume"]) for candle in candles[-20:]]
        avg_volume = sum(volumes) / len(volumes)
        current_volume = volumes[-1]
        
        # Compare current volume to average
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        if volume_ratio > 1.5:
            return 0.9  # High volume
        elif volume_ratio > 1.0:
            return 0.7  # Above average
        elif volume_ratio > 0.7:
            return 0.5  # Below average
        else:
            return 0.3  # Low volume

    def _select_best_strategy_for_conditions(self, volatility: float, momentum: float, trend: float) -> str:
        """Select best strategy based on market conditions."""
        # Calculate strategy scores
        scalping_score = volatility * 0.4 + momentum * 0.4 + trend * 0.2
        swing_score = volatility * 0.2 + momentum * 0.3 + trend * 0.5
        long_swing_score = volatility * 0.1 + momentum * 0.2 + trend * 0.7
        trend_score = volatility * 0.1 + momentum * 0.1 + trend * 0.8
        
        # Select best strategy
        strategies = [
            ("scalping", scalping_score),
            ("swing", swing_score),
            ("long_swing", long_swing_score),
            ("trend", trend_score)
        ]
        
        return max(strategies, key=lambda x: x[1])[0]

    def _calculate_signal_quality(self, rsi: float, macd_data: Dict, ema_20: float, ema_50: float, price: float) -> float:
        """Calculate overall signal quality."""
        quality_factors = []
        
        # RSI quality
        if rsi:
            if 30 <= rsi <= 70:
                quality_factors.append(0.8)
            else:
                quality_factors.append(0.4)
        
        # MACD quality
        if macd_data and "histogram" in macd_data:
            histogram = abs(macd_data["histogram"])
            if histogram > 0.005:
                quality_factors.append(0.8)
            else:
                quality_factors.append(0.5)
        
        # Trend quality
        if all([ema_20, ema_50, price]):
            if abs(ema_20 - ema_50) / ema_50 > 0.01:  # Clear trend
                quality_factors.append(0.8)
            else:
                quality_factors.append(0.5)
        
        return sum(quality_factors) / len(quality_factors) if quality_factors else 0.5

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
        return self.intelligent_monitor.get_active_trades()

    def get_market_conditions(self) -> Dict[str, MarketCondition]:
        """Get current market conditions analysis."""
        return self.market_conditions

    def manual_exit_trade(self, trade_id: str) -> bool:
        """Manually exit a specific trade."""
        return self.intelligent_monitor._execute_intelligent_exit(trade_id, "manual")

    def stop_all_trades(self):
        """Stop all active trades."""
        active_trades = list(self.intelligent_monitor.active_trades.keys())
        
        for trade_id in active_trades:
            self.intelligent_monitor._execute_intelligent_exit(trade_id, "manual")
        
        self.logger.info(f"Stopped {len(active_trades)} intelligent trades") 