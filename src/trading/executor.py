from src.strategies.strategy_manager import StrategyManager
from src.trading.enhanced_signal_generator import EnhancedSignalGenerator
from src.utils.logger import setup_logger
from src.trading.signal_monitor import SignalMonitor
import time
from src.data.fetcher import CoinDCXFetcher
import logging
import threading
import json
import os
from datetime import datetime

class TradeExecutor:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger()
        self.fetcher = CoinDCXFetcher()
        self.strategy_manager = StrategyManager(config)
        self.enhanced_signal_generator = EnhancedSignalGenerator(config)
        self.signal_monitor = SignalMonitor(config)
        self.balance = config["trading"]["initial_balance"]
        self.monitoring_active = False
        self.monitoring_thread = None
        self.logger.info(f"TradeExecutor initialized with balance: {self.balance} INR")
        # Batch writing setup
        self._signal_buffer = []
        self._buffer_lock = threading.Lock()
        self._batch_writer_thread = threading.Thread(target=self._batch_writer_loop, daemon=True)
        self._batch_writer_running = True
        self._batch_writer_thread.start()

    def _batch_writer_loop(self):
        """Background thread to flush signal buffer to file every second."""
        while self._batch_writer_running:
            self._flush_signal_buffer()
            time.sleep(1)

    def _flush_signal_buffer(self):
        with self._buffer_lock:
            if not self._signal_buffer:
                return
            filename = "latest_signals.json"
            # Load existing signals
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    try:
                        signals = json.load(f)
                    except Exception:
                        signals = []
            else:
                signals = []
            # Remove duplicates and add new signals
            existing_keys = set((s.get('symbol'), s.get('strategy'), s.get('timestamp')) for s in signals)
            for signal in self._signal_buffer:
                key = (signal.get('symbol'), signal.get('strategy'), signal.get('timestamp'))
                if key not in existing_keys:
                    signals.append(signal)
                    existing_keys.add(key)
            # Keep only last 100 signals
            
            def safe_timestamp_key(signal):
                timestamp = signal.get('timestamp', '')
                if isinstance(timestamp, str):
                    return timestamp
                elif hasattr(timestamp, 'isoformat'):
                    return timestamp.isoformat()
                else:
                    return str(timestamp)
            
            signals = sorted(signals, key=safe_timestamp_key, reverse=True)[:100]
            with open(filename, 'w') as f:
                json.dump(signals, f, indent=2, default=str)
            self._signal_buffer.clear()

    def shutdown(self):
        """Call this before exiting to flush remaining signals and stop the batch writer."""
        self._batch_writer_running = False
        self._batch_writer_thread.join()
        self._flush_signal_buffer()

    def run(self):
        """Generate and log trading signals, then start monitoring."""
        self.logger.info("Starting signal generation...")
        
        # Use enhanced signal generator instead of old strategy manager
        signals = self.enhanced_signal_generator.generate_entry_signals()
        
        # Convert EntrySignal objects to dictionaries for compatibility
        signal_dicts = []
        for signal in signals:
            signal_dict = signal.to_dict()
            # Convert TradeDirection enum to string for compatibility
            signal_dict["side"] = signal.direction.value
            # Also convert any other enum objects in the signal
            if "indicators" in signal_dict:
                for key, value in signal_dict["indicators"].items():
                    if hasattr(value, 'value') and hasattr(value, '__class__'):  # Enum objects
                        signal_dict["indicators"][key] = value.value
            signal_dicts.append(signal_dict)
        
        self.logger.info(f"Generated {len(signal_dicts)} signals")
        
        if not signal_dicts:
            self.logger.warning("No signals generated - this could be due to strict strategy requirements or API issues")
            return signal_dicts
        
        # Add signals to monitoring
        for signal in signal_dicts:
            score = signal.get("score", 0)
            strategy = signal.get("strategy", "unknown")
            symbol = signal.get("symbol", "unknown")
            
            # DYNAMIC: Calculate position size based on signal strength and market conditions
            base_position_size_ratio = self.config["trading"]["position_size_ratio"]
            
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
                strategy_multiplier = 0.9  # Smaller positions for scalping (faster trades)
            elif strategy == "long_swing":
                strategy_multiplier = 1.1  # Larger positions for long swing (higher confidence)
            
            # Adjust based on estimated profit
            estimated_profit = signal.get("estimated_profit_inr", 100)
            profit_multiplier = 1.0
            if estimated_profit >= 500:
                profit_multiplier = 1.2  # Larger position for high-profit signals
            elif estimated_profit <= 100:
                profit_multiplier = 0.8  # Smaller position for low-profit signals
            
            # Calculate final position size
            final_position_size_ratio = base_position_size_ratio * score_multiplier * strategy_multiplier * profit_multiplier
            position_size_inr = self.balance * final_position_size_ratio
            
            # Calculate USDT position size
            usdt_inr_rate = self.fetcher.fetch_usdt_inr_rate() or 93.0
            position_size_usdt = position_size_inr / usdt_inr_rate
            
            # Calculate quantity based on entry price and leverage
            entry_price = signal.get("entry_price", 0)
            leverage = 25  # 25x leverage
            quantity = (position_size_usdt / entry_price) * leverage if entry_price > 0 else 0
            
            # Calculate notional value
            notional_value = quantity * entry_price
            
            self.logger.info(f"Processing {strategy} signal for {symbol} with score {score}")
            self.logger.info(f"📊 Position Details:")
            self.logger.info(f"   💰 Position Size: ₹{position_size_inr:.2f} (${position_size_usdt:.2f})")
            self.logger.info(f"   📈 Quantity: {quantity:.6f} {symbol.split('_')[0]}")
            self.logger.info(f"   💵 Notional Value: ${notional_value:.2f}")
            self.logger.info(f"   ⚡ Leverage: {leverage}x")
            self.logger.info(f"   🎯 Entry Price: ${entry_price:.6f}")
            self.logger.info(f"   📊 Multipliers: Score={score_multiplier:.1f}x, Strategy={strategy_multiplier:.1f}x, Profit={profit_multiplier:.1f}x")
            
            # Log all signals regardless of score for debugging
            self.logger.info(f"Signal: {signal}")
            
            # Get configuration values for signal filtering
            signal_config = self.config.get("trading", {}).get("signal_generation", {})
            min_score_threshold = signal_config.get("min_score_threshold", 5)  # Increased threshold
            profit_threshold = signal_config.get("profit_threshold", 300)  # Higher profit threshold
            confidence_threshold = signal_config.get("confidence_threshold", 0.5)  # Higher confidence
            min_profit_for_monitoring = signal_config.get("min_profit_for_monitoring", 200)  # Minimum profit for monitoring
            
            # ENHANCED: Stricter filtering for higher profit trades
            score = signal.get("score", 0)
            estimated_profit = signal.get("estimated_profit_inr", 0)
            confidence = signal.get("confidence", 0.0)
            
            # Check if signal meets strict monitoring criteria
            meets_score = score >= min_score_threshold
            meets_profit = estimated_profit >= profit_threshold
            meets_confidence = confidence >= confidence_threshold
            meets_min_profit = estimated_profit >= min_profit_for_monitoring
            
            # Signal qualifies if it meets strict criteria
            accepted = False
            if meets_score and meets_profit and meets_confidence and meets_min_profit:
                self.logger.info(f"🔥 HIGH-PROFIT SIGNAL (score {score}, profit ₹{estimated_profit:.2f}, confidence {confidence:.3f}): {signal}")
                
                # Use high-profit signal monitoring for all qualified signals
                if self.signal_monitor.add_high_profit_signal(signal):
                    self.logger.info(f"✅ Added high-profit signal to monitoring: {symbol} (Profit: ₹{estimated_profit:.2f})")
                    accepted = True
                else:
                    self.logger.warning(f"❌ Failed to add high-profit signal to monitoring: {symbol}")
            elif meets_score and estimated_profit >= min_profit_for_monitoring:
                self.logger.info(f"📊 QUALIFIED SIGNAL (score {score}, profit ₹{estimated_profit:.2f}, confidence {confidence:.3f}): {signal}")
                
                # Add qualified signal to monitoring
                if self.signal_monitor.add_signal(signal):
                    self.logger.info(f"✅ Added qualified signal to monitoring: {symbol}")
                    accepted = True
                else:
                    self.logger.warning(f"❌ Failed to add qualified signal to monitoring: {symbol}")
            else:
                self.logger.info(f"❌ REJECTED SIGNAL (score {score}, profit ₹{estimated_profit:.2f}, confidence {confidence:.3f}): {signal}")
            
            # Add accepted signal to buffer for batch writing
            if accepted:
                with self._buffer_lock:
                    self._signal_buffer.append(signal)
        
        # IMPROVED: Save monitoring data immediately after adding signals
        if self.signal_monitor.active_signals:
            self.signal_monitor.save_monitoring_data("latest_monitoring_data.json")
            self.logger.info(f"Saved {len(self.signal_monitor.active_signals)} signals to monitoring data")
        
        # Start monitoring if we have active signals
        if self.signal_monitor.active_signals:
            self.start_monitoring()
        
        return signal_dicts

    def start_monitoring(self):
        """Start the signal monitoring in a separate thread."""
        if self.monitoring_active:
            self.logger.warning("Monitoring is already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Signal monitoring started")

    def stop_monitoring(self):
        """Stop the signal monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("Signal monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop that runs in a separate thread."""
        self.logger.info("Starting monitoring loop...")
        
        while self.monitoring_active:
            try:
                # Monitor all active signals
                summary = self.signal_monitor.monitor_all_signals()
                
                # Log summary
                if summary["active_signals"] > 0:
                    self.logger.info(f"Monitoring {summary['active_signals']} signals. "
                                   f"Total P&L: {summary['total_profit_loss']:.2f}%")
                    
                    # Log high-priority alerts
                    for alert in summary["alerts"]:
                        self.logger.warning(f"ALERT: {alert['symbol']} - {alert['message']} "
                                          f"(Action: {alert['action']})")
                
                # Save monitoring data periodically
                if summary["active_signals"] > 0:
                    self.signal_monitor.save_monitoring_data("latest_monitoring_data.json")
                
                # Check if monitoring should continue
                if summary["active_signals"] == 0:
                    self.logger.info("No active signals remaining, stopping monitoring")
                    self.monitoring_active = False
                    break
                
                # Wait before next monitoring cycle
                time.sleep(self.signal_monitor.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Wait before retrying
        
        self.logger.info("Monitoring loop ended")

    def get_monitoring_summary(self):
        """Get current monitoring summary."""
        return self.signal_monitor.get_signal_summary()

    def get_active_signals(self):
        """Get details of all active signals."""
        return self.signal_monitor.active_signals

    def get_signal_suggestions(self, signal_id):
        """Get exit suggestions for a specific signal."""
        if signal_id in self.signal_monitor.active_signals:
            signal_data = self.signal_monitor.active_signals[signal_id]
            return signal_data.get("dynamic_exit_suggestions", [])
        return []

    def manual_exit_signal(self, signal_id, exit_price=None, reason="manual"):
        """Manually exit a signal."""
        if signal_id in self.signal_monitor.active_signals:
            signal_data = self.signal_monitor.active_signals[signal_id]
            
            if exit_price is None:
                # Use current market price
                symbol = signal_data["signal"]["symbol"]
                market_data = self.fetcher.fetch_market_data(symbol)
                if market_data:
                    exit_price = market_data["last_price"]
                else:
                    self.logger.error(f"Could not fetch current price for {symbol}")
                    return False
            
            result = self.signal_monitor._handle_exit(signal_id, reason, exit_price)
            self.logger.info(f"Manual exit executed for {signal_id} at {exit_price}")
            return True
        else:
            self.logger.error(f"Signal {signal_id} not found in active signals")
            return False