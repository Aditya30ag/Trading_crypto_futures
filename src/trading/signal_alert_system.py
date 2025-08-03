import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from src.data.fetcher import CoinDCXFetcher
from src.data.indicators import TechnicalIndicators
from src.utils.logger import setup_logger
import threading
import os


class SignalAlertSystem:
    """
    Enhanced alert system for signal monitoring with real-time notifications
    and profit/loss tracking with proper signal confirmation.
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger()
        self.fetcher = CoinDCXFetcher()
        self.indicators = TechnicalIndicators()
        
        # Alert settings
        self.alert_interval = 30  # seconds
        self.profit_alert_threshold = 1.0  # 1% profit alert
        self.loss_alert_threshold = -0.5   # -0.5% loss alert
        self.critical_loss_threshold = -2.0 # -2% critical loss
        self.signal_confirmation_threshold = 3  # minutes
        
        # Enhanced signal validation settings - RELAXED THRESHOLDS
        self.min_score_threshold = 6  # Reduced from 7
        self.min_confidence_threshold = 0.4  # Reduced from 0.5
        self.max_price_deviation = 0.05  # Increased from 0.02 (5% instead of 2%)
        self.min_volume_threshold = 50000  # Reduced from 100000
        
        # Alert history
        self.alert_history = []
        self.active_alerts = {}
        self.confirmed_signals = {}  # Track confirmed signals
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        
        self.logger.info("SignalAlertSystem initialized with relaxed thresholds")

    def start_monitoring(self):
        """Start the alert monitoring system."""
        if self.monitoring_active:
            self.logger.warning("Alert monitoring is already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._alert_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Alert monitoring started with enhanced validation")

    def stop_monitoring(self):
        """Stop the alert monitoring system."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("Alert monitoring stopped")

    def _alert_loop(self):
        """Main alert monitoring loop with enhanced validation."""
        self.logger.info("Starting enhanced alert monitoring loop...")
        
        while self.monitoring_active:
            try:
                # Check for active signals and generate alerts with validation
                self._check_signal_alerts_with_validation()
                
                # Check for market condition alerts
                self._check_market_alerts()
                
                # Save alert data
                self._save_alert_data()
                
                # Wait before next check
                time.sleep(self.alert_interval)
                
            except Exception as e:
                self.logger.error(f"Error in alert loop: {e}")
                time.sleep(10)  # Wait before retrying
        
        self.logger.info("Alert monitoring loop ended")

    def _check_signal_alerts_with_validation(self):
        """Check for signal-specific alerts with enhanced validation."""
        try:
            # Load latest monitoring data
            if os.path.exists("latest_monitoring_data.json"):
                with open("latest_monitoring_data.json", "r") as f:
                    monitoring_data = json.load(f)
                
                active_signals = monitoring_data.get("active_signals", {})
                
                for signal_id, signal_data in active_signals.items():
                    # Enhanced validation before processing
                    if self._validate_signal_before_alert(signal_id, signal_data):
                        self._process_signal_alert(signal_id, signal_data)
                    else:
                        self.logger.warning(f"Signal {signal_id} failed validation - skipping alert")
                    
        except Exception as e:
            self.logger.error(f"Error checking signal alerts: {e}")

    def _validate_signal_before_alert(self, signal_id: str, signal_data: Dict) -> bool:
        """Enhanced validation of signal before generating alert."""
        try:
            signal = signal_data.get("signal", {})
            symbol = signal.get("symbol")
            side = signal.get("side")
            score = signal.get("score", 0)
            confidence = signal.get("confidence", 0.0)
            entry_price = signal.get("entry_price")
            
            # Basic validation checks
            if not all([symbol, side, entry_price]):
                self.logger.warning(f"Missing basic signal data for {signal_id}")
                return False
            
            # Score and confidence validation
            if score < self.min_score_threshold:
                self.logger.warning(f"Signal {signal_id} score too low: {score} < {self.min_score_threshold}")
                return False
            
            if confidence < self.min_confidence_threshold:
                self.logger.warning(f"Signal {signal_id} confidence too low: {confidence} < {self.min_confidence_threshold}")
                return False
            
            # Get fresh market data for validation
            market_data = self.fetcher.fetch_market_data(symbol)
            if not market_data:
                self.logger.warning(f"Could not fetch market data for {symbol} validation")
                return False
            
            current_price = market_data["last_price"]
            
            # Price deviation check - More lenient for volatile markets
            price_deviation = abs(current_price - entry_price) / entry_price
            if price_deviation > self.max_price_deviation:
                self.logger.warning(f"Price deviation high for {symbol}: "
                                  f"{price_deviation:.2%} > {self.max_price_deviation:.2%}")
                # Don't fail validation for price deviation, just warn
                # return False
            
            # Volume validation - More lenient for smaller coins
            volume_24h = market_data.get("volume_24h", 0)
            if volume_24h < self.min_volume_threshold:
                self.logger.warning(f"Volume low for {symbol}: "
                                  f"{volume_24h} < {self.min_volume_threshold}")
                # Don't fail validation for low volume, just warn
                # return False
            
            # Technical indicator validation
            if not self._validate_technical_indicators(signal_id, signal_data, current_price):
                return False
            
            # Direction validation
            if not self._validate_signal_direction(signal_id, signal_data, current_price):
                return False
            
            # Signal confirmation check
            if not self._confirm_signal_with_fresh_data(signal_id, signal_data):
                return False
            
            self.logger.info(f"✅ Signal {signal_id} passed all validation checks")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating signal {signal_id}: {e}")
            return False

    def _validate_technical_indicators(self, signal_id: str, signal_data: Dict, current_price: float) -> bool:
        """Validate technical indicators are correctly calculated."""
        try:
            signal = signal_data.get("signal", {})
            symbol = signal.get("symbol")
            side = signal.get("side")
            indicators = signal.get("indicators", {})
            
            # Get fresh candlestick data
            candles = self.fetcher.fetch_candlestick_data(symbol, "15m", limit=50)
            if not candles or len(candles) < 30:
                self.logger.warning(f"Insufficient candlestick data for {symbol} validation")
                return False
            
            # Recalculate key indicators
            fresh_rsi = self.indicators.calculate_rsi(candles)
            fresh_ema_20 = self.indicators.calculate_ema(candles, 20)
            fresh_ema_50 = self.indicators.calculate_ema(candles, 50)
            fresh_macd = self.indicators.calculate_macd(candles)
            
            if any(x is None for x in [fresh_rsi, fresh_ema_20, fresh_ema_50, fresh_macd]):
                self.logger.warning(f"Failed to calculate fresh indicators for {symbol}")
                return False
            
            # Compare with original indicators
            original_rsi = indicators.get("rsi")
            original_ema_20 = indicators.get("ema_20")
            original_ema_50 = indicators.get("ema_50")
            original_macd = indicators.get("macd")
            
            # Check for significant deviations
            if original_rsi and abs(fresh_rsi - original_rsi) > 10:
                self.logger.warning(f"RSI deviation too high for {symbol}: {original_rsi:.2f} vs {fresh_rsi:.2f}")
                return False
            
            if original_ema_20 and abs(fresh_ema_20 - original_ema_20) / original_ema_20 > 0.01:
                self.logger.warning(f"EMA20 deviation too high for {symbol}")
                return False
            
            # Validate direction logic
            if side == "long":
                if current_price < fresh_ema_20 or fresh_ema_20 < fresh_ema_50:
                    self.logger.warning(f"Long signal validation failed for {symbol}: price={current_price}, ema20={fresh_ema_20}, ema50={fresh_ema_50}")
                    return False
            elif side == "short":
                if current_price > fresh_ema_20 or fresh_ema_20 > fresh_ema_50:
                    self.logger.warning(f"Short signal validation failed for {symbol}: price={current_price}, ema20={fresh_ema_20}, ema50={fresh_ema_50}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating technical indicators for {signal_id}: {e}")
            return False

    def _validate_signal_direction(self, signal_id: str, signal_data: Dict, current_price: float) -> bool:
        """Validate that signal direction is correct based on current market conditions."""
        try:
            signal = signal_data.get("signal", {})
            symbol = signal.get("symbol")
            side = signal.get("side")
            indicators = signal.get("indicators", {})
            
            # Get fresh candlestick data
            candles = self.fetcher.fetch_candlestick_data(symbol, "15m", limit=50)
            if not candles or len(candles) < 30:
                return False
            
            # Calculate fresh indicators
            fresh_rsi = self.indicators.calculate_rsi(candles)
            fresh_ema_20 = self.indicators.calculate_ema(candles, 20)
            fresh_ema_50 = self.indicators.calculate_ema(candles, 50)
            fresh_macd = self.indicators.calculate_macd(candles)
            
            if any(x is None for x in [fresh_rsi, fresh_ema_20, fresh_ema_50, fresh_macd]):
                return False
            
            # Direction validation logic
            if side == "long":
                # For long signals, check bullish conditions
                bullish_conditions = [
                    current_price > fresh_ema_20,  # Price above EMA20
                    fresh_ema_20 > fresh_ema_50,  # EMA20 above EMA50
                    fresh_macd > 0,  # MACD positive
                    fresh_rsi > 40 and fresh_rsi < 70  # RSI in bullish range
                ]
                
                if not all(bullish_conditions):
                    self.logger.warning(f"Long signal direction validation failed for {symbol}")
                    return False
                    
            elif side == "short":
                # For short signals, check bearish conditions
                bearish_conditions = [
                    current_price < fresh_ema_20,  # Price below EMA20
                    fresh_ema_20 < fresh_ema_50,  # EMA20 below EMA50
                    fresh_macd < 0,  # MACD negative
                    fresh_rsi < 60 and fresh_rsi > 30  # RSI in bearish range
                ]
                
                if not all(bearish_conditions):
                    self.logger.warning(f"Short signal direction validation failed for {symbol}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating signal direction for {signal_id}: {e}")
            return False

    def _confirm_signal_with_fresh_data(self, signal_id: str, signal_data: Dict) -> bool:
        """Confirm signal with fresh market data."""
        try:
            signal = signal_data.get("signal", {})
            symbol = signal.get("symbol")
            side = signal.get("side")
            entry_price = signal.get("entry_price")
            
            # Get fresh market data
            market_data = self.fetcher.fetch_market_data(symbol)
            if not market_data:
                return False
            
            current_price = market_data["last_price"]
            
            # Check if price is still in valid range
            price_change = ((current_price - entry_price) / entry_price) * 100
            
            if side == "long":
                # For long signals, price should not have dropped significantly
                if price_change < -3:  # More than 3% drop
                    self.logger.warning(f"Long signal {signal_id} price dropped too much: {price_change:.2f}%")
                    return False
            elif side == "short":
                # For short signals, price should not have risen significantly
                if price_change > 3:  # More than 3% rise
                    self.logger.warning(f"Short signal {signal_id} price rose too much: {price_change:.2f}%")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error confirming signal with fresh data for {signal_id}: {e}")
            return False

    def _process_signal_alert(self, signal_id: str, signal_data: Dict):
        """Process alerts for a validated signal."""
        try:
            signal = signal_data["signal"]
            symbol = signal["symbol"]
            side = signal["side"]
            entry_price = signal["entry_price"]
            score = signal.get("score", 0)
            confidence = signal.get("confidence", 0.0)
            current_price = signal_data.get("current_price", entry_price)
            
            # Get fresh market data
            market_data = self.fetcher.fetch_market_data(symbol)
            if not market_data:
                return
            
            current_price = market_data["last_price"]
            
            # Calculate current P&L
            if side == "long":
                profit_loss_percent = ((current_price - entry_price) / entry_price) * 100
            else:
                profit_loss_percent = ((entry_price - current_price) / entry_price) * 100
            
            # Generate validation success alert
            self._generate_validation_success_alert(signal_id, symbol, side, score, confidence, profit_loss_percent)
            
            # Check for profit alerts
            if profit_loss_percent >= self.profit_alert_threshold:
                self._generate_profit_alert(signal_id, symbol, profit_loss_percent, side)
            
            # Check for loss alerts
            if profit_loss_percent <= self.loss_alert_threshold:
                self._generate_loss_alert(signal_id, symbol, profit_loss_percent, side)
            
            # Check for critical loss alerts
            if profit_loss_percent <= self.critical_loss_threshold:
                self._generate_critical_loss_alert(signal_id, symbol, profit_loss_percent, side)
            
            # Check for signal confirmation alerts
            self._check_signal_confirmation(signal_id, signal_data)
            
            # Check for technical indicator alerts
            self._check_technical_alerts(signal_id, signal_data, current_price)
            
        except Exception as e:
            self.logger.error(f"Error processing signal alert for {signal_id}: {e}")

    def _generate_validation_success_alert(self, signal_id: str, symbol: str, side: str, score: int, confidence: float, profit_loss_percent: float):
        """Generate validation success alert."""
        alert_id = f"{signal_id}_validation_success"
        
        if alert_id not in self.active_alerts:
            alert = {
                "alert_id": alert_id,
                "signal_id": signal_id,
                "symbol": symbol,
                "type": "validation_success",
                "message": f"✅ VALIDATED SIGNAL: {symbol} {side.upper()} (Score: {score}, Confidence: {confidence:.1%}, P&L: {profit_loss_percent:.2f}%)",
                "priority": "high",
                "timestamp": datetime.now(),
                "score": score,
                "confidence": confidence,
                "profit_loss_percent": profit_loss_percent,
                "side": side
            }
            
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            
            self.logger.info(f"🚨 {alert['message']}")
            
            # Save to alert file
            self._save_alert_to_file(alert)

    def _generate_profit_alert(self, signal_id: str, symbol: str, profit_percent: float, side: str):
        """Generate profit alert."""
        alert_id = f"{signal_id}_profit_{int(profit_percent)}"
        
        if alert_id not in self.active_alerts:
            alert = {
                "alert_id": alert_id,
                "signal_id": signal_id,
                "symbol": symbol,
                "type": "profit",
                "message": f"🎯 PROFIT ALERT: {symbol} {side.upper()} is {profit_percent:.2f}% in profit!",
                "priority": "high" if profit_percent >= 2.0 else "medium",
                "timestamp": datetime.now(),
                "profit_percent": profit_percent,
                "side": side
            }
            
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            
            self.logger.info(f"🚨 {alert['message']}")
            
            # Save to alert file
            self._save_alert_to_file(alert)

    def _generate_loss_alert(self, signal_id: str, symbol: str, loss_percent: float, side: str):
        """Generate loss alert."""
        alert_id = f"{signal_id}_loss_{int(abs(loss_percent))}"
        
        if alert_id not in self.active_alerts:
            alert = {
                "alert_id": alert_id,
                "signal_id": signal_id,
                "symbol": symbol,
                "type": "loss",
                "message": f"⚠️ LOSS ALERT: {symbol} {side.upper()} is {loss_percent:.2f}% in loss",
                "priority": "medium",
                "timestamp": datetime.now(),
                "loss_percent": loss_percent,
                "side": side
            }
            
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            
            self.logger.warning(f"🚨 {alert['message']}")
            
            # Save to alert file
            self._save_alert_to_file(alert)

    def _generate_critical_loss_alert(self, signal_id: str, symbol: str, loss_percent: float, side: str):
        """Generate critical loss alert."""
        alert_id = f"{signal_id}_critical_loss_{int(abs(loss_percent))}"
        
        if alert_id not in self.active_alerts:
            alert = {
                "alert_id": alert_id,
                "signal_id": signal_id,
                "symbol": symbol,
                "type": "critical_loss",
                "message": f"🚨 CRITICAL LOSS: {symbol} {side.upper()} is {loss_percent:.2f}% in loss - CONSIDER EXIT!",
                "priority": "critical",
                "timestamp": datetime.now(),
                "loss_percent": loss_percent,
                "side": side
            }
            
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            
            self.logger.critical(f"🚨 {alert['message']}")
            
            # Save to alert file
            self._save_alert_to_file(alert)

    def _check_signal_confirmation(self, signal_id: str, signal_data: Dict):
        """Check if signal needs confirmation after initial generation."""
        try:
            entry_time = signal_data.get("entry_time")
            if not entry_time:
                return
            
            # Convert string to datetime if needed
            if isinstance(entry_time, str):
                entry_time = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
            
            time_elapsed = datetime.now() - entry_time
            
            # Alert if signal hasn't moved in expected direction after confirmation threshold
            if time_elapsed.total_seconds() / 60 >= self.signal_confirmation_threshold:
                symbol = signal_data["signal"]["symbol"]
                side = signal_data["signal"]["side"]
                profit_loss_percent = signal_data.get("profit_loss_percent", 0.0)
                
                # Check if signal is not performing as expected
                if (side == "long" and profit_loss_percent < 0.1) or (side == "short" and profit_loss_percent < 0.1):
                    alert_id = f"{signal_id}_confirmation_warning"
                    
                    if alert_id not in self.active_alerts:
                        alert = {
                            "alert_id": alert_id,
                            "signal_id": signal_id,
                            "symbol": symbol,
                            "type": "confirmation_warning",
                            "message": f"⚠️ SIGNAL CONFIRMATION: {symbol} {side.upper()} not performing as expected after {time_elapsed.total_seconds()/60:.1f} minutes",
                            "priority": "medium",
                            "timestamp": datetime.now(),
                            "time_elapsed_minutes": time_elapsed.total_seconds() / 60
                        }
                        
                        self.active_alerts[alert_id] = alert
                        self.alert_history.append(alert)
                        
                        self.logger.warning(f"🚨 {alert['message']}")
                        
                        # Save to alert file
                        self._save_alert_to_file(alert)
                        
        except Exception as e:
            self.logger.error(f"Error checking signal confirmation: {e}")

    def _check_technical_alerts(self, signal_id: str, signal_data: Dict, current_price: float):
        """Check for technical indicator-based alerts."""
        try:
            symbol = signal_data["signal"]["symbol"]
            side = signal_data["signal"]["side"]
            
            # Get fresh candlestick data
            candles = self.fetcher.fetch_candlestick_data(symbol, "15m", limit=50)
            if not candles or len(candles) < 20:
                return
            
            # Calculate key indicators
            rsi = self.indicators.calculate_rsi(candles)
            ema_20 = self.indicators.calculate_ema(candles, 20)
            macd = self.indicators.calculate_macd(candles)
            
            if any(x is None for x in [rsi, ema_20, macd]):
                return
            
            # Generate technical alerts
            alerts = []
            
            # RSI alerts
            if side == "long" and rsi > 75:
                alerts.append({
                    "type": "rsi_overbought",
                    "message": f"RSI overbought ({rsi:.1f}) - consider taking profits",
                    "priority": "high"
                })
            elif side == "short" and rsi < 25:
                alerts.append({
                    "type": "rsi_oversold", 
                    "message": f"RSI oversold ({rsi:.1f}) - consider taking profits",
                    "priority": "high"
                })
            
            # Trend reversal alerts
            if side == "long" and current_price < ema_20:
                alerts.append({
                    "type": "trend_reversal",
                    "message": f"Price below EMA20 - trend may be reversing",
                    "priority": "medium"
                })
            elif side == "short" and current_price > ema_20:
                alerts.append({
                    "type": "trend_reversal",
                    "message": f"Price above EMA20 - trend may be reversing", 
                    "priority": "medium"
                })
            
            # MACD alerts
            if side == "long" and macd < 0:
                alerts.append({
                    "type": "macd_bearish",
                    "message": f"MACD turned bearish ({macd:.6f}) - monitor closely",
                    "priority": "medium"
                })
            elif side == "short" and macd > 0:
                alerts.append({
                    "type": "macd_bullish",
                    "message": f"MACD turned bullish ({macd:.6f}) - monitor closely",
                    "priority": "medium"
                })
            
            # Generate alerts
            for alert_info in alerts:
                alert_id = f"{signal_id}_{alert_info['type']}"
                
                if alert_id not in self.active_alerts:
                    alert = {
                        "alert_id": alert_id,
                        "signal_id": signal_id,
                        "symbol": symbol,
                        "type": alert_info["type"],
                        "message": f"📊 TECHNICAL: {symbol} {side.upper()} - {alert_info['message']}",
                        "priority": alert_info["priority"],
                        "timestamp": datetime.now(),
                        "side": side
                    }
                    
                    self.active_alerts[alert_id] = alert
                    self.alert_history.append(alert)
                    
                    if alert_info["priority"] == "high":
                        self.logger.warning(f"🚨 {alert['message']}")
                    else:
                        self.logger.info(f"📊 {alert['message']}")
                    
                    # Save to alert file
                    self._save_alert_to_file(alert)
                    
        except Exception as e:
            self.logger.error(f"Error checking technical alerts: {e}")

    def _check_market_alerts(self):
        """Check for general market condition alerts."""
        try:
            # Get top movers to identify market trends
            top_movers = self.fetcher.fetch_top_movers(top_n=10, min_volume=100000)
            
            if top_movers:
                # Check for extreme market conditions
                extreme_moves = [m for m in top_movers if abs(m[1]) > 10]  # >10% moves
                
                if extreme_moves:
                    alert = {
                        "alert_id": f"market_extreme_{int(time.time())}",
                        "type": "market_extreme",
                        "message": f"🌊 MARKET ALERT: {len(extreme_moves)} symbols with >10% moves detected",
                        "priority": "high",
                        "timestamp": datetime.now(),
                        "extreme_moves": extreme_moves[:5]  # Top 5
                    }
                    
                    self.alert_history.append(alert)
                    self.logger.warning(f"🚨 {alert['message']}")
                    
                    # Save to alert file
                    self._save_alert_to_file(alert)
                    
        except Exception as e:
            self.logger.error(f"Error checking market alerts: {e}")

    def _save_alert_to_file(self, alert: Dict):
        """Save alert to file for persistence."""
        try:
            # Convert datetime to string for JSON serialization
            alert_copy = alert.copy()
            if isinstance(alert_copy["timestamp"], datetime):
                alert_copy["timestamp"] = alert_copy["timestamp"].isoformat()
            
            # Load existing alerts
            alerts_file = "signal_alerts.json"
            existing_alerts = []
            
            if os.path.exists(alerts_file):
                try:
                    with open(alerts_file, "r") as f:
                        existing_alerts = json.load(f)
                except:
                    existing_alerts = []
            
            # Add new alert
            existing_alerts.append(alert_copy)
            
            # Keep only last 1000 alerts
            if len(existing_alerts) > 1000:
                existing_alerts = existing_alerts[-1000:]
            
            # Save to file
            with open(alerts_file, "w") as f:
                json.dump(existing_alerts, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving alert to file: {e}")

    def _save_alert_data(self):
        """Save current alert data."""
        try:
            # Convert active alerts to serializable format
            serializable_alerts = {}
            for alert_id, alert in self.active_alerts.items():
                alert_copy = alert.copy()
                if isinstance(alert_copy["timestamp"], datetime):
                    alert_copy["timestamp"] = alert_copy["timestamp"].isoformat()
                serializable_alerts[alert_id] = alert_copy
            
            alert_data = {
                "active_alerts": serializable_alerts,
                "alert_count": len(self.active_alerts),
                "last_update": datetime.now().isoformat()
            }
            
            with open("latest_alerts.json", "w") as f:
                json.dump(alert_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving alert data: {e}")

    def get_active_alerts(self) -> List[Dict]:
        """Get all active alerts."""
        return list(self.active_alerts.values())

    def get_alert_summary(self) -> Dict:
        """Get summary of current alerts."""
        alert_types = {}
        for alert in self.active_alerts.values():
            alert_type = alert["type"]
            if alert_type not in alert_types:
                alert_types[alert_type] = 0
            alert_types[alert_type] += 1
        
        return {
            "total_alerts": len(self.active_alerts),
            "alert_types": alert_types,
            "critical_alerts": len([a for a in self.active_alerts.values() if a["priority"] == "critical"]),
            "high_priority_alerts": len([a for a in self.active_alerts.values() if a["priority"] == "high"]),
            "last_update": datetime.now().isoformat()
        }

    def clear_alert(self, alert_id: str):
        """Clear a specific alert."""
        if alert_id in self.active_alerts:
            del self.active_alerts[alert_id]
            self.logger.info(f"Cleared alert: {alert_id}")

    def clear_all_alerts(self):
        """Clear all active alerts."""
        self.active_alerts.clear()
        self.logger.info("Cleared all active alerts") 