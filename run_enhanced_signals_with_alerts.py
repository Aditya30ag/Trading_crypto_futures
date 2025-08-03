#!/usr/bin/env python3
"""
Enhanced Signal Generation with Alert System
Runs the improved signal generator with real-time alert monitoring
"""

import yaml
import json
import time
import threading
from datetime import datetime
from src.strategies.simple_enhanced_generator import SimpleEnhancedGenerator
from src.trading.signal_alert_system import SignalAlertSystem
from src.trading.signal_monitor import SignalMonitor
from src.utils.logger import setup_logger


def main():
    """Main function to run enhanced signal generation with alerts."""
    logger = setup_logger()
    
    try:
        logger.info("=== Starting Enhanced Signal Generation with Alerts ===")
        
        # Load configuration
        with open("config/config.yaml", "r") as file:
            config = yaml.safe_load(file)
        
        logger.info("Configuration loaded successfully")
        
        # Initialize components
        signal_generator = SimpleEnhancedGenerator()
        signal_monitor = SignalMonitor(config)
        alert_system = SignalAlertSystem(config)
        
        logger.info("Components initialized")
        
        # Generate signals with confirmation
        logger.info("Generating enhanced signals with confirmation...")
        signals = signal_generator.generate_signals()
        
        if not signals:
            logger.warning("No signals generated - check market conditions")
            return
        
        logger.info(f"Generated {len(signals)} confirmed signals")
        
        # Add signals to monitoring
        high_profit_count = 0
        regular_count = 0
        
        for signal in signals:
            signal_dict = signal.copy()
            signal_dict["side"] = signal["side"]  # Ensure side is set correctly
            
            # Convert enum objects in indicators
            if "indicators" in signal_dict:
                for key, value in signal_dict["indicators"].items():
                    if hasattr(value, 'value') and hasattr(value, '__class__'):
                        signal_dict["indicators"][key] = value.value
            
            estimated_profit = signal.get("estimated_profit_inr", 0)
            
            # Add to appropriate monitoring category
            if estimated_profit >= 500:  # High-profit threshold
                if signal_monitor.add_high_profit_signal(signal_dict):
                    high_profit_count += 1
                    logger.info(f"🔥 HIGH-PROFIT: {signal['symbol']} - "
                              f"{signal['strategy']} ({signal['timeframe']}) - "
                              f"Profit: ₹{estimated_profit:.2f}")
                else:
                    logger.warning(f"❌ Failed to add high-profit signal: {signal['symbol']}")
            elif estimated_profit >= 300:  # Qualified signal threshold
                if signal_monitor.add_signal(signal_dict):
                    regular_count += 1
                    logger.info(f"📊 QUALIFIED: {signal['symbol']} - "
                              f"{signal['strategy']} ({signal['timeframe']}) - "
                              f"Profit: ₹{estimated_profit:.2f}")
                else:
                    logger.warning(f"❌ Failed to add qualified signal: {signal['symbol']}")
            else:
                logger.info(f"❌ REJECTED: {signal['symbol']} - "
                          f"{signal['strategy']} ({signal['timeframe']}) - "
                          f"Profit: ₹{estimated_profit:.2f} (Too low)")
        
        # Save signals to file
        with open("enhanced_signals.json", "w") as f:
            json.dump(signals, f, indent=2, default=str)
        
        logger.info(f"Saved {len(signals)} signals to enhanced_signals.json")
        
        # Get monitoring summary
        priority_summary = signal_monitor.get_signal_priority_summary()
        
        logger.info("=== SIGNAL MONITORING SUMMARY ===")
        logger.info(f"Total signals monitored: {priority_summary['total_signals']}")
        logger.info(f"High-profit signals: {priority_summary['high_profit_signals']}")
        logger.info(f"Total estimated profit: ₹{priority_summary['total_estimated_profit']:.2f}")
        logger.info(f"Average profit per signal: ₹{priority_summary['average_profit_per_signal']:.2f}")
        
        # Start alert system
        logger.info("Starting alert monitoring system...")
        alert_system.start_monitoring()
        
        # Start signal monitoring
        logger.info("Starting signal monitoring...")
        signal_monitor.save_monitoring_data("latest_monitoring_data.json")
        
        # Keep the system running
        try:
            while True:
                # Get current alert summary
                alert_summary = alert_system.get_alert_summary()
                
                if alert_summary["total_alerts"] > 0:
                    logger.info(f"Active alerts: {alert_summary['total_alerts']} "
                              f"(Critical: {alert_summary['critical_alerts']}, "
                              f"High: {alert_summary['high_priority_alerts']})")
                
                # Get monitoring summary
                monitoring_summary = signal_monitor.monitor_all_signals()
                
                if monitoring_summary["active_signals"] > 0:
                    logger.info(f"Monitoring {monitoring_summary['active_signals']} signals. "
                              f"Total P&L: {monitoring_summary['total_profit_loss']:.2f}%")
                    
                    # Log high-priority alerts
                    for alert in monitoring_summary["alerts"]:
                        logger.warning(f"ALERT: {alert['symbol']} - {alert['message']} "
                                     f"(Action: {alert['action']})")
                
                # Save monitoring data periodically
                signal_monitor.save_monitoring_data("latest_monitoring_data.json")
                
                # Wait before next cycle
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
        finally:
            # Cleanup
            alert_system.stop_monitoring()
            signal_monitor.save_monitoring_data("latest_monitoring_data.json")
            logger.info("Monitoring stopped and data saved")
        
        logger.info("=== Enhanced Signal Generation with Alerts Completed ===")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


def run_signal_analysis():
    """Analyze existing signals to identify issues."""
    logger = setup_logger()
    
    try:
        logger.info("=== Analyzing Existing Signals ===")
        
        # Load latest signals
        if not os.path.exists("latest_signals.json"):
            logger.error("No signals file found")
            return
        
        with open("latest_signals.json", "r") as f:
            signals = json.load(f)
        
        logger.info(f"Analyzing {len(signals)} signals")
        
        # Analyze each signal
        for i, signal in enumerate(signals[:5]):  # Analyze top 5
            logger.info(f"\n=== Signal {i+1} Analysis ===")
            logger.info(f"Symbol: {signal['symbol']}")
            logger.info(f"Direction: {signal['direction']}")
            logger.info(f"Side: {signal['side']}")
            logger.info(f"Score: {signal['score']}")
            logger.info(f"Estimated Profit: ₹{signal.get('estimated_profit_inr', 0):.2f}")
            
            # Analyze indicators
            indicators = signal.get('indicators', {})
            logger.info("Key Indicators:")
            logger.info(f"  RSI: {indicators.get('rsi', 'N/A')}")
            logger.info(f"  EMA20: {indicators.get('ema_20', 'N/A')}")
            logger.info(f"  EMA50: {indicators.get('ema_50', 'N/A')}")
            logger.info(f"  MACD: {indicators.get('macd', 'N/A')}")
            
            # Check for contradictions
            direction = signal['side']
            rsi = indicators.get('rsi')
            macd = indicators.get('macd')
            
            contradictions = []
            
            if direction == 'long':
                if rsi and rsi > 70:
                    contradictions.append("RSI overbought for long signal")
                if macd and macd < 0:
                    contradictions.append("MACD bearish for long signal")
            else:  # short
                if rsi and rsi < 30:
                    contradictions.append("RSI oversold for short signal")
                if macd and macd > 0:
                    contradictions.append("MACD bullish for short signal")
            
            if contradictions:
                logger.warning(f"⚠️ CONTRADICTIONS DETECTED: {', '.join(contradictions)}")
            else:
                logger.info("✅ No major contradictions detected")
            
            # Analyze score reasons
            score_reasons = signal.get('score_reasons', [])
            logger.info(f"Score Reasons ({len(score_reasons)}):")
            for reason in score_reasons:
                logger.info(f"  - {reason}")
        
        logger.info("\n=== Signal Analysis Complete ===")
        
    except Exception as e:
        logger.error(f"Error in signal analysis: {e}")


if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--analyze', action='store_true', 
                       help='Analyze existing signals for issues')
    parser.add_argument('--run', action='store_true', 
                       help='Run enhanced signal generation with alerts')
    
    args = parser.parse_args()
    
    if args.analyze:
        run_signal_analysis()
    elif args.run:
        main()
    else:
        print("Usage: python run_enhanced_signals_with_alerts.py --run")
        print("       python run_enhanced_signals_with_alerts.py --analyze") 