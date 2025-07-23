#!/usr/bin/env python3
"""
Signal Monitoring System - Start monitoring active signals
This script starts the monitoring system to track active signals and show dynamic adjustments.
"""

import time
import json
from datetime import datetime
from src.trading.signal_monitor import SignalMonitor
from src.utils.logger import setup_logger
import yaml

def start_monitoring():
    """Start monitoring active signals with real-time updates."""
    
    # Setup logging
    logger = setup_logger()
    logger.info("=== Starting Signal Monitoring System ===")
    
    # Load configuration
    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    # Initialize signal monitor
    signal_monitor = SignalMonitor(config)
    
    # Load existing monitoring data if available
    signal_monitor.load_monitoring_data("latest_monitoring_data.json")
    
    active_signals_count = len(signal_monitor.active_signals)
    logger.info(f"Found {active_signals_count} active signals to monitor")
    
    if active_signals_count == 0:
        logger.warning("No active signals found. Please run the main bot first to generate signals.")
        return
    
    # Display initial signal status
    logger.info("=== INITIAL SIGNAL STATUS ===")
    for signal_id, signal_data in signal_monitor.active_signals.items():
        signal = signal_data["signal"]
        logger.info(f"Signal: {signal['symbol']} {signal['strategy']} {signal['side'].upper()}")
        logger.info(f"  Entry Price: {signal['entry_price']:.6f}")
        logger.info(f"  Original Stop Loss: {signal['stop_loss']:.6f}")
        logger.info(f"  Original Target: {signal_data['current_exit_price']:.6f}")
        logger.info(f"  Max Hold Time: {signal['max_hold_time']} minutes")
        logger.info(f"  Entry Time: {signal_data['entry_time']}")
        logger.info("---")
    
    logger.info("=== STARTING REAL-TIME MONITORING ===")
    logger.info("Monitoring will update every 10 seconds...")
    logger.info("Press Ctrl+C to stop monitoring")
    
    try:
        cycle_count = 0
        while True:
            cycle_count += 1
            logger.info(f"\n=== MONITORING CYCLE {cycle_count} - {datetime.now().strftime('%H:%M:%S')} ===")
            
            # Monitor all signals
            summary = signal_monitor.monitor_all_signals()
            
            if summary["active_signals"] == 0:
                logger.info("No active signals remaining. Monitoring complete.")
                break
            
            # Display current status
            logger.info(f"Active Signals: {summary['active_signals']}")
            logger.info(f"Total P&L: {summary['total_profit_loss']:.2f}%")
            
            # Display detailed signal updates
            for signal_info in summary["signals"]:
                symbol = signal_info["symbol"]
                strategy = signal_info["strategy"]
                side = signal_info["side"]
                entry_price = signal_info["entry_price"]
                current_price = signal_info["current_price"]
                profit_pct = signal_info["profit_loss_percent"]
                current_stop = signal_info["current_stop_loss"]
                current_target = signal_info["current_exit_price"]
                
                # Calculate position size information
                position_size_ratio = config["trading"]["position_size_ratio"]
                initial_balance = config["trading"]["initial_balance"]
                
                # DYNAMIC: Calculate position size based on signal characteristics
                score = signal_info.get("score", 6)
                strategy = signal_info.get("strategy", "scalping")
                estimated_profit = signal_info.get("estimated_profit_inr", 100)
                
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
                
                # Calculate USDT position size
                usdt_inr_rate = 93.0  # Default rate, you can fetch this dynamically
                position_size_usdt = position_size_inr / usdt_inr_rate
                
                # Calculate quantity and notional value
                leverage = 25  # 25x leverage
                quantity = (position_size_usdt / entry_price) * leverage if entry_price > 0 else 0
                notional_value = quantity * entry_price
                
                # Calculate current P&L in INR
                current_pnl_inr = (profit_pct / 100) * notional_value * usdt_inr_rate
                
                logger.info(f"\n📊 {symbol} ({strategy} {side.upper()})")
                logger.info(f"   💰 Position Size: ₹{position_size_inr:.2f} (${position_size_usdt:.2f})")
                logger.info(f"   📈 Quantity: {quantity:.6f} {symbol.split('_')[0]}")
                logger.info(f"   💵 Notional Value: ${notional_value:.2f}")
                logger.info(f"   ⚡ Leverage: {leverage}x")
                logger.info(f"   🎯 Entry: {entry_price:.6f} | Current: {current_price:.6f}")
                logger.info(f"   📊 P&L: {profit_pct:.2f}% (₹{current_pnl_inr:.2f})")
                logger.info(f"   🛑 Stop Loss: {current_stop:.6f}")
                logger.info(f"   🎯 Target: {current_target:.6f}")
                logger.info(f"   📊 Multipliers: Score={score_multiplier:.1f}x, Strategy={strategy_multiplier:.1f}x, Profit={profit_multiplier:.1f}x")
                
                # Show exit suggestions
                suggestions = signal_info.get("exit_suggestions", [])
                if suggestions:
                    logger.info(f"   💡 Exit Suggestions:")
                    for suggestion in suggestions:
                        priority = suggestion.get("priority", "low")
                        reason = suggestion.get("reason", "")
                        action = suggestion.get("suggested_action", "")
                        logger.info(f"      [{priority.upper()}] {reason} -> {action}")
            
            # Show alerts
            if summary["alerts"]:
                logger.info(f"\n🚨 ALERTS:")
                for alert in summary["alerts"]:
                    logger.info(f"   {alert['symbol']}: {alert['message']} -> {alert['action']}")
            
            # Save monitoring data
            signal_monitor.save_monitoring_data("latest_monitoring_data.json")
            
            # Wait for next cycle
            logger.info(f"\nWaiting 10 seconds for next update...")
            time.sleep(10)
            
    except KeyboardInterrupt:
        logger.info("\n=== MONITORING STOPPED BY USER ===")
    
    # Final summary
    final_summary = signal_monitor.get_signal_summary()
    logger.info(f"\n=== FINAL SUMMARY ===")
    logger.info(f"Active Signals: {final_summary['active_signals']}")
    logger.info(f"Historical Signals: {final_summary['historical_signals']}")
    logger.info(f"Total Trades: {final_summary['total_trades']}")
    logger.info(f"Profitable Trades: {final_summary['profitable_trades']}")
    logger.info(f"Win Rate: {final_summary['win_rate']:.1f}%")
    logger.info(f"Total Profit: {final_summary['total_profit_percent']:.2f}%")
    logger.info(f"Average Profit: {final_summary['average_profit_percent']:.2f}%")
    
    logger.info("=== Monitoring Complete ===")

if __name__ == "__main__":
    start_monitoring() 