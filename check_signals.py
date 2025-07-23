#!/usr/bin/env python3
"""
Enhanced Signal Checker
Demonstrates the enhanced signal generation capabilities with high-profit signal monitoring
"""

import yaml
import json
from src.trading.enhanced_signal_generator import EnhancedSignalGenerator
from src.trading.signal_monitor import SignalMonitor
from src.utils.logger import setup_logger

def main():
    logger = setup_logger()
    logger.info("=== ENHANCED SIGNAL GENERATION DEMO ===")
    
    try:
        # Load configuration
        with open("config/config.yaml", "r") as file:
            config = yaml.safe_load(file)
        
        # Initialize enhanced signal generator
        signal_generator = EnhancedSignalGenerator(config)
        logger.info("Enhanced signal generator initialized")
        
        # Generate entry signals
        logger.info("Generating enhanced signals...")
        signals = signal_generator.generate_entry_signals()
        
        if not signals:
            logger.warning("No signals generated")
            return
        
        logger.info(f"Generated {len(signals)} enhanced signals")
        
        # Initialize signal monitor
        signal_monitor = SignalMonitor(config)
        
        # Process signals and add to monitoring
        high_profit_count = 0
        regular_count = 0
        
        for signal in signals:
            signal_dict = signal.to_dict()
            signal_dict["side"] = signal.direction.value
            
            # Convert enum objects in indicators
            if "indicators" in signal_dict:
                for key, value in signal_dict["indicators"].items():
                    if hasattr(value, 'value') and hasattr(value, '__class__'):
                        signal_dict["indicators"][key] = value.value
            
            estimated_profit = signal.estimated_profit_inr
            
            # Add to appropriate monitoring category with stricter criteria
            if estimated_profit >= 300:  # High-profit threshold (increased)
                if signal_monitor.add_high_profit_signal(signal_dict):
                    high_profit_count += 1
                    logger.info(f"🔥 HIGH-PROFIT: {signal.symbol} - {signal.strategy} ({signal.timeframe}) - Profit: ₹{estimated_profit:.2f}")
                else:
                    logger.warning(f"❌ Failed to add high-profit signal: {signal.symbol}")
            elif estimated_profit >= 200:  # Qualified signal threshold
                if signal_monitor.add_signal(signal_dict):
                    regular_count += 1
                    logger.info(f"📊 QUALIFIED: {signal.symbol} - {signal.strategy} ({signal.timeframe}) - Profit: ₹{estimated_profit:.2f}")
                else:
                    logger.warning(f"❌ Failed to add qualified signal: {signal.symbol}")
            else:
                logger.info(f"❌ REJECTED: {signal.symbol} - {signal.strategy} ({signal.timeframe}) - Profit: ₹{estimated_profit:.2f} (Too low)")
        
        # Get monitoring summary
        priority_summary = signal_monitor.get_signal_priority_summary()
        
        logger.info("=== SIGNAL MONITORING SUMMARY ===")
        logger.info(f"Total signals monitored: {priority_summary['total_signals']}")
        logger.info(f"High-profit signals: {priority_summary['high_profit_signals']}")
        logger.info(f"Regular signals: {regular_count}")
        logger.info(f"Total estimated profit: ₹{priority_summary['total_estimated_profit']:.2f}")
        logger.info(f"Average profit per signal: ₹{priority_summary['average_profit_per_signal']:.2f}")
        
        # Save signals to file
        signal_data = []
        for signal in signals:
            signal_dict = signal.to_dict()
            signal_dict["side"] = signal.direction.value
            signal_data.append(signal_dict)
        
        with open("latest_signals.json", "w") as f:
            json.dump(signal_data, f, indent=2, default=str)
        
        logger.info("Signals saved to latest_signals.json")
        
        # Save monitoring data
        signal_monitor.save_monitoring_data("latest_monitoring_data.json")
        logger.info("Monitoring data saved to latest_monitoring_data.json")
        
        # Display high-profit signals
        high_profit_signals = signal_monitor.get_high_profit_signals()
        if high_profit_signals:
            logger.info("=== HIGH-PROFIT SIGNALS ===")
            for i, signal_info in enumerate(high_profit_signals, 1):
                signal = signal_info["signal"]
                logger.info(f"{i}. {signal['symbol']} - {signal['strategy']} ({signal['timeframe']})")
                logger.info(f"   Estimated Profit: ₹{signal.get('estimated_profit_inr', 0):.2f}")
                logger.info(f"   Score: {signal.get('score', 0)}")
                logger.info(f"   Direction: {signal.get('side', 'unknown').upper()}")
        
        logger.info("=== ENHANCED SIGNAL GENERATION COMPLETE ===")
        
    except Exception as e:
        logger.error(f"Error in enhanced signal generation: {e}")

if __name__ == "__main__":
    main() 