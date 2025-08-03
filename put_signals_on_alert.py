#!/usr/bin/env python3
"""
Put Signals on Alert with Enhanced Validation
This script puts the highest scoring signals on alert with proper confirmation
"""

import json
import yaml
import time
from datetime import datetime
from src.strategies.simple_enhanced_generator import SimpleEnhancedGenerator
from src.trading.signal_alert_system import SignalAlertSystem
from src.utils.logger import setup_logger


def main():
    """Main function to put signals on alert with enhanced validation."""
    logger = setup_logger()
    
    try:
        logger.info("=== Putting Signals on Alert with Enhanced Validation ===")
        
        # Load configuration
        with open("config/config.yaml", "r") as file:
            config = yaml.safe_load(file)
        
        logger.info("Configuration loaded successfully")
        
        # Initialize components
        signal_generator = SimpleEnhancedGenerator()
        alert_system = SignalAlertSystem(config)
        
        logger.info("Components initialized")
        
        # Generate signals with confirmation
        logger.info("Generating enhanced signals with confirmation...")
        signals = signal_generator.generate_signals()
        
        if not signals:
            logger.warning("No signals generated - check market conditions")
            return
        
        logger.info(f"Generated {len(signals)} signals")
        
        # Filter and validate signals before putting on alert
        valid_signals = []
        
        for signal in signals:
            # Enhanced validation before alert
            if _validate_signal_for_alert(signal):
                valid_signals.append(signal)
                logger.info(f"✅ VALID SIGNAL: {signal['symbol']} - "
                          f"{signal['strategy']} ({signal['timeframe']}) - "
                          f"Score: {signal['score']}, Confidence: {signal['confidence']:.2f}")
            else:
                logger.warning(f"❌ INVALID SIGNAL: {signal['symbol']} - "
                             f"Score: {signal['score']}, Confidence: {signal['confidence']:.2f}")
        
        if not valid_signals:
            logger.warning("No valid signals to put on alert")
            return
        
        # Sort by quality score and take top signals
        valid_signals.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
        top_signals = valid_signals[:3]  # Top 3 signals
        
        logger.info(f"Putting {len(top_signals)} top signals on alert")
        
        # Put signals on alert
        alert_count = 0
        for signal in top_signals:
            signal_id = f"{signal['symbol']}_{signal['side']}_{signal['timestamp']}"
            
            # Add to alert system
            alert_system.active_alerts[signal_id] = {
                "signal": signal,
                "alert_type": "signal_alert",
                "timestamp": datetime.now().isoformat(),
                "status": "active"
            }
            
            alert_count += 1
            logger.info(f"🚨 ALERT SET: {signal['symbol']} - "
                      f"{signal['side'].upper()} - Score: {signal['score']}")
        
        # Save alert data
        alert_data = {
            "active_alerts": alert_system.active_alerts,
            "alert_count": alert_count,
            "last_update": datetime.now().isoformat(),
            "total_signals_generated": len(signals),
            "valid_signals": len(valid_signals),
            "signals_on_alert": len(top_signals)
        }
        
        with open("latest_alerts.json", "w") as f:
            json.dump(alert_data, f, indent=2, default=str)
        
        logger.info(f"✅ Successfully put {alert_count} signals on alert")
        logger.info("Alert data saved to latest_alerts.json")
        
        # Display summary
        print("\n" + "="*80)
        print("🚨 SIGNALS ON ALERT SUMMARY")
        print("="*80)
        for signal in top_signals:
            print(f"📊 {signal['symbol']} - {signal['side'].upper()}")
            print(f"   Score: {signal['score']}, Confidence: {signal['confidence']:.2f}")
            print(f"   Entry: {signal['entry_price']:.6f}")
            print(f"   SL: {signal['stop_loss']:.6f}, TP: {signal['take_profit']:.6f}")
            print(f"   Estimated Profit: ₹{signal.get('estimated_profit_inr', 0):.2f}")
            print()
        
        print("="*80)
        print(f"Total signals generated: {len(signals)}")
        print(f"Valid signals: {len(valid_signals)}")
        print(f"Signals on alert: {len(top_signals)}")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Error putting signals on alert: {e}")
        print(f"❌ Error: {e}")


def _validate_signal_for_alert(signal):
    """Enhanced validation for signals before putting on alert."""
    try:
        # Basic checks
        if not all(key in signal for key in ['symbol', 'side', 'entry_price', 'score', 'confidence']):
            return False
        
        # Score threshold (reduced for better signal generation)
        if signal['score'] < 6:
            return False
        
        # Confidence threshold (reduced for better signal generation)
        if signal['confidence'] < 0.4:
            return False
        
        # Check for reasonable entry price
        if signal['entry_price'] <= 0:
            return False
        
        # Check for reasonable stop loss and take profit
        if signal['side'] == 'long':
            if signal['stop_loss'] >= signal['entry_price']:
                return False
            if signal['take_profit'] <= signal['entry_price']:
                return False
        else:  # short
            if signal['stop_loss'] <= signal['entry_price']:
                return False
            if signal['take_profit'] >= signal['entry_price']:
                return False
        
        # Check for reasonable risk/reward ratio
        if signal['side'] == 'long':
            risk = signal['entry_price'] - signal['stop_loss']
            reward = signal['take_profit'] - signal['entry_price']
        else:
            risk = signal['stop_loss'] - signal['entry_price']
            reward = signal['entry_price'] - signal['take_profit']
        
        if risk <= 0 or reward <= 0:
            return False
        
        rr_ratio = reward / risk
        if rr_ratio < 1.5:  # Minimum 1:1.5 risk/reward
            return False
        
        return True
        
    except Exception as e:
        print(f"Error validating signal: {e}")
        return False


if __name__ == "__main__":
    main() 