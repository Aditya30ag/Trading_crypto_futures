#!/usr/bin/env python3
"""
Enhanced Signal Generator Test Script
Produces 2-3 highly confirmed signals with improved take profit consistency
"""

import json
import sys
from datetime import datetime
from src.strategies.enhanced_signal_generator import EnhancedSignalGenerator
from src.utils.logger import setup_logger


def main():
    """Main function to test enhanced signal generation."""
    logger = setup_logger()
    logger.info("Starting Enhanced Signal Generator Test")
    
    try:
        # Initialize enhanced signal generator
        generator = EnhancedSignalGenerator(balance=10000)
        
        # Generate enhanced signals
        logger.info("Generating enhanced signals...")
        signals = generator.generate_enhanced_signals()
        
        if not signals:
            logger.warning("No enhanced signals generated. Market conditions may not be suitable.")
            return
        
        # Display results
        logger.info(f"Generated {len(signals)} enhanced signals:")
        print("\n" + "="*80)
        print("ENHANCED SIGNAL GENERATOR RESULTS")
        print("="*80)
        
        for i, signal in enumerate(signals, 1):
            print(f"\n🔔 SIGNAL #{i}")
            print("-" * 50)
            print(f"Symbol: {signal['symbol']}")
            print(f"Strategy: {signal['strategy']}")
            print(f"Direction: {signal['direction']}")
            print(f"Timeframe: {signal['timeframe']}")
            print(f"Entry Price: ${signal['entry_price']:.6f}")
            print(f"Stop Loss: ${signal['stop_loss']:.6f}")
            print(f"Take Profit: ${signal['take_profit']:.6f}")
            
            # Calculate risk/reward ratio
            if signal['side'] == 'long':
                risk = signal['entry_price'] - signal['stop_loss']
                reward = signal['take_profit'] - signal['entry_price']
            else:
                risk = signal['stop_loss'] - signal['entry_price']
                reward = signal['entry_price'] - signal['take_profit']
            
            rr_ratio = reward / risk if risk > 0 else 0
            print(f"Risk/Reward Ratio: 1:{rr_ratio:.2f}")
            
            print(f"Quality Score: {signal['score']}/10")
            print(f"Estimated Profit: ₹{signal['estimated_profit_inr']:.2f}")
            print(f"Confidence: {signal['confidence']:.2%}")
            print(f"Max Hold Time: {signal['max_hold_time']} hours")
            
            print("\n📊 Score Reasons:")
            for reason in signal['score_reasons']:
                print(f"  ✓ {reason}")
            
            print("\n📈 Key Indicators:")
            indicators = signal['indicators']
            print(f"  RSI: {indicators['rsi']:.2f}")
            print(f"  EMA20: ${indicators['ema_20']:.6f}")
            print(f"  EMA50: ${indicators['ema_50']:.6f}")
            print(f"  MACD: {indicators['macd']:.6f}")
            print(f"  ATR: {indicators['atr']:.6f}")
            
            print("\n" + "-" * 50)
        
        # Save signals to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"enhanced_signals_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(signals, f, indent=2, default=str)
        
        logger.info(f"Enhanced signals saved to {filename}")
        
        # Summary statistics
        total_profit = sum(s['estimated_profit_inr'] for s in signals)
        avg_score = sum(s['score'] for s in signals) / len(signals)
        avg_confidence = sum(s['confidence'] for s in signals) / len(signals)
        
        print(f"\n📊 SUMMARY STATISTICS")
        print("-" * 30)
        print(f"Total Signals: {len(signals)}")
        print(f"Average Quality Score: {avg_score:.1f}/10")
        print(f"Average Confidence: {avg_confidence:.1%}")
        print(f"Total Estimated Profit: ₹{total_profit:.2f}")
        print(f"Average Profit per Signal: ₹{total_profit/len(signals):.2f}")
        
        # Market analysis
        long_signals = [s for s in signals if s['side'] == 'long']
        short_signals = [s for s in signals if s['side'] == 'short']
        
        print(f"\n📈 SIGNAL DISTRIBUTION")
        print("-" * 30)
        print(f"Long Signals: {len(long_signals)}")
        print(f"Short Signals: {len(short_signals)}")
        
        if long_signals:
            long_avg_profit = sum(s['estimated_profit_inr'] for s in long_signals) / len(long_signals)
            print(f"Average Long Profit: ₹{long_avg_profit:.2f}")
        
        if short_signals:
            short_avg_profit = sum(s['estimated_profit_inr'] for s in short_signals) / len(short_signals)
            print(f"Average Short Profit: ₹{short_avg_profit:.2f}")
        
        print("\n" + "="*80)
        print("✅ Enhanced Signal Generation Complete!")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Error in enhanced signal generation: {e}")
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 