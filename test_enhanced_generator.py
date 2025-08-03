#!/usr/bin/env python3
"""
Test script for enhanced signal generator
"""

import yaml
import json
from src.strategies.simple_enhanced_generator import SimpleEnhancedGenerator
from src.utils.logger import setup_logger


def test_enhanced_generator():
    """Test the enhanced signal generator."""
    logger = setup_logger()
    
    try:
        logger.info("=== Testing Enhanced Signal Generator ===")
        
        # Load configuration
        with open("config/config.yaml", "r") as file:
            config = yaml.safe_load(file)
        
        # Initialize enhanced generator
        generator = SimpleEnhancedGenerator()
        
        logger.info("Testing signal generation with confirmation...")
        
        # Generate signals
        signals = generator.generate_signals()
        
        if not signals:
            logger.warning("No signals generated")
            return
        
        logger.info(f"Generated {len(signals)} confirmed signals")
        
        # Analyze each signal
        for i, signal in enumerate(signals):
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
        
        # Save enhanced signals
        with open("enhanced_signals_test.json", "w") as f:
            json.dump(signals, f, indent=2, default=str)
        
        logger.info(f"\nSaved {len(signals)} enhanced signals to enhanced_signals_test.json")
        
        # Summary
        logger.info("\n=== ENHANCED GENERATOR TEST SUMMARY ===")
        logger.info(f"Total signals generated: {len(signals)}")
        
        long_signals = [s for s in signals if s['side'] == 'long']
        short_signals = [s for s in signals if s['side'] == 'short']
        
        logger.info(f"Long signals: {len(long_signals)}")
        logger.info(f"Short signals: {len(short_signals)}")
        
        # Check for RSI contradictions
        contradictions_found = 0
        for signal in signals:
            direction = signal['side']
            rsi = signal.get('indicators', {}).get('rsi')
            
            if direction == 'short' and rsi and rsi < 30:
                contradictions_found += 1
            elif direction == 'long' and rsi and rsi > 70:
                contradictions_found += 1
        
        logger.info(f"Signals with RSI contradictions: {contradictions_found}")
        
        if contradictions_found == 0:
            logger.info("✅ ENHANCED GENERATOR: No RSI contradictions detected!")
        else:
            logger.warning(f"⚠️ ENHANCED GENERATOR: {contradictions_found} signals still have RSI contradictions")
        
        logger.info("=== Enhanced Generator Test Complete ===")
        
    except Exception as e:
        logger.error(f"Error in enhanced generator test: {e}")
        raise


if __name__ == "__main__":
    test_enhanced_generator() 