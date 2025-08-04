#!/usr/bin/env python3
"""
Enhanced Trading System Demo
Demonstrates the improved signal generation with all filters and dynamic risk management.
"""

import json
import sys
import os
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.strategies.enhanced_strategy_with_filters import EnhancedStrategyWithFilters
from src.utils.logger import setup_logger


def main():
    """
    Demo the enhanced trading system with all improvements.
    """
    logger = setup_logger()
    
    print("🚀 Enhanced Trading System Demo")
    print("="*80)
    
    # Initialize enhanced strategy
    enhanced_strategy = EnhancedStrategyWithFilters(balance=10000)
    
    # Test symbols (high volume symbols)
    test_symbols = [
        "B-BTC_USDT", "B-ETH_USDT", "B-SOL_USDT", "B-MATIC_USDT",
        "B-ADA_USDT", "B-DOT_USDT", "B-LINK_USDT", "B-AVAX_USDT",
        "B-TRX_USDT", "B-XRP_USDT"
    ]
    
    # Test each strategy type
    strategies = ["scalping", "swing", "long_swing"]
    
    all_results = {}
    
    for strategy in strategies:
        print(f"\n📋 Testing {strategy.upper()} Strategy")
        print("-" * 60)
        
        # Generate enhanced signals
        signals = enhanced_strategy.generate_multiple_enhanced_signals(
            symbols=test_symbols,
            strategy=strategy,
            max_signals=5
        )
        
        all_results[strategy] = signals
        
        if signals:
            print(f"✅ Generated {len(signals)} enhanced {strategy} signals:")
            
            for i, signal in enumerate(signals, 1):
                symbol = signal.get("symbol", "Unknown")
                side = signal.get("side", "Unknown")
                quality = signal.get("final_quality_score", 0)
                entry = signal.get("entry_price", 0)
                sl = signal.get("stop_loss", 0)
                tp = signal.get("take_profit", 0)
                hold_time = signal.get("max_hold_time", 0)
                risk_pct = signal.get("risk_management", {}).get("risk_percentage", 0)
                risk_reward = signal.get("risk_management", {}).get("risk_reward_tp1", 0)
                
                print(f"\n   🎯 Signal {i}: {symbol} ({side.upper()})")
                print(f"      Quality Score: {quality:.1f}/100")
                print(f"      Entry: {entry:.6f}")
                print(f"      Stop Loss: {sl:.6f} (Risk: {risk_pct:.2f}%)")
                print(f"      Take Profit: {tp:.6f} (R:R {risk_reward:.2f}:1)")
                print(f"      Hold Time: {hold_time:.1f} hours")
                
                # Show filter analysis
                volume_analysis = signal.get("volume_analysis", {})
                if volume_analysis:
                    print(f"      Volume Score: {volume_analysis.get('combined_score', 0):.1f}/100")
                    print(f"      Liquidity: {volume_analysis.get('liquidity_score', 0)}/100")
                    print(f"      Volatility: {volume_analysis.get('volatility_score', 0)}/100")
        else:
            print(f"❌ No {strategy} signals generated (all filtered out)")
    
    # Summary analysis
    print("\n" + "="*80)
    print("📊 SUMMARY ANALYSIS")
    print("="*80)
    
    total_signals = sum(len(signals) for signals in all_results.values())
    
    if total_signals > 0:
        # Calculate average quality by strategy
        for strategy, signals in all_results.items():
            if signals:
                avg_quality = sum(s.get("final_quality_score", 0) for s in signals) / len(signals)
                avg_risk = sum(s.get("risk_management", {}).get("risk_percentage", 0) for s in signals) / len(signals)
                avg_rr = sum(s.get("risk_management", {}).get("risk_reward_tp1", 0) for s in signals) / len(signals)
                avg_hold = sum(s.get("max_hold_time", 0) for s in signals) / len(signals)
                
                print(f"\n📈 {strategy.upper()} Strategy Performance:")
                print(f"   Signals Generated: {len(signals)}")
                print(f"   Average Quality: {avg_quality:.1f}/100")
                print(f"   Average Risk: {avg_risk:.2f}%")
                print(f"   Average R:R: {avg_rr:.2f}:1")
                print(f"   Average Hold Time: {avg_hold:.1f}h")
        
        # Find best signal overall
        all_signals_flat = []
        for signals in all_results.values():
            all_signals_flat.extend(signals)
        
        if all_signals_flat:
            best_signal = max(all_signals_flat, key=lambda x: x.get("final_quality_score", 0))
            
            print(f"\n🏆 BEST SIGNAL OVERALL:")
            print(f"   Symbol: {best_signal.get('symbol', 'Unknown')}")
            print(f"   Strategy: {best_signal.get('strategy', 'Unknown')}")
            print(f"   Side: {best_signal.get('side', 'Unknown').upper()}")
            print(f"   Quality: {best_signal.get('final_quality_score', 0):.1f}/100")
            print(f"   Risk: {best_signal.get('risk_management', {}).get('risk_percentage', 0):.2f}%")
            print(f"   R:R: {best_signal.get('risk_management', {}).get('risk_reward_tp1', 0):.2f}:1")
    else:
        print("❌ No signals generated across all strategies")
        print("💡 Recommendations:")
        print("   - Check if market data is available")
        print("   - Consider relaxing filter criteria")
        print("   - Verify volume thresholds are appropriate")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"enhanced_system_demo_results_{timestamp}.json"
    
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "test_symbols": test_symbols,
            "strategies_tested": strategies,
            "results": all_results,
            "summary": {
                "total_signals": total_signals,
                "signals_by_strategy": {k: len(v) for k, v in all_results.items()}
            }
        }, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    print("\n" + "="*80)
    print("🎯 Enhanced System Demo Complete!")
    print("="*80)
    
    # Key improvements summary
    print("\n✨ KEY IMPROVEMENTS IMPLEMENTED:")
    print("   1. ✅ Volume-Liquidity Filter (market depth analysis)")
    print("   2. ✅ Volume-Volatility Filter (price efficiency analysis)")
    print("   3. ✅ Dynamic Hold Time Calculation (market-condition based)")
    print("   4. ✅ ATR-based Stop Loss (instead of fixed percentages)")
    print("   5. ✅ Dynamic Take Profit with S/R levels")
    print("   6. ✅ Comprehensive Quality Scoring (0-100 scale)")
    print("   7. ✅ Risk Management Validation (max risk limits)")
    print("   8. ✅ Multi-layered Signal Filtering")
    
    print("\n🔧 BENEFITS:")
    print("   • Reduced false signals through better filtering")
    print("   • Improved risk management with dynamic stops")
    print("   • Optimized hold times prevent stale signals")
    print("   • Better liquidity ensures executable trades")
    print("   • Quality scoring enables signal ranking")


if __name__ == "__main__":
    main()