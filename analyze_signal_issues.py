#!/usr/bin/env python3
"""
Analyze Signal Issues
This script analyzes the signal validation report to understand why signals failed
"""

import json
from datetime import datetime


def analyze_signal_validation_report():
    """Analyze the signal validation report to understand issues."""
    
    try:
        # Load the validation report
        with open("signal_validation_report_20250802_202442.json", "r") as f:
            report = json.load(f)
        
        print("="*80)
        print("🔍 SIGNAL VALIDATION ANALYSIS")
        print("="*80)
        
        summary = report.get("summary", {})
        print(f"Total signals: {summary.get('total_signals', 0)}")
        print(f"Passed signals: {summary.get('passed_signals', 0)}")
        print(f"Failed signals: {summary.get('failed_signals', 0)}")
        print(f"Success rate: {summary.get('success_rate', 0):.1%}")
        
        # Analyze issue types
        issue_types = summary.get("issue_types", {})
        print("\n📊 ISSUE BREAKDOWN:")
        for issue, count in issue_types.items():
            print(f"   {issue}: {count}")
        
        # Analyze individual signals
        signals = report.get("signals", {})
        
        print("\n" + "="*80)
        print("📈 HIGHEST SCORING SIGNALS ANALYSIS")
        print("="*80)
        
        # Sort signals by score
        sorted_signals = sorted(
            signals.items(), 
            key=lambda x: x[1].get('score', 0), 
            reverse=True
        )
        
        for signal_id, signal_data in sorted_signals[:5]:  # Top 5 signals
            symbol = signal_data.get('symbol', 'Unknown')
            side = signal_data.get('side', 'Unknown')
            score = signal_data.get('score', 0)
            confidence = signal_data.get('confidence', 0)
            entry_price = signal_data.get('entry_price', 0)
            current_price = signal_data.get('current_price', 0)
            price_change = signal_data.get('price_change_percent', 0)
            validation_passed = signal_data.get('validation_passed', False)
            
            print(f"\n🎯 {symbol} - {side.upper()}")
            print(f"   Score: {score}, Confidence: {confidence:.2f}")
            print(f"   Entry: {entry_price:.6f}, Current: {current_price:.6f}")
            print(f"   Price Change: {price_change:.2f}%")
            print(f"   Validation: {'✅ PASSED' if validation_passed else '❌ FAILED'}")
            
            # Analyze issues
            issues = signal_data.get('issues', [])
            if issues:
                print("   Issues:")
                for issue in issues:
                    print(f"     ❌ {issue}")
            
            # Analyze strengths
            strengths = signal_data.get('strengths', [])
            if strengths:
                print("   Strengths:")
                for strength in strengths:
                    print(f"     ✅ {strength}")
            
            # Analyze technical analysis
            tech_analysis = signal_data.get('technical_analysis', {})
            if tech_analysis:
                print("   Technical Analysis:")
                for indicator, analysis in tech_analysis.items():
                    if isinstance(analysis, dict):
                        status = analysis.get('status', 'unknown')
                        message = analysis.get('message', '')
                        status_icon = "✅" if status == "good" else "⚠️" if status == "warning" else "❌"
                        print(f"     {status_icon} {indicator}: {message}")
            
            # Analyze direction validation
            direction_validation = signal_data.get('direction_validation', {})
            if direction_validation:
                direction_correct = direction_validation.get('direction_correct', False)
                reasons = direction_validation.get('reasons', [])
                warnings = direction_validation.get('warnings', [])
                
                print(f"   Direction Validation: {'✅ CORRECT' if direction_correct else '❌ INCORRECT'}")
                if reasons:
                    print("     Reasons:")
                    for reason in reasons:
                        print(f"       ✅ {reason}")
                if warnings:
                    print("     Warnings:")
                    for warning in warnings:
                        print(f"       ⚠️ {warning}")
        
        # Analyze the highest scoring signal that hit SL
        print("\n" + "="*80)
        print("🚨 HIGHEST SCORING SIGNAL ANALYSIS (B-HBAR_USDT)")
        print("="*80)
        
        hbar_signal = None
        for signal_id, signal_data in signals.items():
            if "B-HBAR_USDT" in signal_id and signal_data.get('score', 0) == 10:
                hbar_signal = signal_data
                break
        
        if hbar_signal:
            print(f"Symbol: {hbar_signal.get('symbol')}")
            print(f"Side: {hbar_signal.get('side')}")
            print(f"Score: {hbar_signal.get('score')}")
            print(f"Confidence: {hbar_signal.get('confidence', 0):.2f}")
            print(f"Entry Price: {hbar_signal.get('entry_price', 0):.6f}")
            print(f"Current Price: {hbar_signal.get('current_price', 0):.6f}")
            print(f"Price Change: {hbar_signal.get('price_change_percent', 0):.2f}%")
            
            # Analyze indicators
            indicators = hbar_signal.get('indicators', {})
            if indicators:
                print("\n📊 INDICATOR ANALYSIS:")
                print(f"   RSI: {indicators.get('rsi', 'N/A'):.2f}")
                print(f"   EMA20: {indicators.get('ema_20', 'N/A'):.6f}")
                print(f"   EMA50: {indicators.get('ema_50', 'N/A'):.6f}")
                print(f"   MACD: {indicators.get('macd', 'N/A'):.6f}")
                print(f"   CCI: {indicators.get('cci', 'N/A'):.2f}")
                print(f"   Momentum: {indicators.get('momentum_osc', 'N/A'):.6f}")
            
            # Analyze score reasons
            score_reasons = hbar_signal.get('score_reasons', [])
            if score_reasons:
                print("\n📋 SCORE REASONS:")
                for reason in score_reasons:
                    print(f"   ✅ {reason}")
            
            # Analyze issues
            issues = hbar_signal.get('issues', [])
            if issues:
                print("\n❌ VALIDATION ISSUES:")
                for issue in issues:
                    print(f"   ❌ {issue}")
        
        print("\n" + "="*80)
        print("💡 RECOMMENDATIONS")
        print("="*80)
        print("1. Relax validation thresholds for price deviation (5% instead of 2%)")
        print("2. Reduce volume requirements for smaller coins")
        print("3. Lower confidence threshold from 50% to 40%")
        print("4. Improve signal confirmation logic")
        print("5. Add more technical indicators for confirmation")
        print("6. Implement better risk management with ATR-based stops")
        print("="*80)
        
    except Exception as e:
        print(f"❌ Error analyzing signal validation report: {e}")


if __name__ == "__main__":
    analyze_signal_validation_report() 