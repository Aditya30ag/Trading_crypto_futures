#!/usr/bin/env python3
"""
Comprehensive Filter Testing System
Tests each filter combination to find the best performing setup for signal generation.
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Optional
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.enhanced_filters import EnhancedVolumeFilters
from src.utils.dynamic_risk_manager import DynamicRiskManager
from src.data.fetcher import CoinDCXFetcher
from src.utils.logger import setup_logger
from src.strategies.scalping import ScalpingStrategy
from src.strategies.swing import SwingStrategy
from src.strategies.long_swing import LongSwingStrategy


class ComprehensiveFilterTester:
    """
    Test system for comparing different filter combinations and configurations.
    """
    
    def __init__(self):
        self.logger = setup_logger()
        self.fetcher = CoinDCXFetcher()
        self.volume_filters = EnhancedVolumeFilters()
        self.risk_manager = DynamicRiskManager()
        
        # Test configurations
        self.test_symbols = [
            "B-BTC_USDT", "B-ETH_USDT", "B-SOL_USDT", "B-MATIC_USDT",
            "B-ADA_USDT", "B-DOT_USDT", "B-LINK_USDT", "B-AVAX_USDT"
        ]
        
        self.strategies = ["scalping", "swing", "long_swing"]
        self.timeframes = {
            "scalping": ["5m"],
            "swing": ["15m", "1h"],
            "long_swing": ["30m", "1h"]
        }
        
        # Filter test configurations
        self.filter_configs = {
            "baseline": {
                "use_volume_filters": False,
                "use_dynamic_risk": False,
                "description": "Current system without enhancements"
            },
            "volume_only": {
                "use_volume_filters": True,
                "use_dynamic_risk": False,
                "description": "Volume-liquidity + Volume-volatility filters only"
            },
            "risk_only": {
                "use_volume_filters": False,
                "use_dynamic_risk": True,
                "description": "Dynamic risk management only"
            },
            "combined": {
                "use_volume_filters": True,
                "use_dynamic_risk": True,
                "description": "All enhancements combined"
            },
            "conservative": {
                "use_volume_filters": True,
                "use_dynamic_risk": True,
                "conservative_mode": True,
                "description": "Conservative settings with all enhancements"
            }
        }
        
        self.results = {}
    
    def test_single_symbol_strategy(self, symbol: str, strategy: str, timeframe: str, 
                                  filter_config: Dict) -> Optional[Dict]:
        """
        Test a single symbol-strategy combination with specific filter configuration.
        """
        try:
            self.logger.info(f"🧪 Testing {symbol} - {strategy} - {timeframe} with {filter_config['description']}")
            
            # Fetch market data
            market_data = self.fetcher.fetch_market_data(symbol)
            if not market_data:
                return None
            
            # Fetch candlestick data
            candles = self.fetcher.fetch_candlestick_data(symbol, timeframe, limit=100)
            if not candles or len(candles) < 50:
                return None
            
            current_price = float(market_data["last_price"])
            
            # Apply volume filters if enabled
            volume_filter_result = None
            if filter_config.get("use_volume_filters", False):
                volume_filter_result = self.volume_filters.combined_filter_analysis(
                    symbol, candles, strategy
                )
                if not volume_filter_result["passed"]:
                    return {
                        "symbol": symbol,
                        "strategy": strategy,
                        "timeframe": timeframe,
                        "status": "filtered_out",
                        "reason": "Failed volume filters",
                        "volume_score": volume_filter_result["combined_score"],
                        "filter_config": filter_config["description"]
                    }
            
            # Generate signal using appropriate strategy
            signal = None
            if strategy == "scalping":
                strategy_instance = ScalpingStrategy(10000)
                signal = strategy_instance.generate_signal(symbol, timeframe)
            elif strategy == "swing":
                strategy_instance = SwingStrategy(10000)
                signal = strategy_instance.generate_signal(symbol, timeframe)
            elif strategy == "long_swing":
                strategy_instance = LongSwingStrategy(10000)
                signal = strategy_instance.generate_signal(symbol, timeframe)
            
            if not signal:
                return {
                    "symbol": symbol,
                    "strategy": strategy,
                    "timeframe": timeframe,
                    "status": "no_signal",
                    "reason": "Strategy did not generate signal",
                    "filter_config": filter_config["description"]
                }
            
            # Apply dynamic risk management if enabled
            risk_analysis = None
            if filter_config.get("use_dynamic_risk", False):
                # Calculate optimal hold time
                hold_time_result = self.risk_manager.calculate_optimal_hold_time(
                    symbol, candles, strategy, timeframe
                )
                
                # Calculate dynamic stop loss
                stop_loss_result = self.risk_manager.calculate_dynamic_stop_loss(
                    symbol, candles, signal["entry_price"], signal["side"], strategy
                )
                
                # Calculate dynamic take profit
                take_profit_result = self.risk_manager.calculate_dynamic_take_profit(
                    symbol, candles, signal["entry_price"], 
                    stop_loss_result["stop_loss"], signal["side"], strategy
                )
                
                risk_analysis = {
                    "hold_time": hold_time_result,
                    "stop_loss": stop_loss_result,
                    "take_profit": take_profit_result
                }
                
                # Update signal with dynamic values
                signal["max_hold_time"] = hold_time_result["hold_time_hours"]
                signal["stop_loss"] = stop_loss_result["stop_loss"]
                signal["take_profit"] = take_profit_result["tp1"]
                signal["tp1"] = take_profit_result["tp1"]
                signal["tp2"] = take_profit_result["tp2"]
            
            # Calculate signal quality metrics
            entry_price = signal["entry_price"]
            stop_loss = signal["stop_loss"]
            take_profit = signal["take_profit"]
            
            # Risk-reward ratio
            risk_amount = abs(entry_price - stop_loss)
            reward_amount = abs(take_profit - entry_price)
            risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
            
            # Risk percentage
            risk_pct = (risk_amount / entry_price) * 100
            
            # Hold time assessment
            hold_time_hours = signal.get("max_hold_time", 24)
            hold_time_score = self._assess_hold_time(hold_time_hours, strategy)
            
            # Price deviation from current price
            price_deviation = abs(entry_price - current_price) / current_price * 100
            
            # Overall signal quality score
            quality_score = self._calculate_quality_score(
                signal, volume_filter_result, risk_analysis, 
                risk_reward_ratio, risk_pct, hold_time_score, price_deviation
            )
            
            result = {
                "symbol": symbol,
                "strategy": strategy,
                "timeframe": timeframe,
                "status": "signal_generated",
                "signal": signal,
                "quality_metrics": {
                    "risk_reward_ratio": round(risk_reward_ratio, 2),
                    "risk_percentage": round(risk_pct, 2),
                    "hold_time_hours": hold_time_hours,
                    "hold_time_score": hold_time_score,
                    "price_deviation_pct": round(price_deviation, 2),
                    "overall_quality_score": round(quality_score, 1)
                },
                "volume_analysis": volume_filter_result,
                "risk_analysis": risk_analysis,
                "filter_config": filter_config["description"]
            }
            
            self.logger.info(f"✅ Generated signal for {symbol}: Quality {quality_score:.1f}/100")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error testing {symbol} - {strategy}: {e}")
            return {
                "symbol": symbol,
                "strategy": strategy,
                "timeframe": timeframe,
                "status": "error",
                "reason": str(e),
                "filter_config": filter_config["description"]
            }
    
    def _assess_hold_time(self, hold_time_hours: float, strategy: str) -> float:
        """
        Assess hold time appropriateness (0-100 score).
        """
        optimal_ranges = {
            "scalping": (0.5, 2.0),      # 30 min to 2 hours
            "swing": (2.0, 8.0),         # 2 to 8 hours  
            "long_swing": (4.0, 16.0)    # 4 to 16 hours
        }
        
        min_optimal, max_optimal = optimal_ranges.get(strategy, (2.0, 8.0))
        
        if min_optimal <= hold_time_hours <= max_optimal:
            return 100.0  # Perfect
        elif hold_time_hours < min_optimal:
            # Too short
            ratio = hold_time_hours / min_optimal
            return max(50.0, ratio * 100)
        else:
            # Too long
            if hold_time_hours > max_optimal * 2:
                return 10.0  # Very poor
            else:
                excess_ratio = (hold_time_hours - max_optimal) / max_optimal
                return max(20.0, 100 - (excess_ratio * 80))
    
    def _calculate_quality_score(self, signal: Dict, volume_analysis: Optional[Dict],
                               risk_analysis: Optional[Dict], risk_reward_ratio: float,
                               risk_pct: float, hold_time_score: float, price_deviation: float) -> float:
        """
        Calculate overall signal quality score (0-100).
        """
        score = 0
        
        # Base signal score (30 points)
        signal_score = signal.get("score", 0)
        max_signal_score = signal.get("score_max", 10)
        normalized_signal_score = (signal_score / max_signal_score) * 30 if max_signal_score > 0 else 0
        score += normalized_signal_score
        
        # Volume analysis score (20 points)
        if volume_analysis and volume_analysis.get("passed"):
            volume_score = volume_analysis.get("combined_score", 0)
            score += (volume_score / 100) * 20
        else:
            score += 10  # Partial score if no volume analysis
        
        # Risk-reward ratio score (20 points)
        if risk_reward_ratio >= 2.5:
            score += 20
        elif risk_reward_ratio >= 2.0:
            score += 16
        elif risk_reward_ratio >= 1.5:
            score += 12
        elif risk_reward_ratio >= 1.0:
            score += 8
        else:
            score += 4  # Poor risk-reward
        
        # Risk percentage score (15 points)
        if risk_pct <= 2.0:
            score += 15
        elif risk_pct <= 3.0:
            score += 12
        elif risk_pct <= 4.0:
            score += 8
        else:
            score += 4  # High risk
        
        # Hold time score (10 points)
        score += (hold_time_score / 100) * 10
        
        # Price deviation penalty (5 points)
        if price_deviation <= 1.0:
            score += 5
        elif price_deviation <= 2.0:
            score += 3
        elif price_deviation <= 5.0:
            score += 1
        # No points for high deviation
        
        return min(score, 100.0)
    
    def run_comprehensive_test(self) -> Dict:
        """
        Run comprehensive test across all filter configurations.
        """
        self.logger.info("🚀 Starting Comprehensive Filter Testing")
        
        start_time = datetime.now()
        all_results = {}
        
        for config_name, config in self.filter_configs.items():
            self.logger.info(f"\n📋 Testing Configuration: {config_name}")
            self.logger.info(f"   Description: {config['description']}")
            
            config_results = []
            
            for symbol in self.test_symbols:
                for strategy in self.strategies:
                    for timeframe in self.timeframes[strategy]:
                        result = self.test_single_symbol_strategy(
                            symbol, strategy, timeframe, config
                        )
                        if result:
                            config_results.append(result)
                        
                        # Small delay to avoid rate limiting
                        time.sleep(0.1)
            
            all_results[config_name] = config_results
            
            # Calculate configuration summary
            self._summarize_config_results(config_name, config_results)
        
        # Generate final comparison
        comparison = self._generate_final_comparison(all_results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"comprehensive_filter_test_results_{timestamp}.json"
        
        with open(results_file, "w") as f:
            json.dump({
                "timestamp": start_time.isoformat(),
                "test_duration_minutes": (datetime.now() - start_time).total_seconds() / 60,
                "configurations": self.filter_configs,
                "results": all_results,
                "comparison": comparison
            }, f, indent=2)
        
        self.logger.info(f"📊 Results saved to: {results_file}")
        
        return comparison
    
    def _summarize_config_results(self, config_name: str, results: List[Dict]):
        """
        Summarize results for a specific configuration.
        """
        total_tests = len(results)
        signals_generated = len([r for r in results if r["status"] == "signal_generated"])
        filtered_out = len([r for r in results if r["status"] == "filtered_out"])
        errors = len([r for r in results if r["status"] == "error"])
        
        quality_scores = [
            r["quality_metrics"]["overall_quality_score"] 
            for r in results 
            if r["status"] == "signal_generated"
        ]
        
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            high_quality_signals = len([s for s in quality_scores if s >= 70])
        else:
            avg_quality = 0
            high_quality_signals = 0
        
        self.logger.info(f"\n📊 {config_name.upper()} SUMMARY:")
        self.logger.info(f"   Total tests: {total_tests}")
        self.logger.info(f"   Signals generated: {signals_generated}")
        self.logger.info(f"   Filtered out: {filtered_out}")
        self.logger.info(f"   Errors: {errors}")
        self.logger.info(f"   Average quality: {avg_quality:.1f}/100")
        self.logger.info(f"   High quality signals (70+): {high_quality_signals}")
        self.logger.info(f"   Success rate: {(signals_generated/total_tests)*100:.1f}%")
        
        return {
            "total_tests": total_tests,
            "signals_generated": signals_generated,
            "filtered_out": filtered_out,
            "errors": errors,
            "average_quality": avg_quality,
            "high_quality_signals": high_quality_signals,
            "success_rate": (signals_generated/total_tests)*100 if total_tests > 0 else 0
        }
    
    def _generate_final_comparison(self, all_results: Dict) -> Dict:
        """
        Generate final comparison and recommendations.
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("🏆 FINAL COMPARISON AND RECOMMENDATIONS")
        self.logger.info("="*80)
        
        comparison = {}
        
        for config_name, results in all_results.items():
            summary = self._summarize_config_results(config_name, results)
            comparison[config_name] = summary
        
        # Find best configuration
        best_config = max(
            comparison.items(),
            key=lambda x: x[1]["average_quality"] * x[1]["success_rate"] / 100
        )
        
        self.logger.info(f"\n🥇 BEST CONFIGURATION: {best_config[0].upper()}")
        self.logger.info(f"   Configuration: {self.filter_configs[best_config[0]]['description']}")
        self.logger.info(f"   Average Quality: {best_config[1]['average_quality']:.1f}/100")
        self.logger.info(f"   Success Rate: {best_config[1]['success_rate']:.1f}%")
        self.logger.info(f"   High Quality Signals: {best_config[1]['high_quality_signals']}")
        
        # Recommendations
        recommendations = self._generate_recommendations(comparison)
        
        self.logger.info("\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            self.logger.info(f"   {i}. {rec}")
        
        return {
            "summary": comparison,
            "best_configuration": best_config[0],
            "recommendations": recommendations
        }
    
    def _generate_recommendations(self, comparison: Dict) -> List[str]:
        """
        Generate recommendations based on test results.
        """
        recommendations = []
        
        # Compare baseline vs enhanced
        baseline = comparison.get("baseline", {})
        combined = comparison.get("combined", {})
        
        if combined.get("average_quality", 0) > baseline.get("average_quality", 0):
            quality_improvement = combined["average_quality"] - baseline["average_quality"]
            recommendations.append(
                f"Enhanced filters improve signal quality by {quality_improvement:.1f} points"
            )
        
        # Check if volume filters help
        volume_only = comparison.get("volume_only", {})
        if volume_only.get("high_quality_signals", 0) > baseline.get("high_quality_signals", 0):
            recommendations.append("Volume filters significantly improve signal quality")
        
        # Check if risk management helps
        risk_only = comparison.get("risk_only", {})
        if risk_only.get("average_quality", 0) > baseline.get("average_quality", 0):
            recommendations.append("Dynamic risk management improves overall performance")
        
        # Success rate analysis
        best_success_rate = max(comp.get("success_rate", 0) for comp in comparison.values())
        if best_success_rate < 50:
            recommendations.append("Consider loosening filter criteria to increase signal generation")
        elif best_success_rate > 80:
            recommendations.append("Current filters maintain good signal generation rate")
        
        # Quality threshold analysis
        high_quality_counts = [comp.get("high_quality_signals", 0) for comp in comparison.values()]
        if max(high_quality_counts) < 5:
            recommendations.append("Focus on improving signal generation logic before applying filters")
        
        return recommendations


def main():
    """Main execution function."""
    tester = ComprehensiveFilterTester()
    comparison = tester.run_comprehensive_test()
    
    print("\n" + "="*80)
    print("🎯 COMPREHENSIVE TESTING COMPLETED")
    print("="*80)
    print(f"Best configuration: {comparison['best_configuration']}")
    print("\nCheck the generated JSON file for detailed results.")


if __name__ == "__main__":
    main()