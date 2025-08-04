#!/usr/bin/env python3
"""
Enhanced Trading Filters for Volume-Liquidity and Volume-Volatility Analysis
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from src.utils.logger import setup_logger
from src.data.fetcher import CoinDCXFetcher
from src.data.indicators import TechnicalIndicators


class EnhancedVolumeFilters:
    """
    Enhanced volume filtering system for better signal quality and consistency.
    Implements volume-liquidity and volume-volatility filters.
    """
    
    def __init__(self):
        self.logger = setup_logger()
        self.fetcher = CoinDCXFetcher()
        self.indicators = TechnicalIndicators()
    
    def volume_liquidity_filter(self, symbol: str, candles: List[Dict], strategy: str = "swing") -> Dict:
        """
        Volume-Liquidity Filter: Analyzes market depth and liquidity quality.
        
        Args:
            symbol: Trading symbol
            candles: Candlestick data
            strategy: Trading strategy (scalping, swing, long_swing)
            
        Returns:
            Dict with liquidity analysis and pass/fail status
        """
        try:
            self.logger.info(f"🔍 Volume-Liquidity Filter Analysis for {symbol}")
            
            # Get order book for real-time liquidity
            order_book = self.fetcher.fetch_order_book(symbol)
            if not order_book:
                return {"passed": False, "reason": "Order book unavailable", "score": 0}
            
            # Calculate volume metrics
            volumes = [float(candle["volume"]) for candle in candles[-50:]]
            prices = [float(candle["close"]) for candle in candles[-50:]]
            
            if len(volumes) < 20:
                return {"passed": False, "reason": "Insufficient volume data", "score": 0}
            
            # Core Volume Metrics
            avg_volume_20 = np.mean(volumes[-20:])
            avg_volume_50 = np.mean(volumes) if len(volumes) >= 50 else avg_volume_20
            current_volume = volumes[-1]
            volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 0
            
            # Volume Consistency (Standard Deviation Analysis)
            volume_std = np.std(volumes[-20:])
            volume_cv = volume_std / avg_volume_20 if avg_volume_20 > 0 else float('inf')  # Coefficient of Variation
            
            # Price-Volume Correlation (Strong correlation indicates healthy liquidity)
            price_change = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            volume_change = [(volumes[i] - volumes[i-1]) / volumes[i-1] for i in range(1, len(volumes))]
            
            if len(price_change) > 10 and len(volume_change) > 10:
                correlation = np.corrcoef(price_change[-20:], volume_change[-20:])[0,1]
                correlation = correlation if not np.isnan(correlation) else 0
            else:
                correlation = 0
            
            # Order Book Depth Analysis
            bid_volume = float(order_book.get("bid_vol", 0))
            ask_volume = float(order_book.get("ask_vol", 0))
            spread = float(order_book.get("spread", 0))
            current_price = float(order_book.get("best_bid", prices[-1]))
            
            spread_pct = (spread / current_price) * 100 if current_price > 0 else 100
            depth_ratio = bid_volume / ask_volume if ask_volume > 0 else 0
            total_depth = bid_volume + ask_volume
            
            # Strategy-specific thresholds
            thresholds = {
                "scalping": {
                    "min_volume": 250000,
                    "min_volume_ratio": 0.8,
                    "max_volume_cv": 2.0,
                    "max_spread_pct": 0.3,
                    "min_depth": 50000,
                    "min_correlation": -0.5
                },
                "swing": {
                    "min_volume": 400000,
                    "min_volume_ratio": 0.6,
                    "max_volume_cv": 2.5,
                    "max_spread_pct": 0.8,
                    "min_depth": 75000,
                    "min_correlation": -0.3
                },
                "long_swing": {
                    "min_volume": 500000,
                    "min_volume_ratio": 0.5,
                    "max_volume_cv": 3.0,
                    "max_spread_pct": 1.2,
                    "min_depth": 100000,
                    "min_correlation": -0.2
                }
            }
            
            thresh = thresholds.get(strategy, thresholds["swing"])
            
            # Scoring system (0-100)
            score = 0
            checks = []
            
            # Volume adequacy (25 points)
            if avg_volume_20 >= thresh["min_volume"]:
                score += 25
                checks.append(f"✅ Volume adequate: {avg_volume_20:,.0f} >= {thresh['min_volume']:,}")
            else:
                checks.append(f"❌ Volume too low: {avg_volume_20:,.0f} < {thresh['min_volume']:,}")
            
            # Volume consistency (20 points)
            if volume_ratio >= thresh["min_volume_ratio"]:
                score += 20
                checks.append(f"✅ Volume ratio good: {volume_ratio:.2f} >= {thresh['min_volume_ratio']}")
            else:
                checks.append(f"❌ Volume ratio low: {volume_ratio:.2f} < {thresh['min_volume_ratio']}")
            
            # Volume stability (15 points)
            if volume_cv <= thresh["max_volume_cv"]:
                score += 15
                checks.append(f"✅ Volume stable: CV {volume_cv:.2f} <= {thresh['max_volume_cv']}")
            else:
                checks.append(f"⚠️ Volume volatile: CV {volume_cv:.2f} > {thresh['max_volume_cv']}")
            
            # Spread tightness (20 points)
            if spread_pct <= thresh["max_spread_pct"]:
                score += 20
                checks.append(f"✅ Spread tight: {spread_pct:.3f}% <= {thresh['max_spread_pct']}%")
            else:
                checks.append(f"❌ Spread wide: {spread_pct:.3f}% > {thresh['max_spread_pct']}%")
            
            # Market depth (15 points)
            if total_depth >= thresh["min_depth"]:
                score += 15
                checks.append(f"✅ Good depth: {total_depth:,.0f} >= {thresh['min_depth']:,}")
            else:
                checks.append(f"❌ Poor depth: {total_depth:,.0f} < {thresh['min_depth']:,}")
            
            # Price-volume correlation (5 points bonus)
            if correlation >= thresh["min_correlation"]:
                score += 5
                checks.append(f"✅ Healthy correlation: {correlation:.3f} >= {thresh['min_correlation']}")
            else:
                checks.append(f"⚠️ Poor correlation: {correlation:.3f} < {thresh['min_correlation']}")
            
            # Determine pass/fail (need 70+ score for pass)
            passed = score >= 70
            
            result = {
                "passed": passed,
                "score": score,
                "metrics": {
                    "avg_volume_20": avg_volume_20,
                    "current_volume": current_volume,
                    "volume_ratio": volume_ratio,
                    "volume_cv": volume_cv,
                    "spread_pct": spread_pct,
                    "total_depth": total_depth,
                    "correlation": correlation
                },
                "checks": checks,
                "reason": f"Liquidity score: {score}/100" + (" (PASS)" if passed else " (FAIL)")
            }
            
            self.logger.info(f"📊 {symbol} Liquidity Score: {score}/100 {'✅ PASS' if passed else '❌ FAIL'}")
            for check in checks:
                self.logger.info(f"   {check}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Volume-Liquidity Filter error for {symbol}: {e}")
            return {"passed": False, "reason": f"Filter error: {e}", "score": 0}
    
    def volume_volatility_filter(self, symbol: str, candles: List[Dict], strategy: str = "swing") -> Dict:
        """
        Volume-Volatility Filter: Analyzes volume-price volatility relationship.
        
        Args:
            symbol: Trading symbol
            candles: Candlestick data
            strategy: Trading strategy
            
        Returns:
            Dict with volatility analysis and pass/fail status
        """
        try:
            self.logger.info(f"⚡ Volume-Volatility Filter Analysis for {symbol}")
            
            if len(candles) < 50:
                return {"passed": False, "reason": "Insufficient data for volatility analysis", "score": 0}
            
            # Extract price and volume data
            closes = [float(candle["close"]) for candle in candles[-50:]]
            volumes = [float(candle["volume"]) for candle in candles[-50:]]
            highs = [float(candle["high"]) for candle in candles[-50:]]
            lows = [float(candle["low"]) for candle in candles[-50:]]
            
            # Calculate ATR for volatility measurement
            atr = self.indicators.calculate_atr(candles)
            if atr is None:
                return {"passed": False, "reason": "ATR calculation failed", "score": 0}
            
            current_price = closes[-1]
            atr_pct = (atr / current_price) * 100
            
            # Price volatility metrics
            price_returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
            price_volatility = np.std(price_returns) * 100  # As percentage
            
            # True Range calculations for accurate volatility
            true_ranges = []
            for i in range(1, len(candles[-50:])):
                tr1 = highs[i] - lows[i]
                tr2 = abs(highs[i] - closes[i-1])
                tr3 = abs(lows[i] - closes[i-1])
                true_ranges.append(max(tr1, tr2, tr3))
            
            avg_true_range = np.mean(true_ranges) if true_ranges else atr
            
            # Volume-based volatility (Volume Rate of Change)
            volume_roc = [(volumes[i] - volumes[i-1]) / volumes[i-1] for i in range(1, len(volumes)) if volumes[i-1] > 0]
            volume_volatility = np.std(volume_roc) * 100 if volume_roc else 0
            
            # Volume-Price Efficiency (VPE) - How much price movement per unit volume
            volume_sma = np.mean(volumes[-20:])
            price_movement = abs(closes[-1] - closes[-20]) / closes[-20] * 100
            vpe = price_movement / (volume_sma / 1000000) if volume_sma > 0 else 0  # Movement per 1M volume
            
            # Volatility Quality Score
            # Lower volatility is better for consistent trading
            # Higher VPE indicates efficient price discovery
            
            # Strategy-specific thresholds
            volatility_thresholds = {
                "scalping": {
                    "max_atr_pct": 3.0,        # 3% max daily ATR
                    "max_price_vol": 2.5,      # 2.5% max price volatility
                    "max_volume_vol": 150,     # 150% max volume volatility
                    "min_vpe": 0.1,           # Minimum efficiency
                    "max_vpe": 5.0            # Maximum efficiency (too high = manipulated)
                },
                "swing": {
                    "max_atr_pct": 5.0,        # 5% max daily ATR
                    "max_price_vol": 4.0,      # 4% max price volatility
                    "max_volume_vol": 200,     # 200% max volume volatility
                    "min_vpe": 0.05,          # Minimum efficiency
                    "max_vpe": 8.0            # Maximum efficiency
                },
                "long_swing": {
                    "max_atr_pct": 7.0,        # 7% max daily ATR
                    "max_price_vol": 6.0,      # 6% max price volatility
                    "max_volume_vol": 300,     # 300% max volume volatility
                    "min_vpe": 0.02,          # Minimum efficiency
                    "max_vpe": 12.0           # Maximum efficiency
                }
            }
            
            thresh = volatility_thresholds.get(strategy, volatility_thresholds["swing"])
            
            # Scoring system (0-100)
            score = 0
            checks = []
            
            # ATR percentage check (25 points)
            if atr_pct <= thresh["max_atr_pct"]:
                score += 25
                checks.append(f"✅ ATR reasonable: {atr_pct:.2f}% <= {thresh['max_atr_pct']}%")
            else:
                checks.append(f"❌ ATR too high: {atr_pct:.2f}% > {thresh['max_atr_pct']}%")
            
            # Price volatility check (25 points)
            if price_volatility <= thresh["max_price_vol"]:
                score += 25
                checks.append(f"✅ Price volatility controlled: {price_volatility:.2f}% <= {thresh['max_price_vol']}%")
            else:
                checks.append(f"❌ Price too volatile: {price_volatility:.2f}% > {thresh['max_price_vol']}%")
            
            # Volume volatility check (20 points)
            if volume_volatility <= thresh["max_volume_vol"]:
                score += 20
                checks.append(f"✅ Volume volatility acceptable: {volume_volatility:.1f}% <= {thresh['max_volume_vol']}%")
            else:
                checks.append(f"⚠️ Volume very volatile: {volume_volatility:.1f}% > {thresh['max_volume_vol']}%")
            
            # Volume-Price Efficiency check (20 points)
            if thresh["min_vpe"] <= vpe <= thresh["max_vpe"]:
                score += 20
                checks.append(f"✅ VPE efficient: {vpe:.3f} in range [{thresh['min_vpe']}, {thresh['max_vpe']}]")
            else:
                checks.append(f"❌ VPE inefficient: {vpe:.3f} outside range [{thresh['min_vpe']}, {thresh['max_vpe']}]")
            
            # Consistency bonus (10 points)
            recent_atr = avg_true_range / current_price * 100
            if abs(atr_pct - recent_atr) <= 1.0:  # ATR consistent
                score += 10
                checks.append(f"✅ Volatility consistent: ATR deviation {abs(atr_pct - recent_atr):.2f}% <= 1.0%")
            else:
                checks.append(f"⚠️ Volatility inconsistent: ATR deviation {abs(atr_pct - recent_atr):.2f}% > 1.0%")
            
            # Determine pass/fail (need 60+ score for pass)
            passed = score >= 60
            
            result = {
                "passed": passed,
                "score": score,
                "metrics": {
                    "atr_pct": atr_pct,
                    "price_volatility": price_volatility,
                    "volume_volatility": volume_volatility,
                    "vpe": vpe,
                    "recent_atr_pct": recent_atr
                },
                "checks": checks,
                "reason": f"Volatility score: {score}/100" + (" (PASS)" if passed else " (FAIL)")
            }
            
            self.logger.info(f"⚡ {symbol} Volatility Score: {score}/100 {'✅ PASS' if passed else '❌ FAIL'}")
            for check in checks:
                self.logger.info(f"   {check}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Volume-Volatility Filter error for {symbol}: {e}")
            return {"passed": False, "reason": f"Filter error: {e}", "score": 0}
    
    def combined_filter_analysis(self, symbol: str, candles: List[Dict], strategy: str = "swing") -> Dict:
        """
        Combined analysis using both volume-liquidity and volume-volatility filters.
        
        Args:
            symbol: Trading symbol
            candles: Candlestick data
            strategy: Trading strategy
            
        Returns:
            Dict with combined analysis and final recommendation
        """
        try:
            self.logger.info(f"🔄 Combined Filter Analysis for {symbol} ({strategy})")
            
            # Run both filters
            liquidity_result = self.volume_liquidity_filter(symbol, candles, strategy)
            volatility_result = self.volume_volatility_filter(symbol, candles, strategy)
            
            # Combined scoring (weighted average)
            liquidity_weight = 0.6  # Liquidity is more important
            volatility_weight = 0.4
            
            combined_score = (
                liquidity_result["score"] * liquidity_weight + 
                volatility_result["score"] * volatility_weight
            )
            
            # Both filters must pass for overall pass
            overall_passed = liquidity_result["passed"] and volatility_result["passed"]
            
            # Create recommendation
            if overall_passed:
                if combined_score >= 80:
                    recommendation = "EXCELLENT - High-quality signal candidate"
                elif combined_score >= 70:
                    recommendation = "GOOD - Suitable for trading"
                else:
                    recommendation = "ACCEPTABLE - Monitor closely"
            else:
                recommendation = "REJECTED - Fails quality requirements"
            
            result = {
                "passed": overall_passed,
                "combined_score": round(combined_score, 1),
                "recommendation": recommendation,
                "liquidity_analysis": liquidity_result,
                "volatility_analysis": volatility_result,
                "summary": {
                    "liquidity_score": liquidity_result["score"],
                    "volatility_score": volatility_result["score"],
                    "liquidity_passed": liquidity_result["passed"],
                    "volatility_passed": volatility_result["passed"]
                }
            }
            
            self.logger.info(f"📋 {symbol} Combined Analysis:")
            self.logger.info(f"   💧 Liquidity: {liquidity_result['score']}/100 {'✅' if liquidity_result['passed'] else '❌'}")
            self.logger.info(f"   ⚡ Volatility: {volatility_result['score']}/100 {'✅' if volatility_result['passed'] else '❌'}")
            self.logger.info(f"   🎯 Combined: {combined_score:.1f}/100")
            self.logger.info(f"   📊 Result: {recommendation}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Combined Filter Analysis error for {symbol}: {e}")
            return {
                "passed": False,
                "combined_score": 0,
                "recommendation": f"ERROR - {e}",
                "liquidity_analysis": {"passed": False, "score": 0},
                "volatility_analysis": {"passed": False, "score": 0}
            }