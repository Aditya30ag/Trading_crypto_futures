#!/usr/bin/env python3
"""
Enhanced Strategy with Integrated Filters and Dynamic Risk Management
Combines volume-liquidity filters, volume-volatility filters, and dynamic risk management
"""

from typing import Dict, List, Optional
import yaml
from src.data.fetcher import CoinDCXFetcher
from src.data.indicators import TechnicalIndicators
from src.utils.logger import setup_logger
from src.utils.enhanced_filters import EnhancedVolumeFilters
from src.utils.dynamic_risk_manager import DynamicRiskManager


class EnhancedStrategyWithFilters:
    """
    Enhanced trading strategy that integrates all the new filtering and risk management systems.
    """
    
    def __init__(self, balance: float = 10000):
        self.balance = balance
        self.fetcher = CoinDCXFetcher()
        self.indicators = TechnicalIndicators()
        self.volume_filters = EnhancedVolumeFilters()
        self.risk_manager = DynamicRiskManager()
        self.logger = setup_logger()
        
        # Load configuration
        try:
            with open("config/config.yaml", "r") as file:
                self.config = yaml.safe_load(file)
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            self.config = {"trading": {"position_size_ratio": 0.25}}
        
        self.logger.info("✨ Enhanced Strategy with Filters initialized")
    
    def generate_enhanced_signal(self, symbol: str, timeframe: str, strategy: str = "swing", 
                               direction: Optional[str] = None) -> Optional[Dict]:
        """
        Generate enhanced signal with all filters and dynamic risk management applied.
        
        Args:
            symbol: Trading symbol
            timeframe: Chart timeframe
            strategy: Strategy type (scalping, swing, long_swing)
            direction: Optional forced direction (long/short)
            
        Returns:
            Enhanced signal dict or None if filtered out
        """
        try:
            self.logger.info(f"🎯 Generating enhanced signal for {symbol} ({strategy}, {timeframe})")
            
            # Step 1: Fetch market data
            market_data = self.fetcher.fetch_market_data(symbol)
            if not market_data or "last_price" not in market_data:
                self.logger.warning(f"❌ No market data for {symbol}")
                return None
            
            current_price = float(market_data["last_price"])
            
            # Step 2: Fetch sufficient candlestick data
            min_candles = 100
            candles = self.fetcher.fetch_candlestick_data(symbol, timeframe, min_candles)
            if not candles or len(candles) < 50:
                self.logger.warning(f"❌ Insufficient data for {symbol}: {len(candles) if candles else 0} candles")
                return None
            
            # Step 3: Apply Volume-Liquidity and Volume-Volatility Filters
            self.logger.info(f"🔍 Applying volume filters to {symbol}")
            volume_analysis = self.volume_filters.combined_filter_analysis(symbol, candles, strategy)
            
            if not volume_analysis["passed"]:
                self.logger.info(f"❌ {symbol} failed volume filters: {volume_analysis['recommendation']}")
                return {
                    "status": "filtered_out",
                    "reason": "Failed volume filters",
                    "volume_analysis": volume_analysis
                }
            
            self.logger.info(f"✅ {symbol} passed volume filters (score: {volume_analysis['combined_score']:.1f}/100)")
            
            # Step 4: Generate base signal using appropriate strategy
            base_signal = self._generate_base_signal(symbol, timeframe, strategy, direction, candles, current_price)
            if not base_signal:
                self.logger.info(f"❌ No base signal generated for {symbol}")
                return None
            
            self.logger.info(f"✅ Base signal generated for {symbol}: {base_signal['side']} with score {base_signal['score']}")
            
            # Step 5: Apply Dynamic Risk Management
            self.logger.info(f"🛡️ Applying dynamic risk management to {symbol}")
            
            # Calculate optimal hold time
            hold_time_result = self.risk_manager.calculate_optimal_hold_time(
                symbol, candles, strategy, timeframe
            )
            
            # Calculate dynamic stop loss
            stop_loss_result = self.risk_manager.calculate_dynamic_stop_loss(
                symbol, candles, base_signal["entry_price"], base_signal["side"], strategy
            )
            
            # Calculate dynamic take profit
            take_profit_result = self.risk_manager.calculate_dynamic_take_profit(
                symbol, candles, base_signal["entry_price"], 
                stop_loss_result["stop_loss"], base_signal["side"], strategy
            )
            
            # Step 6: Risk Validation
            risk_pct = stop_loss_result["risk_pct"]
            risk_reward_tp1 = take_profit_result["risk_reward_tp1"]
            
            # Risk validation thresholds
            max_risk_limits = {"scalping": 3.0, "swing": 4.0, "long_swing": 5.0}
            min_risk_reward = {"scalping": 1.5, "swing": 2.0, "long_swing": 2.5}
            
            max_risk = max_risk_limits.get(strategy, 4.0)
            min_rr = min_risk_reward.get(strategy, 2.0)
            
            if risk_pct > max_risk:
                self.logger.warning(f"❌ {symbol} risk too high: {risk_pct:.2f}% > {max_risk}%")
                return {
                    "status": "filtered_out",
                    "reason": f"Risk too high: {risk_pct:.2f}% > {max_risk}%"
                }
            
            if risk_reward_tp1 < min_rr:
                self.logger.warning(f"❌ {symbol} poor risk-reward: {risk_reward_tp1:.2f} < {min_rr}")
                return {
                    "status": "filtered_out",
                    "reason": f"Poor risk-reward: {risk_reward_tp1:.2f} < {min_rr}"
                }
            
            # Step 7: Hold Time Validation
            max_hold_limits = {"scalping": 3.0, "swing": 10.0, "long_swing": 18.0}
            max_hold = max_hold_limits.get(strategy, 10.0)
            
            if hold_time_result["hold_time_hours"] > max_hold:
                self.logger.warning(f"❌ {symbol} hold time too long: {hold_time_result['hold_time_hours']:.1f}h > {max_hold}h")
                return {
                    "status": "filtered_out",
                    "reason": f"Hold time too long: {hold_time_result['hold_time_hours']:.1f}h > {max_hold}h"
                }
            
            # Step 8: Build Enhanced Signal
            enhanced_signal = self._build_enhanced_signal(
                base_signal, volume_analysis, hold_time_result, 
                stop_loss_result, take_profit_result, symbol, timeframe, strategy
            )
            
            # Step 9: Final Quality Assessment
            quality_score = self._calculate_final_quality_score(
                enhanced_signal, volume_analysis, risk_pct, risk_reward_tp1, 
                hold_time_result["confidence"]
            )
            
            enhanced_signal["final_quality_score"] = quality_score
            enhanced_signal["status"] = "signal_generated"
            
            # Quality threshold for signal acceptance
            min_quality_threshold = 70.0
            if quality_score < min_quality_threshold:
                self.logger.warning(f"❌ {symbol} quality too low: {quality_score:.1f} < {min_quality_threshold}")
                return {
                    "status": "filtered_out",
                    "reason": f"Quality too low: {quality_score:.1f} < {min_quality_threshold}",
                    "quality_score": quality_score
                }
            
            self.logger.info(f"✅ Enhanced signal generated for {symbol}:")
            self.logger.info(f"   📊 Quality Score: {quality_score:.1f}/100")
            self.logger.info(f"   💰 Risk: {risk_pct:.2f}% | R:R {risk_reward_tp1:.2f}:1")
            self.logger.info(f"   ⏰ Hold Time: {hold_time_result['hold_time_hours']:.1f}h")
            self.logger.info(f"   🎯 Entry: {enhanced_signal['entry_price']:.6f}")
            self.logger.info(f"   🛡️ SL: {enhanced_signal['stop_loss']:.6f}")
            self.logger.info(f"   📈 TP: {enhanced_signal['take_profit']:.6f}")
            
            return enhanced_signal
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced signal generation error for {symbol}: {e}")
            return None
    
    def _generate_base_signal(self, symbol: str, timeframe: str, strategy: str, 
                            direction: Optional[str], candles: List[Dict], current_price: float) -> Optional[Dict]:
        """
        Generate base signal using the appropriate strategy.
        """
        try:
            # Import strategy classes
            if strategy == "scalping":
                from src.strategies.scalping import ScalpingStrategy
                strategy_instance = ScalpingStrategy(self.balance)
                return strategy_instance.generate_signal(symbol, timeframe)
            elif strategy == "swing":
                from src.strategies.swing import SwingStrategy
                strategy_instance = SwingStrategy(self.balance)
                return strategy_instance.generate_signal(symbol, timeframe, direction)
            elif strategy == "long_swing":
                from src.strategies.long_swing import LongSwingStrategy
                strategy_instance = LongSwingStrategy(self.balance)
                return strategy_instance.generate_signal(symbol, timeframe, direction)
            else:
                self.logger.error(f"Unknown strategy: {strategy}")
                return None
                
        except Exception as e:
            self.logger.error(f"Base signal generation error: {e}")
            return None
    
    def _build_enhanced_signal(self, base_signal: Dict, volume_analysis: Dict,
                             hold_time_result: Dict, stop_loss_result: Dict,
                             take_profit_result: Dict, symbol: str, timeframe: str, strategy: str) -> Dict:
        """
        Build the final enhanced signal with all improvements.
        """
        enhanced_signal = base_signal.copy()
        
        # Update with dynamic risk management
        enhanced_signal.update({
            "max_hold_time": hold_time_result["hold_time_hours"],
            "stop_loss": stop_loss_result["stop_loss"],
            "take_profit": take_profit_result["tp1"],
            "tp1": take_profit_result["tp1"],
            "tp2": take_profit_result["tp2"],
            
            # Enhanced metadata
            "enhancement_version": "v2.0",
            "filters_applied": {
                "volume_liquidity": True,
                "volume_volatility": True,
                "dynamic_risk_management": True
            },
            
            # Analysis results
            "volume_analysis": {
                "passed": volume_analysis["passed"],
                "combined_score": volume_analysis["combined_score"],
                "liquidity_score": volume_analysis["summary"]["liquidity_score"],
                "volatility_score": volume_analysis["summary"]["volatility_score"]
            },
            
            "risk_management": {
                "method": stop_loss_result["method"],
                "risk_percentage": stop_loss_result["risk_pct"],
                "risk_reward_tp1": take_profit_result["risk_reward_tp1"],
                "risk_reward_tp2": take_profit_result["risk_reward_tp2"],
                "dynamic_hold_time": hold_time_result["hold_time_hours"],
                "hold_time_confidence": hold_time_result["confidence"]
            },
            
            # Market conditions at signal time
            "market_conditions": hold_time_result.get("market_conditions", {}),
            
            # Signal generation timestamp
            "enhanced_timestamp": self._get_current_timestamp(),
            
            # Enhanced scoring
            "enhancement_score_breakdown": {
                "base_signal_score": base_signal.get("score", 0),
                "volume_filter_score": volume_analysis["combined_score"],
                "risk_management_score": stop_loss_result.get("confidence", 0.5) * 100,
                "hold_time_score": hold_time_result["confidence"] * 100
            }
        })
        
        return enhanced_signal
    
    def _calculate_final_quality_score(self, signal: Dict, volume_analysis: Dict,
                                     risk_pct: float, risk_reward: float, hold_time_confidence: float) -> float:
        """
        Calculate final quality score for the enhanced signal.
        """
        # Base signal quality (30 points)
        base_score = signal.get("score", 0)
        base_max = signal.get("score_max", 10)
        base_quality = (base_score / base_max) * 30 if base_max > 0 else 0
        
        # Volume analysis quality (25 points)
        volume_quality = (volume_analysis["combined_score"] / 100) * 25
        
        # Risk management quality (25 points)
        risk_quality = 0
        if risk_pct <= 2.0:
            risk_quality += 15  # Excellent risk
        elif risk_pct <= 3.0:
            risk_quality += 12  # Good risk
        elif risk_pct <= 4.0:
            risk_quality += 8   # Acceptable risk
        else:
            risk_quality += 4   # Poor risk
        
        if risk_reward >= 3.0:
            risk_quality += 10  # Excellent R:R
        elif risk_reward >= 2.5:
            risk_quality += 8   # Good R:R
        elif risk_reward >= 2.0:
            risk_quality += 6   # Acceptable R:R
        else:
            risk_quality += 3   # Poor R:R
        
        # Hold time confidence (10 points)
        hold_time_quality = hold_time_confidence * 10
        
        # Market timing bonus (10 points)
        timing_quality = 10  # Base timing score
        
        total_quality = base_quality + volume_quality + risk_quality + hold_time_quality + timing_quality
        return min(total_quality, 100.0)
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def generate_multiple_enhanced_signals(self, symbols: List[str], strategy: str = "swing", 
                                         max_signals: int = 10) -> List[Dict]:
        """
        Generate multiple enhanced signals and return the best ones.
        
        Args:
            symbols: List of symbols to analyze
            strategy: Strategy type
            max_signals: Maximum number of signals to return
            
        Returns:
            List of enhanced signals sorted by quality
        """
        self.logger.info(f"🎯 Generating enhanced signals for {len(symbols)} symbols ({strategy})")
        
        signals = []
        timeframes = {
            "scalping": ["5m"],
            "swing": ["15m", "1h"],
            "long_swing": ["30m", "1h"]
        }
        
        strategy_timeframes = timeframes.get(strategy, ["15m"])
        
        for symbol in symbols:
            for timeframe in strategy_timeframes:
                signal = self.generate_enhanced_signal(symbol, timeframe, strategy)
                if signal and signal.get("status") == "signal_generated":
                    signals.append(signal)
        
        # Sort by final quality score
        signals.sort(key=lambda x: x.get("final_quality_score", 0), reverse=True)
        
        # Return top signals
        top_signals = signals[:max_signals]
        
        self.logger.info(f"✅ Generated {len(signals)} signals, returning top {len(top_signals)}")
        for i, signal in enumerate(top_signals[:5], 1):
            quality = signal.get("final_quality_score", 0)
            symbol = signal.get("symbol", "Unknown")
            side = signal.get("side", "Unknown")
            self.logger.info(f"   {i}. {symbol} {side}: Quality {quality:.1f}/100")
        
        return top_signals