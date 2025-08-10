#!/usr/bin/env python3
"""
Enhanced Signal Generator with Dynamic Exit Management
Implements single-entry enforcement and intelligent post-entry management
"""

import time
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd
import concurrent.futures

from src.data.fetcher import CoinDCXFetcher
from src.data.indicators import TechnicalIndicators
from src.utils.logger import setup_logger


class TradeDirection(Enum):
    """Trade direction"""
    LONG = "long"
    SHORT = "short"


class ExitReason(Enum):
    """Exit reasons"""
    RSI_OVERBOUGHT = "rsi_overbought"
    RSI_OVERSOLD = "rsi_oversold"
    RSI_DIVERGENCE = "rsi_divergence"
    STOCH_FLIP = "stoch_flip"
    MACD_REVERSAL = "macd_reversal"
    VWAP_BREACH = "vwap_breach"
    VOLUME_DROP = "volume_drop"
    VOLUME_SPIKE = "volume_spike"
    CANDLE_TRAIL = "candle_trail"
    ATR_TRAIL = "atr_trail"
    BB_BREACH = "bb_breach"
    REVERSAL_CANDLE = "reversal_candle"
    MAX_HOLD_TIME = "max_hold_time"
    TP_HIT = "tp_hit"
    SL_HIT = "sl_hit"


@dataclass
class EntrySignal:
    """Entry signal with quality metrics"""
    symbol: str
    timeframe: str
    strategy: str
    direction: TradeDirection
    entry_price: float
    stop_loss: float
    take_profit: float
    tp1: float
    tp2: float
    max_hold_time: int  # minutes
    score: int
    score_reasons: List[str]
    indicators: Dict[str, Any]
    timestamp: datetime
    estimated_profit_inr: float
    confidence: float = 0.0  # Add confidence field
    quality_score: float = 0.0  # Add quality score field
    normalized_score: float = 0.0  # Add normalized score field for fair comparison
    
    def to_dict(self):
        return asdict(self)


@dataclass
class ActiveTrade:
    """Active trade with dynamic management"""
    trade_id: str
    symbol: str
    timeframe: str
    strategy: str
    direction: TradeDirection
    entry_price: float
    entry_time: datetime
    current_price: float
    stop_loss: float
    take_profit: float
    tp1: float
    tp2: float
    max_hold_time: int
    current_pnl: float
    max_pnl: float
    min_pnl: float
    candle_count: int
    last_update: datetime
    
    # Dynamic levels
    breakeven_stop: float
    trailing_stop: float
    profit_lock_level: float
    
    # Signal journey tracking
    entry_indicators: Dict[str, Any]
    current_indicators: Dict[str, Any]
    signal_strength_history: List[float]
    
    def to_dict(self):
        return asdict(self)


class DynamicExitMonitor:
    """
    Monitors open trades for dynamic exit conditions and manages stop-loss/take-profit adjustments.
    """
    def __init__(self, indicators: TechnicalIndicators, fetcher: CoinDCXFetcher, logger=None):
        self.indicators = indicators
        self.fetcher = fetcher
        self.logger = logger or setup_logger()

    def should_exit(self, trade: ActiveTrade) -> Optional[ExitReason]:
        """
        Check for exit conditions: trend weakening, indicator reversals, volume anomalies, VWAP/EMA breaches, higher timeframe exhaustion.
        PROTECTS PROFITS: No dynamic exits until at least 7% profit is reached.
        Returns an ExitReason if exit is recommended, else None.
        """
        try:
            # Calculate current profit percentage
            if trade.direction == TradeDirection.LONG:
                profit_pct = ((trade.current_price - trade.entry_price) / trade.entry_price) * 100
            else:
                profit_pct = ((trade.entry_price - trade.current_price) / trade.entry_price) * 100
            
            # CRITICAL: No dynamic exits until at least 5% profit
            if profit_pct < 5.0:
                self.logger.debug(f"{trade.symbol}: Profit {profit_pct:.2f}% < 5%, allowing trade to run")
                return None
            
            # Only check exit conditions if profit >= 5%
            self.logger.info(f"{trade.symbol}: Profit {profit_pct:.2f}% >= 5%, checking exit conditions")
            
            # Fetch latest candles for the trade's symbol and timeframe
            candles = self.fetcher.fetch_candlestick_data(trade.symbol, trade.timeframe, 100)
            if not candles or len(candles) < 50:
                return None
                
            # Calculate indicators
            rsi = self.indicators.calculate_rsi(candles)
            macd = self.indicators.calculate_macd(candles)
            stoch = self.indicators.calculate_stoch_rsi(candles)
            vwap = self.indicators.calculate_vwap(candles)
            ema_20 = self.indicators.calculate_ema(candles, 20)
            bb_upper, bb_lower = self.indicators.calculate_bollinger_bands(candles)
            atr = self.indicators.calculate_atr(candles)
            volume = float(candles[-1]["volume"])

            # STRICT EXIT CONDITIONS - Only trigger on major reversals
            if trade.direction == TradeDirection.LONG:
                # Long exit conditions (only if profit >= 7%)
                if rsi and rsi > 85:  # Very overbought
                    return ExitReason.RSI_OVERBOUGHT
                if macd and macd < -0.001:  # Strong bearish MACD
                    return ExitReason.MACD_REVERSAL
                if stoch and stoch["stoch_rsi"] > 0.95:  # Extremely overbought
                    return ExitReason.STOCH_FLIP
            else:
                # Short exit conditions (only if profit >= 7%)
                if rsi and rsi < 15:  # Very oversold
                    return ExitReason.RSI_OVERSOLD
                if macd and macd > 0.001:  # Strong bullish MACD
                    return ExitReason.MACD_REVERSAL
                if stoch and stoch["stoch_rsi"] < 0.05:  # Extremely oversold
                    return ExitReason.STOCH_FLIP

            # Volume conditions - only extreme drops
            volume_sma = np.mean([float(c["volume"]) for c in candles[-20:]])
            if volume < 0.2 * volume_sma:  # Very low volume
                return ExitReason.VOLUME_DROP

            # VWAP/EMA breach - only if significant
            if trade.direction == TradeDirection.LONG and vwap:
                vwap_breach_pct = ((trade.current_price - vwap) / vwap) * 100
                if vwap_breach_pct < -2.0:  # 2% below VWAP
                    return ExitReason.VWAP_BREACH
            if trade.direction == TradeDirection.SHORT and vwap:
                vwap_breach_pct = ((trade.current_price - vwap) / vwap) * 100
                if vwap_breach_pct > 2.0:  # 2% above VWAP
                    return ExitReason.VWAP_BREACH

            # Bollinger Band boundaries - only extreme breaches
            if trade.direction == TradeDirection.LONG and bb_upper:
                bb_breach_pct = ((trade.current_price - bb_upper) / bb_upper) * 100
                if bb_breach_pct > 1.0:  # 1% above upper BB
                    return ExitReason.BB_BREACH
            if trade.direction == TradeDirection.SHORT and bb_lower:
                bb_breach_pct = ((trade.current_price - bb_lower) / bb_lower) * 100
                if bb_breach_pct < -1.0:  # 1% below lower BB
                    return ExitReason.BB_BREACH

            return None
        except Exception as e:
            self.logger.error(f"DynamicExitMonitor error: {e}")
            return None


class EnhancedSignalGenerator:
    """
    Enhanced signal generator with single-entry enforcement and dynamic exit management.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = setup_logger()
        self.fetcher = CoinDCXFetcher()
        self.indicators = TechnicalIndicators()
        self.exit_monitor = DynamicExitMonitor(self.indicators, self.fetcher, self.logger)
        
        # Active trades tracking (one per symbol-timeframe combo)
        self.active_trades: Dict[str, ActiveTrade] = {}
        self.trade_history: List[Dict] = []
        
        # Signal and trade limits
        signal_config = config.get("trading", {}).get("signal_generation", {})
        self.max_trades = signal_config.get("max_active_trades", 20)  # Use config value
        self.max_signals = signal_config.get("max_signals", 40)  # Generate 40 signals
        self.max_monitor_signals = signal_config.get("max_monitor_signals", 20)  # Monitor top 20
        self.signals_generated = 0  # Counter for generated signals
        self.trades_entered = 0  # Counter for entered trades
        
        # Exit thresholds
        self.exit_thresholds = {
            "rsi_overbought": 72,
            "rsi_oversold": 28,
            "volume_sma_period": 20,
            "atr_trail_period": 14,
            "candle_trail_threshold": 0.4,  # 0.4% move
            "breakeven_threshold": 0.4,     # 0.4% move
            "profit_lock_threshold": 0.8,   # 0.8% move
            "trailing_activation": 0.6      # 0.6% move
        }
        
        # Monitoring settings
        self.monitoring_interval = 5  # seconds
        self.monitoring_active = False
        self.monitoring_thread = None
        
        self.logger.info("EnhancedSignalGenerator initialized")

    def _fetch_high_volume_symbols(self, min_volume=1000000, max_symbols=50) -> List[str]:
        """Fetch high-volume trading symbols (predefined only, no API)."""
        # Use only the default/predefined symbols, no API fetching
        default_symbols = [
            "B-BTC_USDT", "B-ETH_USDT", "B-XRP_USDT", "B-ADA_USDT", "B-DOT_USDT",
            "B-SOL_USDT", "B-BNB_USDT", "B-MATIC_USDT", "B-LINK_USDT", "B-TRX_USDT",
            "B-AVAX_USDT", "B-ATOM_USDT", "B-UNI_USDT", "B-LTC_USDT", "B-BCH_USDT",
            "B-FIL_USDT", "B-ALGO_USDT", "B-VET_USDT", "B-ICP_USDT", "B-AAVE_USDT",
            "B-APT_USDT", "B-ARB_USDT", "B-OP_USDT", "B-MKR_USDT", "B-COMP_USDT",
            "B-SNX_USDT", "B-1INCH_USDT", "B-ZRX_USDT", "B-BAL_USDT", "B-CRV_USDT",
            "B-REN_USDT", "B-BAND_USDT", "B-OCEAN_USDT", "B-ALPHA_USDT", "B-SAND_USDT",
            "B-MANA_USDT", "B-ENJ_USDT", "B-CHZ_USDT", "B-HOT_USDT", "B-BAT_USDT",
            "B-BRAVE_USDT", "B-ZIL_USDT", "B-VTHO_USDT", "B-VET_USDT", "B-ICX_USDT"
        ]
        return default_symbols[:max_symbols]

    def _fetch_extended_symbols(self, min_volume=500000, max_symbols=75) -> List[str]:
        """Fetch extended list of symbols for additional signal generation (predefined only, no API)."""
        # Use only the extended default/predefined symbols, no API fetching
        extended_symbols = [
            "B-BTC_USDT", "B-ETH_USDT", "B-XRP_USDT", "B-ADA_USDT", "B-DOT_USDT",
            "B-SOL_USDT", "B-BNB_USDT", "B-MATIC_USDT", "B-LINK_USDT", "B-TRX_USDT",
            "B-AVAX_USDT", "B-ATOM_USDT", "B-UNI_USDT", "B-LTC_USDT", "B-BCH_USDT",
            "B-FIL_USDT", "B-ALGO_USDT", "B-VET_USDT", "B-ICP_USDT", "B-AAVE_USDT",
            "B-APT_USDT", "B-ARB_USDT", "B-OP_USDT", "B-MKR_USDT", "B-COMP_USDT",
            "B-SNX_USDT", "B-1INCH_USDT", "B-ZRX_USDT", "B-BAL_USDT", "B-CRV_USDT",
            "B-REN_USDT", "B-BAND_USDT", "B-OCEAN_USDT", "B-ALPHA_USDT", "B-SAND_USDT",
            "B-MANA_USDT", "B-ENJ_USDT", "B-CHZ_USDT", "B-HOT_USDT", "B-BAT_USDT",
            "B-BRAVE_USDT", "B-ZIL_USDT", "B-VTHO_USDT", "B-VET_USDT", "B-ICX_USDT",
            "B-NEAR_USDT", "B-FTM_USDT", "B-EGLD_USDT", "B-THETA_USDT", "B-XLM_USDT",
            "B-EOS_USDT", "B-TRX_USDT", "B-XTZ_USDT", "B-NEO_USDT", "B-ONT_USDT",
            "B-VET_USDT", "B-ICX_USDT", "B-ZIL_USDT", "B-BAT_USDT", "B-ENJ_USDT",
            "B-MANA_USDT", "B-SAND_USDT", "B-ALPHA_USDT", "B-OCEAN_USDT", "B-BAND_USDT",
            "B-REN_USDT", "B-CRV_USDT", "B-BAL_USDT", "B-ZRX_USDT", "B-1INCH_USDT",
            "B-SNX_USDT", "B-COMP_USDT", "B-MKR_USDT", "B-OP_USDT", "B-ARB_USDT",
            "B-APT_USDT", "B-AAVE_USDT", "B-ICP_USDT", "B-VET_USDT", "B-ALGO_USDT"
        ]
        return extended_symbols[:max_symbols]

    def generate_additional_signals(self) -> List[EntrySignal]:
        """Generate additional high-profit signals from extended symbol list."""
        try:
            self.logger.info("=== GENERATING ADDITIONAL HIGH-PROFIT SIGNALS ===")
            
            # Get extended symbol list
            extended_symbols = self._fetch_extended_symbols()
            if not extended_symbols:
                self.logger.warning("No extended symbols found")
                return []
            
            additional_signals = []
            
            # Analyze each symbol with focus on high-profit strategies
            for symbol in extended_symbols:
                symbol_signals = []
                
                # Focus on strategies that typically generate high profits
                high_profit_strategies = ["scalping", "swing", "long_swing"]  # Prioritize all strategies including scalping
                
                for strategy in high_profit_strategies:
                    try:
                        # Use timeframes from config
                        config_timeframes = self.config.get("trading", {}).get("timeframes", {})
                        timeframes = config_timeframes.get(strategy, [])
                        if not timeframes:
                            continue
                        # Test each timeframe for this strategy
                        for timeframe in timeframes:
                            signal = self._generate_signal_for_timeframe(symbol, timeframe, strategy)
                            # Different profit thresholds for different strategies
                            min_profit = 100 if strategy == "scalping" else 300  # Lower threshold for scalping
                            if signal and signal.estimated_profit_inr >= min_profit:
                                symbol_signals.append(signal)
                                self.logger.debug(f"Generated additional {strategy} signal for {symbol} {timeframe}: Profit ₹{signal.estimated_profit_inr:.2f}")
                    except Exception as e:
                        self.logger.error(f"Error generating additional {strategy} signal for {symbol}: {e}")
                
                # Select best signal for this symbol
                if symbol_signals:
                    # Sort by estimated profit
                    symbol_signals.sort(key=lambda s: s.estimated_profit_inr, reverse=True)
                    best_signal = symbol_signals[0]
                    
                    # Only include if profit is significant (strategy-specific threshold)
                    min_best_profit = 150 if best_signal.strategy == "scalping" else 400  # Lower threshold for scalping
                    if best_signal.estimated_profit_inr >= min_best_profit:
                        additional_signals.append(best_signal)
                        self.logger.info(f"Additional high-profit signal: {symbol} - {best_signal.strategy} ({best_signal.timeframe}): Profit ₹{best_signal.estimated_profit_inr:.2f}")
            
            # Sort by profit and limit to top additional signals
            additional_signals.sort(key=lambda s: s.estimated_profit_inr, reverse=True)
            max_additional = min(len(additional_signals), 15)  # Top 15 additional signals
            final_additional = additional_signals[:max_additional]
            
            self.logger.info(f"=== ADDITIONAL SIGNAL GENERATION COMPLETE ===")
            self.logger.info(f"Generated {len(final_additional)} additional high-profit signals")
            
            return final_additional
            
        except Exception as e:
            self.logger.error(f"Error generating additional signals: {e}")
            return []

    def generate_entry_signals(self) -> List[EntrySignal]:
        try:
            self.logger.info("=== GENERATING ENTRY SIGNALS (HIGH VOLUME SYMBOLS, MIXED STRATEGY) ===")
            # Configurable top-mover scan params
            sg_cfg = self.config.get("trading", {}).get("signal_generation", {})
            self.logger.info("Fetching top movers with configurable volume filter...")
            min_volume = int(sg_cfg.get("min_volume_for_top_movers", 300000))
            topn = int(sg_cfg.get("top_movers_count", 40))
            top_movers = self.fetcher.fetch_top_movers(top_n=topn, min_volume=min_volume)
            if not top_movers:
                self.logger.warning("No top movers found")
                return []
            active_instruments_data = self.fetcher.fetch_active_instruments(margin_currency="USDT")
            if isinstance(active_instruments_data, list):
                active_symbols = set()
                for item in active_instruments_data:
                    if isinstance(item, dict):
                        symbol = item.get("symbol") or item.get("pair") or item.get("instrument_name")
                        if symbol:
                            active_symbols.add(symbol)
                    elif isinstance(item, str):
                        active_symbols.add(item)
            else:
                active_symbols = set(active_instruments_data) if isinstance(active_instruments_data, list) else set()
            filtered_movers = [(symbol, pct_change, volume) for symbol, pct_change, volume in top_movers if symbol in active_symbols]
            if not filtered_movers:
                self.logger.warning("No top movers are active instruments")
                return []
            self.logger.info(f"Filtered to {len(filtered_movers)} top movers that are active instruments")
            top_movers = filtered_movers
            self.logger.info(f"Found {len(top_movers)} top movers to analyze (min_volume={min_volume})")

            # --- Configurable: Filter by ATR% (max of 15m, 30m, 1h) and volume ---
            filtered_by_atr = []
            for symbol, pct_change, volume in top_movers:
                atr_pcts = []
                for tf in ["15m", "30m", "1h"]:
                    candles = self.fetcher.fetch_candlestick_data(symbol, tf, limit=50)
                    if not candles or len(candles) < 20:
                        continue
                    close_prices = [float(c["close"]) for c in candles[-20:]]
                    last_price = close_prices[-1]
                    atr = self.indicators.calculate_atr(candles)
                    if not atr or last_price == 0:
                        continue
                    atr_pct = (atr / last_price) * 100
                    atr_pcts.append(atr_pct)
                if not atr_pcts:
                    continue
                max_atr_pct = max(atr_pcts)
                min_vol_cfg = float(sg_cfg.get("min_volume_for_atr_filter", 300000))
                min_atr_pct_cfg = float(sg_cfg.get("min_atr_pct_filter", 0.6))
                if volume >= min_vol_cfg and max_atr_pct >= min_atr_pct_cfg:
                    filtered_by_atr.append((symbol, pct_change, volume, max_atr_pct))
            # Sort by ATR% * volume (composite score)
            filtered_by_atr.sort(key=lambda x: x[2] * x[3], reverse=True)
            # Only keep top N configurable
            top_filtered_count = int(sg_cfg.get("top_filtered_count", 15))
            top_filtered = filtered_by_atr[:top_filtered_count]
            self.logger.info(f"Selected {len(top_filtered)} symbols with high ATR% (max of 15m/30m/1h) and volume for signal generation.")
            # Use only these for signal generation
            top_movers = [(symbol, pct_change, volume) for symbol, pct_change, volume, atr_pct in top_filtered]
            
            all_signals = []
            swing_signals = []
            long_swing_signals = []
            scalping_signals = []
            profit_threshold = self.config.get("trading", {}).get("signal_generation", {}).get("profit_threshold", 300)
            min_candles = 100
            
            def process_symbol(args):
                symbol, pct_change, volume = args
                # Blacklist certain symbols
                blacklist = {"B-FARTCOIN_USDT", "B-FIDA_USDT", "B-API3_USDT"}
                if symbol in blacklist:
                    self.logger.info(f"Skipping blacklisted symbol: {symbol}")
                    return []
                candidate_signals = []
                direction = TradeDirection.LONG if pct_change > 0 else TradeDirection.SHORT
                # Strategy priority configurable; default keep scalping first
                strategy_order = sg_cfg.get("strategy_order", ["scalping", "swing", "long_swing"])
                for strategy in strategy_order:
                    config_timeframes = self.config.get("trading", {}).get("timeframes", {})
                    timeframes = config_timeframes.get(strategy, [])
                    if not timeframes:
                        continue
                    for timeframe in timeframes:
                        try:
                            min_candles_tf = 50 if strategy == "scalping" else min_candles
                            candles = self.fetcher.fetch_candlestick_data(symbol, timeframe, min_candles_tf)
                            if not candles or len(candles) < min_candles_tf:
                                self.logger.debug(f"Skipping {symbol} {timeframe}: insufficient data ({len(candles) if candles else 0} candles)")
                                continue
                            signal = self._generate_signal_for_timeframe(symbol, timeframe, strategy, direction)
                            if signal and signal.estimated_profit_inr:
                                candidate_signals.append(signal)
                                self.logger.debug(f"Candidate {strategy} {direction.value} signal for {symbol} {timeframe}: Profit ₹{signal.estimated_profit_inr:.2f}, Confidence {signal.confidence:.3f}")
                        except Exception as e:
                            self.logger.debug(f"Skipping {symbol} {timeframe}: {e}")
                            continue
                return candidate_signals

            # Use ThreadPoolExecutor for parallel processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(process_symbol, top_movers))
            # Flatten results
            for candidate_signals in results:
                swing_best = None
                long_swing_best = None
                scalping_best = None
                for s in candidate_signals:
                    if s.strategy == "swing" and (swing_best is None or s.estimated_profit_inr > swing_best.estimated_profit_inr):
                        swing_best = s
                    if s.strategy == "long_swing" and (long_swing_best is None or s.estimated_profit_inr > long_swing_best.estimated_profit_inr):
                        long_swing_best = s
                    if s.strategy == "scalping" and (scalping_best is None or s.estimated_profit_inr > scalping_best.estimated_profit_inr):
                        scalping_best = s
                if swing_best:
                    swing_signals.append(swing_best)
                if long_swing_best:
                    long_swing_signals.append(long_swing_best)
                if scalping_best:
                    scalping_signals.append(scalping_best)
                all_signals.extend(candidate_signals)
            # Sort by profit
            swing_signals.sort(key=lambda s: s.estimated_profit_inr, reverse=True)
            long_swing_signals.sort(key=lambda s: s.estimated_profit_inr, reverse=True)
            scalping_signals.sort(key=lambda s: s.estimated_profit_inr, reverse=True)
            all_signals.sort(key=lambda s: s.estimated_profit_inr, reverse=True)
            final_signals = []
            # PRIORITIZE SHORT-TERM SIGNALS: counts configurable
            max_scalp = int(sg_cfg.get("max_scalping_signals", 8))
            max_swing = int(sg_cfg.get("max_swing_signals", 6))
            max_long = int(sg_cfg.get("max_long_swing_signals", 1))
            final_signals.extend(scalping_signals[:max_scalp])
            final_signals.extend(swing_signals[:max_swing])
            final_signals.extend(long_swing_signals[:max_long])
            used = set((s.symbol, s.strategy, s.timeframe) for s in final_signals)
            for s in all_signals:
                key = (s.symbol, s.strategy, s.timeframe)
                max_final = int(sg_cfg.get("max_final_signals", 20))
                if key not in used and len(final_signals) < max_final:
                    final_signals.append(s)
                    used.add(key)
            final_signals.sort(key=lambda s: s.estimated_profit_inr, reverse=True)
            self.logger.info(f"Generated {len(final_signals)} profitable signals (PRIORITIZING scalping & swing over long_swing) from {len(top_movers)} top movers")
            for i, signal in enumerate(final_signals, 1):
                self.logger.info(f"{i}. {signal.symbol} - {signal.strategy} ({signal.timeframe}): Score {signal.score}, Normalized {signal.normalized_score:.1f}%, Confidence {signal.confidence:.3f}, Est. Profit ₹{signal.estimated_profit_inr:.2f}")
            return final_signals
        except Exception as e:
            self.logger.error(f"Error generating entry signals: {e}")
            return []

    def _generate_signal_for_timeframe(self, symbol: str, timeframe: str, strategy: str, direction: TradeDirection = None) -> Optional[EntrySignal]:
        """Generate signal for specific symbol and timeframe using the specified strategy and forced direction."""
        try:
            # Check if we already have an active trade for this symbol
            if self._has_active_trade(symbol, timeframe):
                self.logger.debug(f"Already have active trade for {symbol} {timeframe}")
                return None
            # Use the specified strategy instead of determining from timeframe
            if strategy == "scalping":
                from src.strategies.scalping import ScalpingStrategy
                strategy_instance = ScalpingStrategy(self.config.get("trading", {}).get("balance", 10000))
                signal_data = strategy_instance.generate_signal(symbol, timeframe)  # No direction param for scalping
            elif strategy == "swing":
                from src.strategies.swing import SwingStrategy
                strategy_instance = SwingStrategy(self.config.get("trading", {}).get("balance", 10000))
                signal_data = strategy_instance.generate_signal(symbol, timeframe, direction.value if direction else None)
            elif strategy == "long_swing":
                from src.strategies.long_swing import LongSwingStrategy
                strategy_instance = LongSwingStrategy(self.config.get("trading", {}).get("balance", 10000))
                signal_data = strategy_instance.generate_signal(symbol, timeframe, direction.value if direction else None)
            else:
                self.logger.error(f"Unknown strategy: {strategy}")
                return None
            
            # Generate signal using the strategy instance
            if not signal_data:
                return None
            
            # Convert signal data to EntrySignal
            try:
                # Always use take_profit = take_profit or exit_price for all strategies
                take_profit = signal_data.get("take_profit", signal_data.get("exit_price"))
                
                # Debug: Log available keys if take_profit is None
                if take_profit is None:
                    self.logger.error(f"Missing take_profit/exit_price in signal_data. Available keys: {list(signal_data.keys())}")
                    # Use tp1 as fallback
                    take_profit = signal_data.get("tp1")
                    if take_profit is None:
                        self.logger.error(f"Missing tp1 as well. Cannot create EntrySignal for {signal_data.get('symbol', 'unknown')}")
                        return None
                
                entry_signal = EntrySignal(
                    symbol=signal_data["symbol"],
                    timeframe=signal_data["timeframe"],
                    strategy=signal_data["strategy"],
                    direction=TradeDirection.LONG if signal_data["side"] == "long" else TradeDirection.SHORT,
                    entry_price=signal_data["entry_price"],
                    stop_loss=signal_data["stop_loss"],
                    take_profit=take_profit,
                    tp1=signal_data["tp1"],
                    tp2=signal_data["tp2"],
                    max_hold_time=signal_data["max_hold_time"],
                    score=signal_data["score"],
                    score_reasons=signal_data["score_reasons"],
                    indicators=signal_data["indicators"],
                    timestamp=datetime.fromisoformat(signal_data["timestamp"]),
                    estimated_profit_inr=signal_data["estimated_profit_inr"],
                    confidence=0.0,  # Will be calculated below
                    quality_score=0.0  # Will be calculated below
                )
                
                # Calculate confidence and quality score
                entry_signal.confidence = self._calculate_confidence(entry_signal.score, entry_signal.indicators, entry_signal.direction, strategy)
                entry_signal.quality_score = (entry_signal.score * 10) + (entry_signal.confidence * 100)
                
                # IMPROVED: Calculate normalized score for fair comparison across strategies
                max_possible_score = 7  # Maximum score for any strategy
                entry_signal.normalized_score = (entry_signal.score / max_possible_score) * 100
                
                # IMPROVED: Adjust estimated profit based on signal strength and market conditions
                base_profit = entry_signal.estimated_profit_inr
                
                # Adjust profit based on score (higher score = higher potential profit)
                score_multiplier = 1.0 + (entry_signal.score - 5) * 0.1  # 10% increase per score point above 5
                
                # Adjust based on RSI position
                rsi = entry_signal.indicators.get("rsi")
                rsi_multiplier = 1.0
                if rsi:
                    if entry_signal.direction == TradeDirection.LONG:
                        if 45 <= rsi <= 65:  # Optimal bullish RSI
                            rsi_multiplier = 1.15
                        elif rsi > 70:  # Overbought
                            rsi_multiplier = 0.85
                    else:  # SHORT
                        if 35 <= rsi <= 55:  # Optimal bearish RSI
                            rsi_multiplier = 1.15
                        elif rsi < 30:  # Oversold
                            rsi_multiplier = 0.85
                
                # Adjust based on MACD strength
                macd = entry_signal.indicators.get("macd")
                macd_multiplier = 1.0
                if macd:
                    if (entry_signal.direction == TradeDirection.LONG and macd > 0.001) or \
                       (entry_signal.direction == TradeDirection.SHORT and macd < -0.001):
                        macd_multiplier = 1.1
                    else:
                        macd_multiplier = 0.9
                
                # Adjust based on volume
                volume_ok = entry_signal.indicators.get("volume_ok", False)
                volume_multiplier = 1.05 if volume_ok else 0.95
                
                # Apply all multipliers
                final_profit = base_profit * score_multiplier * rsi_multiplier * macd_multiplier * volume_multiplier
                
                # Ensure minimum profit
                final_profit = max(final_profit, 20.0)  # Minimum ₹20 profit
                entry_signal.estimated_profit_inr = final_profit
                
                return entry_signal
                
            except Exception as e:
                self.logger.error(f"Error converting signal data to EntrySignal: {e}")
                return None
            
        except Exception as e:
            self.logger.error(f"Error generating {strategy} signal for {symbol} {timeframe}: {e}")
            return None

    def _calculate_indicators(self, candles: List[Dict], strategy: str) -> Optional[Dict]:
        """Calculate technical indicators for signal generation."""
        try:
            if not candles or len(candles) < 20:
                return None
            
            # Get current price from latest candle
            current_price = float(candles[-1]["close"])
            
            # Calculate basic indicators
            rsi = self.indicators.calculate_rsi(candles)
            ema_20 = self.indicators.calculate_ema(candles, 20)
            ema_50 = self.indicators.calculate_ema(candles, 50)
            macd = self.indicators.calculate_macd(candles)
            bb_upper, bb_lower = self.indicators.calculate_bollinger_bands(candles)
            stoch_rsi = self.indicators.calculate_stoch_rsi(candles)
            vwap_daily = self.indicators.calculate_vwap_daily(candles)
            
            # IMPROVED: Add ATR calculation for dynamic exit levels
            atr = self.indicators.calculate_atr(candles)
            
            # Calculate volume SMA
            volume_sma = self._calculate_volume_sma(candles, 20)
            
            # Validate indicators
            if any(x is None for x in [rsi, ema_20, ema_50, macd, bb_upper, bb_lower, stoch_rsi, vwap_daily]):
                self.logger.warning(f"Some indicators failed to calculate")
                return None
            
            # Extract StochRSI values
            stoch_k = stoch_rsi["%K"] if stoch_rsi else 50
            stoch_d = stoch_rsi["%D"] if stoch_rsi else 50
            
            # Calculate additional metrics
            bb_middle = (bb_upper + bb_lower) / 2 if bb_upper and bb_lower else current_price
            bb_width = ((bb_upper - bb_lower) / bb_middle) * 100 if bb_middle > 0 else 0
            
            # Calculate price position relative to indicators
            price_vs_ema20 = ((current_price - ema_20) / ema_20) * 100 if ema_20 else 0
            price_vs_ema50 = ((current_price - ema_50) / ema_50) * 100 if ema_50 else 0
            price_vs_vwap = ((current_price - vwap_daily) / vwap_daily) * 100 if vwap_daily else 0
            price_vs_bb_middle = ((current_price - bb_middle) / bb_middle) * 100 if bb_middle else 0
            
            return {
                "current_price": current_price,
                "rsi": rsi,
                "ema_20": ema_20,
                "ema_50": ema_50,
                "macd": macd,
                "bollinger_upper": bb_upper,
                "bollinger_lower": bb_lower,
                "bollinger_middle": bb_middle,
                "bollinger_width": bb_width,
                "vwap_daily": vwap_daily,
                "stoch_rsi_k": stoch_k,
                "stoch_rsi_d": stoch_d,
                "volume_sma": volume_sma,
                "atr": atr,  # IMPROVED: Include ATR for dynamic exits
                "price_vs_ema20": price_vs_ema20,
                "price_vs_ema50": price_vs_ema50,
                "price_vs_vwap": price_vs_vwap,
                "price_vs_bb_middle": price_vs_bb_middle,
                "candle_count": len(candles)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            return None

    def _determine_direction(self, indicators: Dict, strategy: str) -> Optional[TradeDirection]:
        """Determine trade direction based on indicators."""
        try:
            rsi = indicators.get("rsi")
            current_price = indicators.get("current_price")
            ema_20 = indicators.get("ema_20")
            ema_50 = indicators.get("ema_50")
            
            if not all([rsi, current_price, ema_20, ema_50]):
                return None
            
            # Scalping strategy logic
            if strategy == "scalping":
                macd = indicators.get("macd")
                vwap = indicators.get("vwap")
                stoch_k = indicators.get("stoch_rsi_k")
                stoch_d = indicators.get("stoch_rsi_d")
                
                # Long conditions
                long_conditions = [
                    current_price > ema_20,
                    ema_20 > ema_50,
                    rsi > 40 and rsi < 70,
                    macd > 0 if macd else False,
                    current_price > vwap if vwap else True,
                    stoch_k < 80 if stoch_k else True
                ]
                
                # Short conditions
                short_conditions = [
                    current_price < ema_20,
                    ema_20 < ema_50,
                    rsi > 30 and rsi < 60,
                    macd < 0 if macd else False,
                    current_price < vwap if vwap else True,
                    stoch_k > 20 if stoch_k else True
                ]
                
                if sum(long_conditions) >= 3:
                    return TradeDirection.LONG
                elif sum(short_conditions) >= 3:
                    return TradeDirection.SHORT
            
            # Swing strategy logic
            elif strategy == "swing":
                macd_histogram = indicators.get("macd_histogram")
                vwap = indicators.get("vwap")
                stoch_k = indicators.get("stoch_rsi_k")
                stoch_d = indicators.get("stoch_rsi_d")
                
                # Long conditions
                long_conditions = [
                    current_price > ema_20,
                    ema_20 > ema_50,
                    rsi > 35 and rsi < 65,
                    macd_histogram > 0 if macd_histogram else False,
                    current_price > vwap if vwap else True,
                    stoch_k < 85 if stoch_k else True
                ]
                
                # Short conditions
                short_conditions = [
                    current_price < ema_20,
                    ema_20 < ema_50,
                    rsi > 25 and rsi < 55,
                    macd_histogram < 0 if macd_histogram else False,
                    current_price < vwap if vwap else True,
                    stoch_k > 15 if stoch_k else True
                ]
                
                if sum(long_conditions) >= 3:
                    return TradeDirection.LONG
                elif sum(short_conditions) >= 3:
                    return TradeDirection.SHORT
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error determining direction: {e}")
            return None

    def _calculate_entry_parameters(self, candles: List[Dict], indicators: Dict, direction: TradeDirection, strategy: str) -> Optional[Dict]:
        """Calculate entry price, stop loss, and take profit levels with dynamic ATR-based targets."""
        try:
            current_price = indicators["current_price"]
            atr = indicators.get("atr", 0)
            
            # IMPROVED: Use ATR-based dynamic calculations instead of fixed percentages
            if atr and atr > 0:
                # Calculate ATR-based levels with IMPROVED multipliers for crypto volatility
                atr_multiplier_tp = 3.0 if strategy == "scalping" else 4.0  # Increased for better R:R
                atr_multiplier_sl = 2.0 if strategy == "scalping" else 2.5  # Increased to avoid premature hits
                
                if direction == TradeDirection.LONG:
                    stop_loss = current_price - (atr * atr_multiplier_sl)
                    take_profit = current_price + (atr * atr_multiplier_tp)
                    tp1 = current_price + (atr * atr_multiplier_tp * 0.7)  # 70% of target
                    tp2 = current_price + (atr * atr_multiplier_tp * 1.2)  # 120% of target
                else:  # SHORT
                    stop_loss = current_price + (atr * atr_multiplier_sl)
                    take_profit = current_price - (atr * atr_multiplier_tp)
                    tp1 = current_price - (atr * atr_multiplier_tp * 0.7)  # 70% of target
                    tp2 = current_price - (atr * atr_multiplier_tp * 1.2)  # 120% of target
            else:
                # Fallback to percentage-based calculations with IMPROVED percentages for crypto volatility
                # Strategy-specific parameters with REALISTIC percentages for crypto markets
                if strategy == "scalping":
                    # Scalping: Wider targets to account for crypto volatility
                    target_pct = 0.015  # 1.5% (increased from 0.8%)
                    sl_pct = 0.012      # 1.2% (increased from 0.5%)
                    max_hold_time = 30   # 30 minutes
                elif strategy == "swing":
                    # Swing: Moderate targets with crypto volatility consideration
                    target_pct = 0.025  # 2.5% (increased from 1.5%)
                    sl_pct = 0.018      # 1.8% (increased from 0.8%)
                    max_hold_time = 120  # 2 hours
                else:  # long_swing
                    # Long Swing: Larger targets for crypto markets
                    target_pct = 0.040  # 4.0% (increased from 2.5%)
                    sl_pct = 0.025      # 2.5% (increased from 1.2%)
                    max_hold_time = 480  # 8 hours
                
                # Calculate levels using percentage-based targets
                if direction == TradeDirection.LONG:
                    stop_loss = current_price * (1 - sl_pct)
                    take_profit = current_price * (1 + target_pct)
                    tp1 = current_price * (1 + target_pct * 0.7)  # 70% of target
                    tp2 = current_price * (1 + target_pct * 1.2)  # 120% of target
                else:  # SHORT
                    stop_loss = current_price * (1 + sl_pct)
                    take_profit = current_price * (1 - target_pct)
                    tp1 = current_price * (1 - target_pct * 0.7)  # 70% of target
                    tp2 = current_price * (1 - target_pct * 1.2)  # 120% of target
            
            # IMPROVED: Validate exit levels are reasonable for crypto markets
            # Ensure stop loss is not too close to entry (increased minimum distance)
            min_sl_distance = current_price * 0.008  # Minimum 0.8% distance (increased from 0.2%)
            if direction == TradeDirection.LONG:
                if (current_price - stop_loss) < min_sl_distance:
                    stop_loss = current_price - min_sl_distance
            else:
                if (stop_loss - current_price) < min_sl_distance:
                    stop_loss = current_price + min_sl_distance
            
            # Ensure take profit is not too far from entry (increased maximum distance)
            max_tp_distance = current_price * 0.08  # Maximum 8% distance (increased from 5%)
            if direction == TradeDirection.LONG:
                if (take_profit - current_price) > max_tp_distance:
                    take_profit = current_price + max_tp_distance
                    tp1 = current_price + (max_tp_distance * 0.7)
                    tp2 = current_price + (max_tp_distance * 1.2)
            else:
                if (current_price - take_profit) > max_tp_distance:
                    take_profit = current_price - max_tp_distance
                    tp1 = current_price - (max_tp_distance * 0.7)
                    tp2 = current_price - (max_tp_distance * 1.2)
            
            # IMPROVED: Ensure minimum risk-reward ratio of 1:1.2
            if direction == TradeDirection.LONG:
                risk = current_price - stop_loss
                reward = take_profit - current_price
            else:
                risk = stop_loss - current_price
                reward = current_price - take_profit
            
            min_rr_ratio = 1.2
            if reward < (risk * min_rr_ratio):
                # Adjust take profit to meet minimum R:R ratio
                if direction == TradeDirection.LONG:
                    take_profit = current_price + (risk * min_rr_ratio)
                    tp1 = current_price + (risk * min_rr_ratio * 0.7)
                    tp2 = current_price + (risk * min_rr_ratio * 1.2)
                else:
                    take_profit = current_price - (risk * min_rr_ratio)
                    tp1 = current_price - (risk * min_rr_ratio * 0.7)
                    tp2 = current_price - (risk * min_rr_ratio * 1.2)
            
            # Set max hold time based on strategy
            if strategy == "scalping":
                max_hold_time = 30   # 30 minutes
            elif strategy == "swing":
                max_hold_time = 120  # 2 hours
            else:  # long_swing
                max_hold_time = 480  # 8 hours
            
            # Calculate estimated profit in INR with 25x leverage
            usdt_inr_rate = self.fetcher.fetch_usdt_inr_rate() or 93.0
            position_size = 100  # USDT (example, adjust as needed)
            leverage = 25
            
            # Calculate profit based on tp1
            if direction == TradeDirection.LONG:
                price_diff = tp1 - current_price
            else:
                price_diff = current_price - tp1
            
            profit_usdt = (price_diff / current_price) * position_size * leverage
            profit_inr = profit_usdt * usdt_inr_rate
            
            # Calculate fees
            taker_fee_rate = self.config["trading"]["fees"]["taker"]
            taker_fee = profit_inr * taker_fee_rate
            net_profit = profit_inr - taker_fee
            
            return {
                "entry_price": current_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "tp1": tp1,
                "tp2": tp2,
                "max_hold_time": max_hold_time,
                "estimated_profit_inr": net_profit,
                "atr_used": atr > 0,
                "atr_value": atr,
                "exit_calculation_method": "atr_based" if atr > 0 else "percentage_based",
                "risk_reward_ratio": reward / risk if risk > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating entry parameters: {e}")
            return None

    def _calculate_signal_score(self, indicators: Dict, direction: TradeDirection, strategy: str) -> Tuple[int, List[str]]:
        """Calculate signal score and reasons."""
        score = 0
        reasons = []
        
        try:
            rsi = indicators.get("rsi")
            current_price = indicators.get("current_price")
            ema_20 = indicators.get("ema_20")
            ema_50 = indicators.get("ema_50")
            volume_ok = indicators.get("volume_ok", False)
            
            # Basic trend alignment
            if direction == TradeDirection.LONG:
                if current_price > ema_20:
                    score += 1
                    reasons.append("Price > EMA20")
                if ema_20 > ema_50:
                    score += 1
                    reasons.append("EMA20 > EMA50")
                if rsi and 40 <= rsi <= 70:
                    score += 1
                    reasons.append("RSI bullish zone (40-70)")
            else:  # SHORT
                if current_price < ema_20:
                    score += 1
                    reasons.append("Price < EMA20")
                if ema_20 < ema_50:
                    score += 1
                    reasons.append("EMA20 < EMA50")
                if rsi and 30 <= rsi <= 60:
                    score += 1
                    reasons.append("RSI bearish zone (30-60)")
            
            # Strategy-specific scoring
            if strategy == "scalping":
                macd = indicators.get("macd")
                vwap = indicators.get("vwap")
                stoch_k = indicators.get("stoch_rsi_k")
                
                if direction == TradeDirection.LONG:
                    if macd and macd > 0:
                        score += 1
                        reasons.append("MACD positive")
                    if vwap and current_price > vwap:
                        score += 1
                        reasons.append("Price > VWAP")
                    if stoch_k and stoch_k < 80:
                        score += 1
                        reasons.append("StochRSI not overbought")
                else:  # SHORT
                    if macd and macd < 0:
                        score += 1
                        reasons.append("MACD negative")
                    if vwap and current_price < vwap:
                        score += 1
                        reasons.append("Price < VWAP")
                    if stoch_k and stoch_k > 20:
                        score += 1
                        reasons.append("StochRSI not oversold")
            
            elif strategy == "swing":
                macd_histogram = indicators.get("macd_histogram")
                vwap = indicators.get("vwap")
                stoch_k = indicators.get("stoch_rsi_k")
                
                if direction == TradeDirection.LONG:
                    if macd_histogram and macd_histogram > 0:
                        score += 1
                        reasons.append("MACD histogram positive")
                    if vwap and current_price > vwap:
                        score += 1
                        reasons.append("Price > VWAP")
                    if stoch_k and stoch_k < 85:
                        score += 1
                        reasons.append("StochRSI not overbought")
                else:  # SHORT
                    if macd_histogram and macd_histogram < 0:
                        score += 1
                        reasons.append("MACD histogram negative")
                    if vwap and current_price < vwap:
                        score += 1
                        reasons.append("Price < VWAP")
                    if stoch_k and stoch_k > 15:
                        score += 1
                        reasons.append("StochRSI not oversold")
            
            # Volume confirmation
            if volume_ok:
                score += 1
                reasons.append("Volume > 20-SMA")
            
            return score, reasons
            
        except Exception as e:
            self.logger.error(f"Error calculating signal score: {e}")
            return 0, []

    def _calculate_volume_sma(self, candles: List[Dict], period: int) -> Optional[float]:
        """Calculate volume SMA."""
        try:
            volumes = [float(candle["volume"]) for candle in candles[-period:]]
            return sum(volumes) / len(volumes) if volumes else None
        except Exception as e:
            self.logger.error(f"Error calculating volume SMA: {e}")
            return None

    def _has_active_trade(self, symbol: str, timeframe: str = None) -> bool:
        """Check if there is an active trade for the symbol or symbol+timeframe."""
        if timeframe:
            trade_key = f"{symbol}_{timeframe}"
            return trade_key in self.active_trades
        else:
            # Check for any active trade for this symbol (any timeframe)
            return any(key.startswith(f"{symbol}_") for key in self.active_trades)

    def _validate_entry_conditions(self, signal: EntrySignal, live_price: float) -> Tuple[bool, str]:
        """
        Validate entry conditions in real-time before trade execution.
        
        Returns:
            Tuple[bool, str]: (is_valid, reason)
        """
        try:
            # 1. Check slippage tolerance based on strategy
            price_difference = abs(live_price - signal.entry_price)
            price_difference_pct = (price_difference / signal.entry_price) * 100
            
            # Strategy-specific slippage tolerance
            if signal.strategy == "scalping":
                max_slippage_pct = 0.5  # 0.5% for scalping (tight)
            elif signal.strategy == "swing":
                max_slippage_pct = 1.0  # 1.0% for swing
            elif signal.strategy == "long_swing":
                max_slippage_pct = 1.5  # 1.5% for long swing
            else:
                max_slippage_pct = 1.0  # Default for other strategies
            
            if price_difference_pct > max_slippage_pct:
                return False, f"Slippage {price_difference_pct:.2f}% exceeds maximum {max_slippage_pct}% for {signal.strategy}"
            
            # 2. Check spread based on strategy
            order_book = self.fetcher.fetch_order_book(signal.symbol)
            if order_book and order_book.get("spread"):
                spread_pct = (order_book["spread"] / live_price) * 100
                
                # Strategy-specific spread tolerance
                if signal.strategy == "scalping":
                    max_spread_pct = 0.3  # 0.3% for scalping
                elif signal.strategy == "swing":
                    max_spread_pct = 0.5  # 0.5% for swing
                elif signal.strategy == "long_swing":
                    max_spread_pct = 0.8  # 0.8% for long swing
                else:
                    max_spread_pct = 0.5  # Default for other strategies
                
                if spread_pct > max_spread_pct:
                    return False, f"Spread {spread_pct:.2f}% exceeds maximum {max_spread_pct}% for {signal.strategy}"
            
            # 3. Check current market conditions
            candles = self.fetcher.fetch_candlestick_data(signal.symbol, signal.timeframe, 50)
            if candles and len(candles) >= 20:
                current_indicators = self._calculate_indicators(candles, signal.strategy)
                if current_indicators:
                    # Validate trend alignment based on strategy
                    if signal.direction == TradeDirection.LONG:
                        # For long signals, check bullish conditions
                        if (current_indicators.get("ema_20", 0) > 0 and 
                            live_price < current_indicators["ema_20"]):
                            return False, "Live price below EMA20, long signal invalid"
                        
                        # Strategy-specific RSI validation
                        if signal.strategy == "scalping" and current_indicators.get("rsi", 50) > 70:
                            return False, "RSI overbought (>70), long scalping signal invalid"
                        elif signal.strategy in ["swing", "long_swing"] and current_indicators.get("rsi", 50) > 75:
                            return False, "RSI overbought (>75), long signal invalid"
                            
                    else:  # SHORT
                        # For short signals, check bearish conditions
                        if (current_indicators.get("ema_20", 0) > 0 and 
                            live_price > current_indicators["ema_20"]):
                            return False, "Live price above EMA20, short signal invalid"
                        
                        # Strategy-specific RSI validation
                        if signal.strategy == "scalping" and current_indicators.get("rsi", 50) < 30:
                            return False, "RSI oversold (<30), short scalping signal invalid"
                        elif signal.strategy in ["swing", "long_swing"] and current_indicators.get("rsi", 50) < 25:
                            return False, "RSI oversold (<25), short signal invalid"
            
            # 4. Check volume conditions
            if candles and len(candles) >= 20:
                current_volume = float(candles[-1]["volume"]) if candles else 0
                volume_sma = self._calculate_volume_sma(candles, 20)
                
                # Strategy-specific volume requirements
                if signal.strategy == "scalping":
                    min_volume_ratio = 0.5  # 50% of SMA for scalping
                elif signal.strategy == "swing":
                    min_volume_ratio = 0.7  # 70% of SMA for swing
                elif signal.strategy == "long_swing":
                    min_volume_ratio = 0.8  # 80% of SMA for long swing
                else:
                    min_volume_ratio = 0.6  # Default for other strategies
                
                if volume_sma and current_volume < volume_sma * min_volume_ratio:
                    return False, f"Volume {current_volume:.0f} below {min_volume_ratio*100:.0f}% of SMA {volume_sma:.0f} for {signal.strategy}"
            
            # 5. Check if we already have an active trade for this symbol
            if self._has_active_trade(signal.symbol, signal.timeframe):
                return False, f"Already have active trade for {signal.symbol} {signal.timeframe}"
            
            # 6. CRITICAL: Ensure estimated profit is sufficient to avoid losses
            if signal.estimated_profit_inr < 500:  # Minimum ₹500 profit
                return False, f"Estimated profit ₹{signal.estimated_profit_inr:.2f} below minimum ₹500 threshold"
            
            # 7. CRITICAL: Check if current price is too close to stop loss (risk of immediate loss)
            if signal.direction == TradeDirection.LONG:
                stop_distance_pct = ((live_price - signal.stop_loss) / live_price) * 100
                if stop_distance_pct < 1.0:  # Less than 1% from stop loss
                    return False, f"Price too close to stop loss: {stop_distance_pct:.2f}% distance"
            else:  # SHORT
                stop_distance_pct = ((signal.stop_loss - live_price) / live_price) * 100
                if stop_distance_pct < 1.0:  # Less than 1% from stop loss
                    return False, f"Price too close to stop loss: {stop_distance_pct:.2f}% distance"
            
            return True, "All validation checks passed"
            
        except Exception as e:
            self.logger.error(f"Error validating entry conditions: {e}")
            return False, f"Validation error: {e}"

    def enter_trade(self, signal: EntrySignal) -> Optional[str]:
        self.logger.info(f"[DEBUG] enter_trade called for {signal.symbol} {signal.timeframe} (score: {signal.score}, confidence: {signal.confidence:.3f})")
        self.logger.info(f"[DEBUG] Signal attributes: entry_price={signal.entry_price}, stop_loss={signal.stop_loss}, take_profit={signal.take_profit}, tp1={signal.tp1}, tp2={signal.tp2}, max_hold_time={signal.max_hold_time}")
        try:
            # Validate signal attributes
            if (signal.entry_price is None or signal.stop_loss is None or signal.take_profit is None or
                signal.entry_price <= 0 or signal.stop_loss <= 0 or signal.take_profit <= 0 or
                signal.tp1 is None or signal.tp2 is None or signal.max_hold_time is None):
                self.logger.error(f"[ERROR] Invalid signal attributes: {signal}")
                return None

            # IMPROVED: Enhanced entry price validation with slippage protection
            market_data = self.fetcher.fetch_market_data(signal.symbol)
            if not market_data or market_data.get('last_price', 0) <= 0:
                self.logger.error(f"[ERROR] Could not fetch live market price for {signal.symbol}")
                return None
            
            live_entry_price = float(market_data['last_price'])
            live_entry_time = datetime.now()
            
            # IMPROVED: Use comprehensive validation function
            is_valid, validation_reason = self._validate_entry_conditions(signal, live_entry_price)
            if not is_valid:
                self.logger.warning(f"[ENTRY REJECTION] {signal.symbol}: {validation_reason}")
                return None
            
            # IMPROVED: Calculate slippage for logging
            signal_entry_price = signal.entry_price
            price_difference = abs(live_entry_price - signal_entry_price)
            price_difference_pct = (price_difference / signal_entry_price) * 100
            
            # Check if we've reached max trades
            if len(self.active_trades) >= self.max_trades:
                self.logger.info(f"Maximum trades reached ({self.max_trades}), cannot enter new trade")
                return None

            trade_key = f"{signal.symbol}_{signal.timeframe}"
            trade_id = f"{signal.symbol}_{signal.timeframe}_{int(time.time())}"
            
            # IMPROVED: Use live entry price for trade execution
            # But keep signal entry price for reference
            active_trade = ActiveTrade(
                trade_id=trade_id,
                symbol=signal.symbol,
                timeframe=signal.timeframe,
                strategy=signal.strategy,
                direction=signal.direction,
                entry_price=live_entry_price,  # Use live price for actual entry
                entry_time=live_entry_time,
                current_price=live_entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                tp1=signal.tp1,
                tp2=signal.tp2,
                max_hold_time=signal.max_hold_time,
                current_pnl=0.0,
                max_pnl=0.0,
                min_pnl=0.0,
                candle_count=0,
                last_update=live_entry_time,
                breakeven_stop=live_entry_price,
                trailing_stop=signal.stop_loss,
                profit_lock_level=live_entry_price,
                entry_indicators=signal.indicators,
                current_indicators=signal.indicators,
                signal_strength_history=[signal.score]
            )
            self.active_trades[trade_key] = active_trade
            self.trades_entered += 1
            
            self.logger.info(f"=== TRADE ENTERED ===")
            self.logger.info(f"Trade ID: {trade_id}")
            self.logger.info(f"Symbol: {signal.symbol}")
            self.logger.info(f"Timeframe: {signal.timeframe}")
            self.logger.info(f"Strategy: {signal.strategy}")
            self.logger.info(f"Direction: {signal.direction.value.upper()}")
            self.logger.info(f"Signal Entry Price: {signal_entry_price:.6f}")
            self.logger.info(f"Live Entry Price: {live_entry_price:.6f}")
            self.logger.info(f"Price Difference: {price_difference_pct:.2f}%")
            self.logger.info(f"Validation: {validation_reason}")
            self.logger.info(f"Entry Time: {live_entry_time.strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"Stop Loss: {signal.stop_loss:.6f}")
            self.logger.info(f"Take Profit: {signal.take_profit:.6f}")
            self.logger.info(f"TP1: {signal.tp1:.6f}")
            self.logger.info(f"TP2: {signal.tp2:.6f}")
            self.logger.info(f"Max Hold Time: {signal.max_hold_time} minutes")
            self.logger.info(f"Signal Score: {signal.score}/7")
            self.logger.info(f"Quality Score: {signal.quality_score:.1f}")
            self.logger.info(f"Score Reasons: {', '.join(signal.score_reasons)}")
            self.logger.info(f"Estimated Profit (INR): ₹{signal.estimated_profit_inr:.2f}")
            self.logger.info(f"Active Trades: {len(self.active_trades)}/{self.max_trades}")
            self.logger.info(f"=====================")
            return trade_id
        except Exception as e:
            import traceback
            self.logger.error(f"Error entering trade: {e}\n{traceback.format_exc()}")
            return None

    def start_monitoring(self):
        """Start monitoring active trades."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            self.logger.info("Started enhanced trade monitoring")

    def stop_monitoring(self):
        """Stop monitoring active trades."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        self.logger.info("Stopped enhanced trade monitoring")

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self._process_active_trades()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)

    def _process_active_trades(self):
        """
        Process and monitor all active trades for exit conditions using DynamicExitMonitor.
        """
        self.logger.info(f"[DEBUG] _process_active_trades called. Active trades: {len(self.active_trades)}")
        if not self.active_trades:
            return
        self.logger.info(f"Processing {len(self.active_trades)} active trades for dynamic exit monitoring...")
        for trade_key, trade in list(self.active_trades.items()):
            try:
                # Update current price and indicators
                self._update_trade_data(trade)
                # Log current trade status
                pnl_status = "[PROFIT]" if trade.current_pnl > 0 else "[LOSS]" if trade.current_pnl < 0 else "[BREAKEVEN]"
                self.logger.info(f"  {pnl_status} {trade.symbol} {trade.timeframe}: {trade.direction.value} Entry: {trade.entry_price:.6f} Current: {trade.current_price:.6f} PnL: {trade.current_pnl:.2f}%")
                
                # 🛡️ PROFIT PROTECTION: No exits until 5% profit
                if trade.current_pnl < 5.0:
                    self.logger.debug(f"🛡️ {trade.symbol}: Profit {trade.current_pnl:.2f}% < 5%, PROTECTING TRADE")
                    self._update_dynamic_levels(trade)
                    continue
                
                # Only check exit conditions if profit >= 5%
                self.logger.info(f"✅ {trade.symbol}: Profit {trade.current_pnl:.2f}% >= 5%, checking exit conditions")
                
                # Improved exit logic - ONLY for trades with >= 7% profit
                exit_reason = None
                
                # Check if take profit hit
                if trade.direction == TradeDirection.LONG and trade.current_price >= trade.take_profit:
                    exit_reason = ExitReason.TP_HIT
                elif trade.direction == TradeDirection.SHORT and trade.current_price <= trade.take_profit:
                    exit_reason = ExitReason.TP_HIT
                
                # Check if stop loss hit
                elif trade.direction == TradeDirection.LONG and trade.current_price <= trade.stop_loss:
                    exit_reason = ExitReason.SL_HIT
                elif trade.direction == TradeDirection.SHORT and trade.current_price >= trade.stop_loss:
                    exit_reason = ExitReason.SL_HIT
                
                # Check max hold time
                elif (datetime.now() - trade.entry_time).total_seconds() / 60 >= trade.max_hold_time:
                    exit_reason = ExitReason.MAX_HOLD_TIME
                
                # Only check dynamic exits if no basic exit conditions met
                if not exit_reason:
                    exit_reason = self.exit_monitor.should_exit(trade)
                if exit_reason:
                    self.logger.info(f"🚨 DYNAMIC EXIT TRIGGERED for {trade.symbol} ({trade.timeframe}): {exit_reason.value}")
                    self.logger.info(f"   Final PnL: {trade.current_pnl:.2f}% | Max PnL: {trade.max_pnl:.2f}% | Min PnL: {trade.min_pnl:.2f}%")
                    self._execute_exit(trade, exit_reason)
                else:
                    self._update_dynamic_levels(trade)
            except Exception as e:
                self.logger.error(f"Error processing trade {trade.trade_id}: {e}")
        self.logger.info(f"Dynamic exit monitoring completed for {len(self.active_trades)} trades")

    def _update_trade_data(self, trade: ActiveTrade):
        """Update trade with current market data."""
        try:
            # Fetch current price
            market_data = self.fetcher.fetch_market_data(trade.symbol)
            if market_data:
                trade.current_price = market_data["last_price"]
                trade.current_volume = market_data["volume"]
            
            # Fetch recent candles for indicators
            candles = self.fetcher.fetch_candlestick_data(trade.symbol, trade.timeframe, 50)
            if candles:
                trade.current_indicators = self._calculate_indicators(candles, trade.strategy)
                trade.candle_count += 1
            
            # Update PnL
            if trade.direction == TradeDirection.LONG:
                trade.current_pnl = ((trade.current_price - trade.entry_price) / trade.entry_price) * 100
            else:
                trade.current_pnl = ((trade.entry_price - trade.current_price) / trade.entry_price) * 100
            
            trade.max_pnl = max(trade.max_pnl, trade.current_pnl)
            trade.min_pnl = min(trade.min_pnl, trade.current_pnl)
            trade.last_update = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error updating trade data: {e}")

    def _update_dynamic_levels(self, trade: ActiveTrade):
        """Update dynamic stop loss and take profit levels."""
        try:
            # Move to breakeven after threshold
            if abs(trade.current_pnl) >= self.exit_thresholds["breakeven_threshold"]:
                if trade.direction == TradeDirection.LONG:
                    trade.breakeven_stop = trade.entry_price + (trade.entry_price * 0.001)  # Entry + 0.1%
                else:
                    trade.breakeven_stop = trade.entry_price - (trade.entry_price * 0.001)  # Entry - 0.1%
                
                if trade.stop_loss != trade.breakeven_stop:
                    old_sl = trade.stop_loss
                    trade.stop_loss = trade.breakeven_stop
                    self.logger.info(f"[BREAKEVEN] {trade.symbol} SL moved from {old_sl:.6f} to {trade.stop_loss:.6f} (0.4% profit reached)")
            
            # Activate trailing stop after threshold
            if abs(trade.current_pnl) >= self.exit_thresholds["trailing_activation"]:
                atr = trade.current_indicators.get("atr", trade.entry_price * 0.01)
                
                if trade.direction == TradeDirection.LONG:
                    new_trailing_stop = trade.current_price - (atr * 2)
                    if new_trailing_stop > trade.trailing_stop:
                        old_sl = trade.stop_loss
                        trade.trailing_stop = new_trailing_stop
                        trade.stop_loss = max(trade.stop_loss, trade.trailing_stop)
                        if old_sl != trade.stop_loss:
                            self.logger.info(f"[TRAILING] {trade.symbol} SL moved from {old_sl:.6f} to {trade.stop_loss:.6f} (0.6% profit reached)")
                else:
                    new_trailing_stop = trade.current_price + (atr * 2)
                    if new_trailing_stop < trade.trailing_stop or trade.trailing_stop == trade.entry_price:
                        old_sl = trade.stop_loss
                        trade.trailing_stop = new_trailing_stop
                        trade.stop_loss = min(trade.stop_loss, trade.trailing_stop)
                        if old_sl != trade.stop_loss:
                            self.logger.info(f"[TRAILING] {trade.symbol} SL moved from {old_sl:.6f} to {trade.stop_loss:.6f} (0.6% profit reached)")
            
            # Profit locking
            if abs(trade.current_pnl) >= self.exit_thresholds["profit_lock_threshold"]:
                if trade.direction == TradeDirection.LONG:
                    trade.profit_lock_level = trade.entry_price + (trade.entry_price * 0.005)  # Entry + 0.5%
                else:
                    trade.profit_lock_level = trade.entry_price - (trade.entry_price * 0.005)  # Entry - 0.5%
                
                if trade.stop_loss != trade.profit_lock_level:
                    old_sl = trade.stop_loss
                    trade.stop_loss = trade.profit_lock_level
                    self.logger.info(f"[PROFIT LOCK] {trade.symbol} SL moved from {old_sl:.6f} to {trade.stop_loss:.6f} (0.8% profit reached)")
                    
        except Exception as e:
            self.logger.error(f"Error updating dynamic levels: {e}")

    def _execute_exit(self, trade: ActiveTrade, reason: ExitReason):
        """Execute trade exit."""
        try:
            final_pnl = trade.current_pnl
            hold_time_minutes = (datetime.now() - trade.entry_time).total_seconds() / 60
            exit_time = datetime.now()
            # Calculate profit in INR
            usdt_inr_rate = 93  # You can fetch live rate if needed
            position_size = 100  # USDT (example, adjust as needed)
            price_diff = abs(trade.current_price - trade.entry_price)
            profit_usdt = (price_diff / trade.entry_price) * position_size
            profit_inr = profit_usdt * usdt_inr_rate
            self.logger.info(f"=== TRADE EXITED ===")
            self.logger.info(f"Trade ID: {trade.trade_id}")
            self.logger.info(f"Symbol: {trade.symbol}")
            self.logger.info(f"Timeframe: {trade.timeframe}")
            self.logger.info(f"Strategy: {trade.strategy}")
            self.logger.info(f"Direction: {trade.direction.value.upper()}")
            self.logger.info(f"Entry Price: {trade.entry_price:.6f}")
            self.logger.info(f"Exit Price: {trade.current_price:.6f}")
            self.logger.info(f"Entry Time: {trade.entry_time.strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"Exit Time: {exit_time.strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"Exit Reason: {reason.value.upper()}")
            self.logger.info(f"Final PnL: {final_pnl:.2f}%")
            self.logger.info(f"Profit (INR): ₹{profit_inr:.2f}")
            self.logger.info(f"Max PnL: {trade.max_pnl:.2f}%")
            self.logger.info(f"Min PnL: {trade.min_pnl:.2f}%")
            self.logger.info(f"Hold Time: {hold_time_minutes:.1f} minutes")
            self.logger.info(f"==================")
            # Add to trade history
            trade_record = {
                "trade_id": trade.trade_id,
                "symbol": trade.symbol,
                "timeframe": trade.timeframe,
                "strategy": trade.strategy,
                "direction": trade.direction.value,
                "entry_price": trade.entry_price,
                "exit_price": trade.current_price,
                "entry_time": trade.entry_time.isoformat(),
                "exit_time": exit_time.isoformat(),
                "exit_reason": reason.value,
                "final_pnl": final_pnl,
                "profit_inr": profit_inr,
                "max_pnl": trade.max_pnl,
                "min_pnl": trade.min_pnl,
                "hold_time_minutes": hold_time_minutes
            }
            self.trade_history.append(trade_record)
        except Exception as e:
            self.logger.error(f"Error executing exit: {e}")

    def get_active_trades(self) -> Dict[str, ActiveTrade]:
        """Get all active trades."""
        return self.active_trades

    def get_trade_history(self) -> List[Dict]:
        """Get trade history."""
        return self.trade_history

    def get_performance_summary(self) -> Dict:
        """Get performance summary."""
        if not self.trade_history:
            return {
                "total_trades": 0, 
                "winning_trades": 0,
                "win_rate": 0, 
                "avg_pnl": 0,
                "total_pnl": 0
            }
        
        total_trades = len(self.trade_history)
        winning_trades = len([t for t in self.trade_history if t["final_pnl"] > 0])
        win_rate = (winning_trades / total_trades) * 100
        avg_pnl = sum(t["final_pnl"] for t in self.trade_history) / total_trades
        total_pnl = sum(t["final_pnl"] for t in self.trade_history)
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "win_rate": win_rate,
            "avg_pnl": avg_pnl,
            "total_pnl": total_pnl
        }

    def _calculate_confidence(self, score: int, indicators: Dict, direction: TradeDirection, strategy: str) -> float:
        """
        Calculate sophisticated confidence based on score, trend strength, volume confirmation, and indicator convergence.
        """
        try:
            # Base confidence from score
            base_confidence = self._score_to_confidence(score)
            
            # Trend strength multiplier
            trend_strength = 1.0
            ema_20 = indicators.get("ema_20")
            ema_50 = indicators.get("ema_50")
            current_price = indicators.get("current_price")
            
            if all([ema_20, ema_50, current_price]):
                if direction == TradeDirection.LONG:
                    # Strong uptrend: price > EMA20 > EMA50 with good separation
                    ema_separation = (ema_20 - ema_50) / ema_50
                    price_ema_separation = (current_price - ema_20) / ema_20
                    if ema_separation > 0.005 and price_ema_separation > 0.002:  # 0.5% and 0.2%
                        trend_strength = 1.2
                    elif ema_separation > 0.002 and price_ema_separation > 0.001:  # 0.2% and 0.1%
                        trend_strength = 1.1
                else:  # SHORT
                    # Strong downtrend: price < EMA20 < EMA50 with good separation
                    ema_separation = (ema_50 - ema_20) / ema_20
                    price_ema_separation = (ema_20 - current_price) / ema_20
                    if ema_separation > 0.005 and price_ema_separation > 0.002:
                        trend_strength = 1.2
                    elif ema_separation > 0.002 and price_ema_separation > 0.001:
                        trend_strength = 1.1
            
            # Volume confirmation multiplier
            volume_multiplier = 1.0
            volume_ok = indicators.get("volume_ok", False)
            if volume_ok:
                volume_multiplier = 1.15  # 15% boost for volume confirmation
            
            # RSI zone multiplier
            rsi_multiplier = 1.0
            rsi = indicators.get("rsi")
            if rsi:
                if direction == TradeDirection.LONG and 45 <= rsi <= 65:
                    rsi_multiplier = 1.1  # Sweet spot for longs
                elif direction == TradeDirection.SHORT and 35 <= rsi <= 55:
                    rsi_multiplier = 1.1  # Sweet spot for shorts
                elif direction == TradeDirection.LONG and rsi > 70:
                    rsi_multiplier = 0.8  # Overbought penalty
                elif direction == TradeDirection.SHORT and rsi < 30:
                    rsi_multiplier = 0.8  # Oversold penalty
            
            # MACD confirmation multiplier
            macd_multiplier = 1.0
            if strategy == "scalping":
                macd = indicators.get("macd")
                if macd:
                    if (direction == TradeDirection.LONG and macd > 0) or (direction == TradeDirection.SHORT and macd < 0):
                        macd_multiplier = 1.1
                    else:
                        macd_multiplier = 0.9
            else:  # swing
                macd_histogram = indicators.get("macd_histogram")
                if macd_histogram:
                    if (direction == TradeDirection.LONG and macd_histogram > 0) or (direction == TradeDirection.SHORT and macd_histogram < 0):
                        macd_multiplier = 1.1
                    else:
                        macd_multiplier = 0.9
            
            # VWAP alignment multiplier
            vwap_multiplier = 1.0
            vwap = indicators.get("vwap")
            if vwap and current_price:
                if (direction == TradeDirection.LONG and current_price > vwap) or (direction == TradeDirection.SHORT and current_price < vwap):
                    vwap_multiplier = 1.05
                else:
                    vwap_multiplier = 0.95
            
            # Stoch RSI multiplier
            stoch_multiplier = 1.0
            stoch_k = indicators.get("stoch_rsi_k")
            if stoch_k:
                if direction == TradeDirection.LONG and stoch_k < 80:
                    stoch_multiplier = 1.05
                elif direction == TradeDirection.SHORT and stoch_k > 20:
                    stoch_multiplier = 1.05
                elif direction == TradeDirection.LONG and stoch_k > 90:
                    stoch_multiplier = 0.8
                elif direction == TradeDirection.SHORT and stoch_k < 10:
                    stoch_multiplier = 0.8
            
            # Calculate final confidence with all multipliers
            final_confidence = base_confidence * trend_strength * volume_multiplier * rsi_multiplier * macd_multiplier * vwap_multiplier * stoch_multiplier
            
            # Ensure confidence stays within reasonable bounds
            final_confidence = max(0.01, min(0.8, final_confidence))
            
            # Ensure high-quality signals (score >= 6) never get below 0.3
            if score >= 6:
                final_confidence = max(0.3, final_confidence)
            
            return round(final_confidence, 3)
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return self._score_to_confidence(score)  # Fallback to basic calculation

    def _score_to_confidence(self, score: int, max_score: int = 7) -> float:
        """Map score to confidence with fallback logic. Ensure high-quality signals never get zero confidence."""
        if score >= 6:
            return 0.3 + 0.1 * (score - 6)  # 0.3 for 6, 0.4 for 7
        elif score >= 4:
            return 0.1 + 0.05 * (score - 4)  # 0.1-0.2 for 4-5
        elif score > 0:
            return 0.05 * score
        else:
            return 0.01  # fallback for neutral/missing but not negative 

    def _generate_multi_tf_signal(self, symbol: str) -> list:
        """
        Generate multi-timeframe signal using 1H trend (EMA9/21, MACD, ADX) and 5M confirmation.
        Returns a list of EntrySignal objects for scalping, swing, and long swing if conditions align.
        """
        from datetime import datetime
        signals = []
        try:
            # Fetch candles
            candles_1h = self.fetcher.fetch_candlestick_data(symbol, "1h", 100)
            candles_5m = self.fetcher.fetch_candlestick_data(symbol, "5m", 100)
            if not candles_1h or not candles_5m:
                return []

            # Calculate indicators for 1H
            ema9_1h = self.indicators.calculate_ema(candles_1h, 9)
            ema21_1h = self.indicators.calculate_ema(candles_1h, 21)
            macd_1h = self.indicators.calculate_macd(candles_1h)
            adx_1h = self.indicators.calculate_adx(candles_1h)
            # Calculate indicators for 5M
            ema9_5m = self.indicators.calculate_ema(candles_5m, 9)
            ema21_5m = self.indicators.calculate_ema(candles_5m, 21)
            macd_5m = self.indicators.calculate_macd(candles_5m)
            adx_5m = self.indicators.calculate_adx(candles_5m)
            current_price = float(candles_5m[-1]["close"])

            # Determine 1H trend
            trend_1h = None
            if (
                ema9_1h is not None and ema21_1h is not None and macd_1h is not None and adx_1h is not None
                and adx_1h["adx"] > 20
            ):
                if ema9_1h > ema21_1h and macd_1h > 0 and adx_1h["di_plus"] > adx_1h["di_minus"]:
                    trend_1h = "bullish"
                elif ema9_1h < ema21_1h and macd_1h < 0 and adx_1h["di_minus"] > adx_1h["di_plus"]:
                    trend_1h = "bearish"
                else:
                    trend_1h = "sideways"
            else:
                return []

            # Entry logic
            direction = None
            if (
                trend_1h == "bullish"
                and ema9_5m is not None and ema21_5m is not None and macd_5m is not None and adx_5m is not None
                and ema9_5m > ema21_5m and macd_5m > 0 and adx_5m["adx"] > 20
            ):
                direction = TradeDirection.LONG
            elif (
                trend_1h == "bearish"
                and ema9_5m is not None and ema21_5m is not None and macd_5m is not None and adx_5m is not None
                and ema9_5m < ema21_5m and macd_5m < 0 and adx_5m["adx"] > 20
            ):
                direction = TradeDirection.SHORT
            else:
                return []  # Wait or skip

            # Set targets and SLs - GUARANTEED MINIMUM 5% PROFIT TARGETS
            targets = {
                "scalping": (0.08, 0.02, 30),         # 8% target, 2% SL, 30 minutes
                "swing": (0.12, 0.03, 120),           # 12% target, 3% SL, 2 hours
                "long_swing": (0.20, 0.05, 480)       # 20% target, 5% SL, 8 hours
            }
            for style, (target_pct, sl_pct, max_hold) in targets.items():
                if direction == TradeDirection.LONG:
                    target = current_price * (1 + target_pct)
                    stop_loss = current_price * (1 - sl_pct)
                else:
                    target = current_price * (1 - target_pct)
                    stop_loss = current_price * (1 + sl_pct)
                signal = EntrySignal(
                    symbol=symbol,
                    timeframe="5m",
                    strategy=style,
                    direction=direction,
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=target,  # This is already correct since we're creating it directly
                    tp1=target,
                    tp2=target,
                    max_hold_time=max_hold,
                    score=0,  # You can enhance this with a scoring logic
                    score_reasons=[
                        f"1H trend: {trend_1h}",
                        f"5M EMA9/21: {ema9_5m:.2f}/{ema21_5m:.2f}",
                        f"5M MACD: {macd_5m:.4f}",
                        f"5M ADX: {adx_5m['adx']:.2f}"
                    ],
                    indicators={
                        "symbol": symbol,
                        "ema9_1h": ema9_1h,
                        "ema21_1h": ema21_1h,
                        "macd_1h": macd_1h,
                        "adx_1h": adx_1h,
                        "ema9_5m": ema9_5m,
                        "ema21_5m": ema21_5m,
                        "macd_5m": macd_5m,
                        "adx_5m": adx_5m,
                        "current_price": current_price
                    },
                    timestamp=datetime.now(),
                    estimated_profit_inr=0,  # Can be calculated if needed
                    confidence=0.5  # Can be enhanced
                )
                signals.append(signal)
        except Exception as e:
            self.logger.error(f"Error in multi-timeframe signal for {symbol}: {e}")
        return signals 