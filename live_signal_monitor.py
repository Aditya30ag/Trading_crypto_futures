#!/usr/bin/env python3
"""
Live Signal Monitor for CoinDCX App Testing
Monitors enhanced signals in real-time and provides entry/exit guidance
"""

import json
import time
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from src.strategies.simple_enhanced_generator import SimpleEnhancedGenerator
from src.data.fetcher import CoinDCXFetcher
from src.utils.logger import setup_logger


class LiveSignalMonitor:
    """Real-time monitor for enhanced signals on CoinDCX."""
    
    def __init__(self, balance: float = 10000):
        self.generator = SimpleEnhancedGenerator(balance)
        self.fetcher = CoinDCXFetcher()
        self.logger = setup_logger()
        self.active_signals = {}
        self.signal_history = []
        
    def start_monitoring(self, duration_hours: int = 24):
        """Start real-time monitoring of enhanced signals."""
        self.logger.info(f"Starting live signal monitoring for {duration_hours} hours")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)
        
        print("\n" + "="*80)
        print("🚀 LIVE SIGNAL MONITOR FOR COINDCX APP")
        print("="*80)
        print("Monitoring enhanced signals in real-time...")
        print("Press Ctrl+C to stop monitoring")
        print("="*80)
        
        try:
            while datetime.now() < end_time:
                # Generate fresh signals
                signals = self.generator.generate_signals()
                
                # Update active signals
                self._update_active_signals(signals)
                
                # Display current status
                self._display_status()
                
                # Check for signal completions
                self._check_signal_completions()
                
                # Wait 5 minutes before next update
                time.sleep(300)  # 5 minutes
                
        except KeyboardInterrupt:
            print("\n\n⏹️  Monitoring stopped by user")
            self._save_final_report()
        except Exception as e:
            self.logger.error(f"Error in live monitoring: {e}")
            print(f"\n❌ Error: {e}")
    
    def _update_active_signals(self, new_signals: List[Dict]):
        """Update active signals with fresh data."""
        current_time = datetime.now()
        
        for signal in new_signals:
            signal_id = f"{signal['symbol']}_{signal['side']}_{signal['timestamp']}"
            
            # Get current market price
            try:
                market_data = self.fetcher.fetch_market_data(signal['symbol'])
                if market_data and 'last_price' in market_data:
                    current_price = market_data['last_price']
                    
                    # Calculate current P&L
                    if signal['side'] == 'long':
                        if current_price >= signal['take_profit']:
                            pnl_percent = ((current_price - signal['entry_price']) / signal['entry_price']) * 100
                            status = "TP_HIT"
                        elif current_price <= signal['stop_loss']:
                            pnl_percent = ((current_price - signal['entry_price']) / signal['entry_price']) * 100
                            status = "SL_HIT"
                        else:
                            pnl_percent = ((current_price - signal['entry_price']) / signal['entry_price']) * 100
                            status = "ACTIVE"
                    else:  # short
                        if current_price <= signal['take_profit']:
                            pnl_percent = ((signal['entry_price'] - current_price) / signal['entry_price']) * 100
                            status = "TP_HIT"
                        elif current_price >= signal['stop_loss']:
                            pnl_percent = ((signal['entry_price'] - current_price) / signal['entry_price']) * 100
                            status = "SL_HIT"
                        else:
                            pnl_percent = ((signal['entry_price'] - current_price) / signal['entry_price']) * 100
                            status = "ACTIVE"
                    
                    # Update signal with current data
                    signal.update({
                        'current_price': current_price,
                        'pnl_percent': pnl_percent,
                        'status': status,
                        'last_update': current_time.isoformat()
                    })
                    
                    self.active_signals[signal_id] = signal
                    
            except Exception as e:
                self.logger.error(f"Error updating signal {signal['symbol']}: {e}")
    
    def _display_status(self):
        """Display current signal status."""
        current_time = datetime.now()
        
        print(f"\n📊 LIVE STATUS - {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 60)
        
        if not self.active_signals:
            print("⏳ No active signals at the moment...")
            return
        
        for signal_id, signal in self.active_signals.items():
            symbol = signal['symbol']
            side = signal['side']
            entry = signal['entry_price']
            current = signal.get('current_price', entry)
            pnl = signal.get('pnl_percent', 0)
            status = signal.get('status', 'ACTIVE')
            
            # Status emoji
            status_emoji = {
                'ACTIVE': '🟡',
                'TP_HIT': '🟢',
                'SL_HIT': '🔴'
            }.get(status, '🟡')
            
            print(f"\n{status_emoji} {symbol} ({side.upper()})")
            print(f"   Entry: ${entry:.6f}")
            print(f"   Current: ${current:.6f}")
            print(f"   P&L: {pnl:+.2f}%")
            print(f"   Status: {status}")
            
            # Show distance to targets
            if side == 'long':
                tp_distance = ((signal['take_profit'] - current) / current) * 100
                sl_distance = ((current - signal['stop_loss']) / current) * 100
            else:
                tp_distance = ((current - signal['take_profit']) / current) * 100
                sl_distance = ((signal['stop_loss'] - current) / current) * 100
            
            print(f"   Distance to TP: {tp_distance:.2f}%")
            print(f"   Distance to SL: {sl_distance:.2f}%")
            
            # Trading advice
            if status == 'ACTIVE':
                if pnl > 0:
                    print(f"   💡 HOLD - Signal is profitable")
                else:
                    print(f"   ⚠️  WATCH - Signal is at loss")
            elif status == 'TP_HIT':
                print(f"   ✅ TAKE PROFIT HIT! Close position")
            elif status == 'SL_HIT':
                print(f"   ❌ STOP LOSS HIT! Close position")
        
        print("\n" + "-" * 60)
    
    def _check_signal_completions(self):
        """Check for completed signals and move to history."""
        completed_signals = []
        
        for signal_id, signal in list(self.active_signals.items()):
            if signal.get('status') in ['TP_HIT', 'SL_HIT']:
                # Calculate final P&L
                if signal['side'] == 'long':
                    if signal['status'] == 'TP_HIT':
                        final_pnl = ((signal['take_profit'] - signal['entry_price']) / signal['entry_price']) * 100
                    else:  # SL_HIT
                        final_pnl = ((signal['stop_loss'] - signal['entry_price']) / signal['entry_price']) * 100
                else:  # short
                    if signal['status'] == 'TP_HIT':
                        final_pnl = ((signal['entry_price'] - signal['take_profit']) / signal['entry_price']) * 100
                    else:  # SL_HIT
                        final_pnl = ((signal['entry_price'] - signal['stop_loss']) / signal['entry_price']) * 100
                
                signal['final_pnl'] = final_pnl
                signal['completion_time'] = datetime.now().isoformat()
                
                self.signal_history.append(signal)
                completed_signals.append(signal)
                del self.active_signals[signal_id]
                
                # Display completion message
                if signal['status'] == 'TP_HIT':
                    print(f"\n🎉 TAKE PROFIT HIT! {signal['symbol']} {signal['side'].upper()}")
                    print(f"   Final P&L: {final_pnl:+.2f}%")
                else:
                    print(f"\n💔 STOP LOSS HIT! {signal['symbol']} {signal['side'].upper()}")
                    print(f"   Final P&L: {final_pnl:+.2f}%")
        
        return completed_signals
    
    def _save_final_report(self):
        """Save final monitoring report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"live_monitoring_report_{timestamp}.json"
        
        report = {
            'monitoring_start': datetime.now().isoformat(),
            'active_signals': self.active_signals,
            'signal_history': self.signal_history,
            'summary': self._calculate_summary()
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📊 Final report saved to: {filename}")
        self._display_final_summary()
    
    def _calculate_summary(self):
        """Calculate monitoring summary."""
        total_signals = len(self.signal_history)
        tp_hits = len([s for s in self.signal_history if s['status'] == 'TP_HIT'])
        sl_hits = len([s for s in self.signal_history if s['status'] == 'SL_HIT'])
        
        win_rate = (tp_hits / total_signals * 100) if total_signals > 0 else 0
        
        total_pnl = sum(s.get('final_pnl', 0) for s in self.signal_history)
        avg_pnl = total_pnl / total_signals if total_signals > 0 else 0
        
        return {
            'total_signals': total_signals,
            'tp_hits': tp_hits,
            'sl_hits': sl_hits,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl
        }
    
    def _display_final_summary(self):
        """Display final monitoring summary."""
        summary = self._calculate_summary()
        
        print("\n" + "="*80)
        print("📊 FINAL MONITORING SUMMARY")
        print("="*80)
        print(f"Total Signals Monitored: {summary['total_signals']}")
        print(f"Take Profit Hits: {summary['tp_hits']}")
        print(f"Stop Loss Hits: {summary['sl_hits']}")
        print(f"Win Rate: {summary['win_rate']:.1f}%")
        print(f"Total P&L: {summary['total_pnl']:+.2f}%")
        print(f"Average P&L per Signal: {summary['avg_pnl']:+.2f}%")
        print("="*80)


def main():
    """Main function to start live monitoring."""
    print("🚀 Starting Live Signal Monitor for CoinDCX App Testing")
    print("This will monitor enhanced signals in real-time")
    print("Use the signals to trade manually on CoinDCX app")
    print("\nPress Enter to start monitoring...")
    input()
    
    monitor = LiveSignalMonitor(balance=10000)
    monitor.start_monitoring(duration_hours=24)


if __name__ == "__main__":
    main() 