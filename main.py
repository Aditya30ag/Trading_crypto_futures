#!/usr/bin/env python3
"""
Crypto Futures Trading Bot
Main entry point for the trading bot application
"""

# Initialize logger first, before any other imports
from src.utils.logger import setup_logger
logger = setup_logger()

# Now import other modules
from src.trading.executor import TradeExecutor
import json
import yaml
import argparse
from flask import Flask, jsonify, send_from_directory, request
import threading
import os
from src.data.fetcher import CoinDCXFetcher
import time
import signal
from datetime import datetime, timedelta

def get_top_signals(signals, top_n=10):
    # Sort by score, then estimated profit
    signals.sort(key=lambda s: (s.get('score', 0), s.get('estimated_profit_inr', 0)), reverse=True)
    return signals[:top_n]

def append_signals_continuously(config, interval_minutes=5, top_n=10):
    executor = TradeExecutor(config)
    logger.info("Continuous signal appending started.")
    running = True
    def handle_sigint(sig, frame):
        nonlocal running
        running = False
        logger.info("Graceful shutdown requested. Exiting loop...")
    signal.signal(signal.SIGINT, handle_sigint)
    while running:
        logger.info("Generating trading signals...")
        signals = executor.run()
        top_new_signals = get_top_signals(signals, top_n=top_n)
        try:
            with open('latest_signals.json', 'r') as f:
                existing_signals = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_signals = []
        now = datetime.now()
        # Keep only signals from last 24 hours
        filtered_signals = [
            s for s in existing_signals
            if 'timestamp' in s and datetime.fromisoformat(s['timestamp'].split('+')[0]) > now - timedelta(hours=24)
        ]
        # Add only new unique top signals
        for signal in top_new_signals:
            if not any(
                s['symbol'] == signal['symbol'] and
                s['strategy'] == signal['strategy'] and
                s['timestamp'] == signal['timestamp']
                for s in filtered_signals
            ):
                filtered_signals.append(signal)
        with open('latest_signals.json', 'w') as f:
            json.dump(filtered_signals, f, indent=2, default=str)
        logger.info(f"Appended {len(top_new_signals)} top signals. Total signals in file: {len(filtered_signals)}")
        for signal in top_new_signals:
            logger.info(f"APPEND: {signal['symbol']} {signal['side'].upper()} {signal['strategy']} - Score: {signal['score']}")
        for _ in range(interval_minutes * 6):
            if not running:
                break
            time.sleep(10)
    logger.info("Continuous signal appending stopped.")

def main():
    try:
        logger.info("=== Starting Crypto Futures Trading Bot ===")
        
        # Load configuration
        with open("config/config.yaml", "r") as file:
            config = yaml.safe_load(file)
        
        logger.info(f"Configuration loaded successfully")
        logger.info(f"Trading instruments: {len(config['trading']['instruments'])}")
        logger.info(f"Initial balance: {config['trading']['initial_balance']} INR")
        
        # Initialize trade executor
        executor = TradeExecutor(config)
        logger.info("Trade executor initialized")
        
        # Generate signals
        logger.info("Generating trading signals...")
        signals = executor.run()
        
        # Save signals to file
        if signals:
            with open("latest_signals.json", "w") as f:
                json.dump(signals, f, indent=2, default=str)
            logger.info(f"Generated {len(signals)} signals, saved to latest_signals.json")
            
            # Log each signal
            for signal in signals:
                if signal.get("score", 0) >= 5:
                    logger.info(f"SIGNAL: {signal['symbol']} {signal['side'].upper()} {signal['strategy']} - Score: {signal['score']}")
                else:
                    logger.debug(f"Signal: {signal['symbol']} {signal['side']} {signal['strategy']} - Score: {signal['score']}")
        else:
            logger.info("No signals generated")
        
        logger.info("=== Trading Bot execution completed ===")
        
        return executor
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


def run_api_server(executor=None):
    app = Flask(__name__)

    @app.route('/api/signals')
    def get_signals():
        if os.path.exists('latest_signals.json'):
            with open('latest_signals.json') as f:
                data = json.load(f)
            return jsonify(data)
        return jsonify([])

    @app.route('/api/balance')
    def get_balance():
        fetcher = CoinDCXFetcher()
        balance = fetcher.fetch_account_balance()
        return jsonify(balance)
    
    @app.route('/api/logs')
    def get_logs():
        log_path = os.path.join('logs', 'trading_bot.log')
        if not os.path.exists(log_path):
            return jsonify([])
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # Only keep lines with both 'INFO' and 'Generated signal'
        signal_lines = [line.strip() for line in lines if 'INFO' in line and 'Generated signal' in line]
        return jsonify(signal_lines[-100:])  # last 100 signal logs

    # New monitoring endpoints
    @app.route('/api/monitoring/summary')
    def get_monitoring_summary():
        if executor:
            summary = executor.get_monitoring_summary()
            return jsonify(summary)
        return jsonify({"error": "Executor not available"})

    @app.route('/api/signals/high-profit')
    def get_high_profit_signals():
        if executor:
            high_profit_signals = executor.signal_monitor.get_high_profit_signals()
            return jsonify(high_profit_signals)
        return jsonify([])

    @app.route('/api/signals/priority-summary')
    def get_signal_priority_summary():
        if executor:
            priority_summary = executor.signal_monitor.get_signal_priority_summary()
            return jsonify(priority_summary)
        return jsonify({"error": "Executor not available"})

    @app.route('/api/monitoring/active-signals')
    def get_active_signals():
        if executor:
            active_signals = executor.get_active_signals()
            # Convert datetime objects to strings for JSON serialization
            serializable_signals = {}
            for signal_id, signal_data in active_signals.items():
                serializable_data = {}
                for key, value in signal_data.items():
                    if isinstance(value, dict):
                        serializable_data[key] = value
                    elif hasattr(value, 'isoformat'):
                        serializable_data[key] = value.isoformat()
                    else:
                        serializable_data[key] = value
                serializable_signals[signal_id] = serializable_data
            return jsonify(serializable_signals)
        return jsonify({})

    @app.route('/api/monitoring/suggestions/<signal_id>')
    def get_signal_suggestions(signal_id):
        if executor:
            suggestions = executor.get_signal_suggestions(signal_id)
            return jsonify(suggestions)
        return jsonify([])

    @app.route('/api/monitoring/exit-signal', methods=['POST'])
    def exit_signal():
        if executor:
            data = request.get_json()
            signal_id = data.get('signal_id')
            exit_price = data.get('exit_price')
            reason = data.get('reason', 'manual')
            
            if signal_id:
                success = executor.manual_exit_signal(signal_id, exit_price, reason)
                return jsonify({"success": success, "signal_id": signal_id})
            else:
                return jsonify({"error": "signal_id is required"})
        return jsonify({"error": "Executor not available"})

    @app.route('/api/monitoring/status')
    def get_monitoring_status():
        if executor:
            status = {
                "monitoring_active": executor.monitoring_active,
                "active_signals_count": len(executor.get_active_signals()),
                "last_update": time.time()
            }
            return jsonify(status)
        return jsonify({"error": "Executor not available"})

    @app.route('/')
    def index():
        return 'API Futures Backend Running'

    app.run(host='0.0.0.0', port=8000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--serve', action='store_true', help='Run API server for frontend')
    parser.add_argument('--monitor', action='store_true', help='Run monitoring dashboard')
    parser.add_argument('--test', action='store_true', help='Run monitoring test')
    parser.add_argument('--continuous', action='store_true', help='Continuously append top signals to latest_signals.json')
    parser.add_argument('--interval', type=int, default=5, help='Interval in minutes for continuous mode')
    parser.add_argument('--topn', type=int, default=10, help='Number of top signals to append each cycle')
    args = parser.parse_args()
    
    if args.test:
        from test_monitoring import main as test_main
        test_main()
    elif args.monitor:
        from monitor_dashboard import main as dashboard_main
        dashboard_main()
    elif args.continuous:
        with open("config/config.yaml", "r") as file:
            config = yaml.safe_load(file)
        append_signals_continuously(config, interval_minutes=args.interval, top_n=args.topn)
    else:
        executor = main()
        if args.serve:
            run_api_server(executor)