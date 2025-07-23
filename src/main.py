from src.trading.executor import TradeExecutor
import json

def main():
    executor = TradeExecutor()
    signals = executor.run()
    print(json.dumps(signals, indent=2))

if __name__ == "__main__":
    main()