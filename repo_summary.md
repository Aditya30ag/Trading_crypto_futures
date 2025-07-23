
+# Crypto Futures Trading Bot – Repository Summary
+
+## High-Level Overview
+This repository implements a **crypto futures trading bot** with a web-based dashboard. It features automated signal generation, trade execution, monitoring, and a React-based frontend for visualization and control.
+
+---
+
+## Main Components & Flow
+
+### 1. Backend (Python)
+#### Entry Point: `main.py`
+- **Purpose:** Orchestrates the bot, runs the API server, and manages signal generation and monitoring.
+- **Flow:**
+  1. **Configuration** is loaded from `config/config.yaml`.
+  2. **TradeExecutor** is initialized (core class for running strategies and managing trades).
+  3. **Signal Generation:** Generates trading signals using strategies and saves them to `latest_signals.json`.
+  4. **API Server:** Exposes REST endpoints (using Flask) for:
+     - `/api/signals` – Get latest signals
+     - `/api/balance` – Get current balance
+     - `/api/logs` – Get logs
+     - `/api/monitoring/*` – Monitoring and suggestions
+     - `/` – Serves the frontend
+
+#### Core Logic: `src/`
+- **`trading/executor.py`**: The `TradeExecutor` class
+  - Runs signal generation (via `EnhancedSignalGenerator`)
+  - Manages monitoring, trade execution, and balance
+  - Handles writing signals to file and API endpoints
+- **`trading/enhanced_signal_generator.py`**: The heart of signal generation
+  - Implements advanced logic for generating entry signals, scoring, and dynamic exit management
+  - Uses technical indicators, market data, and strategy rules
+- **`strategies/strategy_manager.py`**: Manages multiple trading strategies
+  - Loads and runs strategies like scalping, swing, long swing, and trend
+  - Selects top instruments based on volatility and volume
+- **`data/fetcher.py`**: Fetches real-time and historical market data from CoinDCX API
+  - Used by strategies and signal generator for analysis
+- **`utils/logger.py`**: Sets up logging for the entire backend
+
+#### Configuration: `config/config.yaml`
+- Contains API keys, trading parameters (instruments, timeframes, leverage, risk, etc.), and logging settings.
+
+---
+
+### 2. Frontend (React + Vite)
+#### Location: `frontend/`
+- **Purpose:** Provides a live dashboard for monitoring signals, balance, logs, and (future) settings.
+- **Key Files:**
+  - `src/App.jsx`: Main React component
+    - Fetches signals, balance, and logs from backend API
+    - Displays dashboard stats, trading signals, top volatile instruments, and logs
+    - Allows filtering/searching signals and navigation between pages
+  - `src/main.jsx`: React entry point
+- **Flow:**
+  1. On load, fetches signals and balance from backend every few seconds.
+  2. Displays live stats, signal table, and logs.
+  3. Sidebar navigation for Dashboard, Balance, Instruments, Logs, and Settings (some are placeholders).
+
+---
+
+### 3. How Everything Connects
+- **Signal Generation:** 
+  - `main.py` (or a monitoring script) triggers `TradeExecutor.run()`, which uses `EnhancedSignalGenerator` to analyze market data and generate signals.
+  - Signals are scored, filtered, and saved to `latest_signals.json`.
+- **API Server:**
+  - Flask server in `main.py` exposes endpoints for the frontend to fetch signals, balance, and logs.
+- **Frontend:**
+  - React app polls the backend for live data and displays it in a user-friendly dashboard.
+- **Configuration:**
+  - All trading logic is parameterized via `config/config.yaml` for easy tuning.
+
+---
+
+## Directory Structure (Key Parts)
+```
+/main.py                # Entry point, API server, orchestrator
+/src/
+  trading/
+    executor.py         # TradeExecutor: runs signals, manages trades
+    enhanced_signal_generator.py # Advanced signal generation logic
+  strategies/
+    strategy_manager.py # Loads and runs multiple strategies
+    scalping.py, swing.py, etc. # Individual strategy logic
+  data/
+    fetcher.py          # Fetches market data from CoinDCX
+  utils/
+    logger.py           # Logging setup
+/config/
+  config.yaml           # All trading and API configuration
+/frontend/
+  src/
+    App.jsx, main.jsx   # React frontend for dashboard
+```
+
+---
+
+## Typical Flow
+
+1. **Start the bot** (`main.py`):
+   - Loads config, initializes executor, generates signals.
+2. **Signal Generation:**
+   - Fetches market data, runs strategies, scores signals, saves to file.
+3. **API Server:**
+   - Serves signals, balance, logs, and monitoring data to frontend.
+4. **Frontend:**
+   - Fetches and displays live trading data, signals, and stats.
+
+---
+
+## What Does What (Summary Table)
+
+| Component/File                        | Responsibility                                      |
+|---------------------------------------|-----------------------------------------------------|
+| `main.py`                            | Entry point, API server, orchestrates everything    |
+| `src/trading/executor.py`             | Runs trading logic, manages signals/trades          |
+| `src/trading/enhanced_signal_generator.py` | Generates and manages trading signals         |
+| `src/strategies/strategy_manager.py`  | Loads and runs multiple trading strategies          |
+| `src/data/fetcher.py`                 | Fetches market and historical data                  |
+| `config/config.yaml`                  | All configuration (API, trading, logging)           |
+| `frontend/src/App.jsx`                | Main React dashboard, fetches and displays data     |
+
+---
+
+## How to Explain to a Developer
+
+- **Backend**: Python, Flask, modular, config-driven, fetches data, generates signals, exposes REST API.
+- **Frontend**: React, fetches from backend, live dashboard, user-friendly.
+- **Extensible**: Add new strategies in `src/strategies/`, tune config in YAML, extend frontend as needed.