import requests
import yaml
import time
from datetime import datetime, timezone, timedelta
from src.utils.logger import setup_logger
import socket
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class CoinDCXFetcher:
    def __init__(self):
        with open("config/config.yaml", "r") as file:
            self.config = yaml.safe_load(file)
        self.public_base_url = self.config["api"]["public_base_url"]
        self.api_base_url = self.config["api"]["api_base_url"]
        self.logger = setup_logger()
        
        # Configure external APIs
        self.external_apis = self.config.get("external_apis", {})
        self.cryptocompare_enabled = self.external_apis.get("cryptocompare", {}).get("enabled", False)
        self.cryptocompare_config = self.external_apis.get("cryptocompare", {})
        
        # Create a session with better connection handling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Configure session for better reliability
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            max_retries=3
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
        # Initialize CryptoCompare error tracking to reduce spam
        self._cryptocompare_failed_symbols = set()
        self._cryptocompare_error_count = 0
        self._max_cryptocompare_errors = 10  # Stop trying after 10 errors
        
        self.logger.info("CoinDCXFetcher initialized with public_base_url: %s, api_base_url: %s", 
                         self.public_base_url, self.api_base_url)
        self.logger.info("CryptoCompare API enabled: %s", self.cryptocompare_enabled)

    def fetch_market_data(self, symbol, retries=3):
        """Fetch real-time market data for a futures symbol with retries."""
        for attempt in range(retries):
            try:
                # Test DNS resolution first
                try:
                    socket.gethostbyname('public.coindcx.com')
                except socket.gaierror:
                    self.logger.error(f"DNS resolution failed for public.coindcx.com on attempt {attempt + 1}")
                    if attempt < retries - 1:
                        time.sleep(2)
                    continue
                
                response = self.session.get(
                    f"{self.public_base_url}/market_data/v3/current_prices/futures/rt", 
                    timeout=10,  # Increased timeout
                    verify=False  # Disable SSL verification if needed
                )
                response.raise_for_status()
                data = response.json()
                if symbol in data.get("prices", {}):
                    price_data = data["prices"][symbol]
                    return {
                        "last_price": float(price_data.get("ls", 0)),
                        "volume": float(price_data.get("v", 0)),
                        "change_24h": float(price_data.get("pc", 0))
                    }
                self.logger.error(f"Symbol {symbol} not found in market data")
                return None
            except requests.exceptions.ConnectionError as e:
                self.logger.error(f"Attempt {attempt + 1}/{retries} - Connection error for {symbol}: {e}")
                if attempt < retries - 1:
                    time.sleep(3)  # Longer delay for connection errors
                continue
            except requests.exceptions.Timeout as e:
                self.logger.error(f"Attempt {attempt + 1}/{retries} - Timeout for {symbol}: {e}")
                if attempt < retries - 1:
                    time.sleep(2)
                continue
            except Exception as e:
                self.logger.error(f"Attempt {attempt + 1}/{retries} - Error fetching market data for {symbol}: {e}")
                if attempt < retries - 1:
                    time.sleep(1)
                continue
        
        # Fallback: Return cached data or default values if API is completely down
        self.logger.warning(f"API unavailable for {symbol}, using fallback data")
        return {
            "last_price": 0.0,
            "volume": 0.0,
            "change_24h": 0.0
        }

    def fetch_candlestick_data(self, symbol, timeframe, limit=100, retries=3):
        """Fetch historical candlestick data with retries."""
        for attempt in range(retries):
            try:
                # Test DNS resolution first
                try:
                    socket.gethostbyname('public.coindcx.com')
                except socket.gaierror:
                    self.logger.error(f"DNS resolution failed for public.coindcx.com on attempt {attempt + 1}")
                    if attempt < retries - 1:
                        time.sleep(2)
                    continue
                
                # Map timeframe to supported resolutions
                time_map = {"1m": "1", "5m": "5", "15m": "15", "30m": "30", "1h": "60", "1d": "1D"}
                resolution = time_map.get(timeframe, timeframe)
                if resolution not in ["1", "5", "15", "30", "60", "1D"]:
                    self.logger.warning(f"Unsupported resolution: {resolution} for {symbol}, skipping")
                    return []
                
                to_time = int(time.time()) 
                if resolution == "1D":
                    from_time = to_time - (180 * 24 * 60 * 60)  # 180 days for 1d
                else:
                    from_time = to_time - (30 * 24 * 60 * 60)   # 30 days for others
                
                response = self.session.get(
                    f"{self.public_base_url}/market_data/candlesticks",
                    params={
                        "pair": symbol,
                        "from": from_time,
                        "to": to_time,
                        "resolution": resolution,
                        "pcode": "f"
                    },
                    timeout=15,  # Increased timeout
                    verify=False  # Disable SSL verification if needed
                )
                response.raise_for_status()
                data = response.json()
                
                # Handle different response formats
                if data.get("s") == "ok" and data.get("data"):
                    return data["data"]
                elif data.get("s") == "no_data":
                    self.logger.warning(f"No data available for {symbol} on {timeframe} timeframe")
                    return []
                elif data.get("s") == "error":
                    error_msg = data.get("message", "Unknown error")
                    self.logger.error(f"API error for {symbol}: {error_msg}")
                    return []
                else:
                    self.logger.warning(f"Unexpected response format for {symbol}: {data}")
                    return []
                    
            except requests.exceptions.ConnectionError as e:
                self.logger.error(f"Connection error on attempt {attempt + 1}/{retries} for {symbol}: {e}")
                if attempt < retries - 1:
                    time.sleep(3)  # Longer delay for connection errors
                continue
            except requests.exceptions.Timeout as e:
                self.logger.warning(f"Timeout on attempt {attempt + 1}/{retries} for {symbol}: {e}")
                if attempt < retries - 1:
                    time.sleep(2)  # Longer delay for timeouts
                continue
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Request error on attempt {attempt + 1}/{retries} for {symbol}: {e}")
                if attempt < retries - 1:
                    time.sleep(1)
                continue
            except Exception as e:
                self.logger.error(f"Unexpected error on attempt {attempt + 1}/{retries} for {symbol}: {e}")
                if attempt < retries - 1:
                    time.sleep(1)
                continue
                
        self.logger.error(f"Failed to fetch candlestick data for {symbol} after {retries} attempts")
        return []

    def fetch_trade_history(self, symbol, retries=3):
        """Fetch real-time trade history for a futures symbol with retries."""
        for attempt in range(retries):
            try:
                headers = {
                    "X-API-KEY": self.config["api"]["api_key"],
                    "X-API-SECRET": self.config["api"]["api_secret"]
                }
                response = requests.get(
                    f"{self.api_base_url}/exchange/v1/derivatives/futures/data/trades?pair={symbol}",
                    headers=headers,
                    timeout=5
                )
                response.raise_for_status()
                data = response.json()
                if data and isinstance(data, list):
                    return data
                self.logger.error(f"No trade history data for {symbol} in response: {data}")
                return []
            except Exception as e:
                self.logger.error(f"Attempt {attempt + 1}/{retries} - Error fetching trade history for {symbol}: {e}")
                if attempt < retries - 1:
                    time.sleep(1)
                continue
        self.logger.error(f"Failed to fetch trade history for {symbol} after {retries} attempts")
        return []

    def fetch_active_instruments(self, margin_currency="USDT", retries=3):
        """Fetch active futures instruments with retries."""
        for attempt in range(retries):
            try:
                headers = {
                    "X-API-KEY": self.config["api"]["api_key"],
                    "X-API-SECRET": self.config["api"]["api_secret"]
                }
                response = requests.get(
                    f"{self.api_base_url}/exchange/v1/derivatives/futures/data/active_instruments",
                    params={"margin_currency_short_name[]": margin_currency},
                    headers=headers,
                    timeout=5
                )
                response.raise_for_status()
                data = response.json()
                return data
            except Exception as e:
                self.logger.error(f"Attempt {attempt + 1}/{retries} - Error fetching active instruments: {e}")
                if attempt < retries - 1:
                    time.sleep(1)
                continue
        self.logger.error("Failed to fetch active instruments, using config fallback")
        return self.config["trading"]["instruments"]

    def fetch_usdt_inr_rate(self, retries=3):
        """Fetch live USDT/INR rate with retries."""
        for attempt in range(retries):
            try:
                response = requests.get(f"{self.public_base_url}/market_data/v3/current_prices/futures/rt", timeout=5)
                response.raise_for_status()
                data = response.json()
                return 93  # Fallback rate based on recent web data
            except Exception as e:
                self.logger.error(f"Attempt {attempt + 1}/{retries} - Error fetching USDT/INR rate: {e}")
                if attempt < retries - 1:
                    time.sleep(1)
                continue
        return 93  # Fallback rate

    def test_connection(self):
        """Test if the API is accessible."""
        try:
            # Test DNS resolution
            socket.gethostbyname('public.coindcx.com')
            
            # Test API endpoint
            response = self.session.get(
                f"{self.public_base_url}/market_data/v3/current_prices/futures/rt",
                timeout=5,
                verify=False
            )
            response.raise_for_status()
            return True
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False

    def get_ist_timestamp(self):
        """Get current timestamp in IST."""
        ist = timezone(timedelta(hours=5, minutes=30))
        return datetime.now(ist).isoformat()

    def fetch_account_balance(self, retries=3):
        """Fetch real account balance (INR and USDT) from CoinDCX."""
        for attempt in range(retries):
            try:
                headers = {
                    "X-API-KEY": self.config["api"]["api_key"],
                    "X-API-SECRET": self.config["api"]["api_secret"]
                }
                response = requests.get(
                    f"{self.api_base_url}/exchange/v1/users/balances",
                    headers=headers,
                    timeout=5
                )
                response.raise_for_status()
                data = response.json()
                # Find INR and USDT balances
                inr = next((float(x["balance_amount"])
                            for x in data if x["currency"] == "INR"), 0)
                usdt = next((float(x["balance_amount"])
                             for x in data if x["currency"] == "USDT"), 0)
                return {"INR": inr, "USDT": usdt}
            except Exception as e:
                self.logger.error(f"Attempt {attempt + 1}/{retries} - Error fetching account balance: {e}")
                if attempt < retries - 1:
                    time.sleep(1)
                continue
        self.logger.error("Failed to fetch account balance after retries")
        return {"INR": 0, "USDT": 0}

    def fetch_order_book(self, symbol, depth=50, retries=3):
        """Fetch order book for a futures symbol. Returns best bid/ask, spread, and total volume at top 5 levels."""
        for attempt in range(retries):
            try:
                url = f"{self.public_base_url}/market_data/v3/orderbook/{symbol}-futures/{depth}"
                response = requests.get(url, timeout=5)
                response.raise_for_status()
                data = response.json()
                asks = data.get("asks", {})
                bids = data.get("bids", {})
                # Parse price/volume pairs, sort by price
                ask_prices = sorted([(float(price), float(vol)) for price, vol in asks.items()])
                bid_prices = sorted([(float(price), float(vol)) for price, vol in bids.items()], reverse=True)
                best_ask = ask_prices[0][0] if ask_prices else None
                best_bid = bid_prices[0][0] if bid_prices else None
                spread = (best_ask - best_bid) if (best_ask and best_bid) else None
                # Total volume at top 5 levels
                ask_vol = sum(vol for _, vol in ask_prices[:5])
                bid_vol = sum(vol for _, vol in bid_prices[:5])
                return {
                    "best_ask": best_ask,
                    "best_bid": best_bid,
                    "spread": spread,
                    "ask_vol": ask_vol,
                    "bid_vol": bid_vol,
                    "raw": data
                }
            except Exception as e:
                self.logger.error(f"Attempt {attempt + 1}/{retries} - Error fetching order book for {symbol}: {e}")
                if attempt < retries - 1:
                    time.sleep(1)
                continue
        self.logger.error(f"Failed to fetch order book for {symbol} after {retries} attempts")
        return None

    def fetch_top_movers(self, top_n=40, min_volume=500000, retries=3):
        """Fetch top N symbols by volume (REMOVED 24h change logic) from CoinDCX futures market data, filtered by min_volume (in USDT)."""
        for attempt in range(retries):
            try:
                # Test DNS resolution first
                try:
                    socket.gethostbyname('public.coindcx.com')
                except socket.gaierror:
                    self.logger.error(f"DNS resolution failed for public.coindcx.com on attempt {attempt + 1}")
                    if attempt < retries - 1:
                        time.sleep(2)
                    continue
                
                response = self.session.get(
                    f"{self.public_base_url}/market_data/v3/current_prices/futures/rt",
                    timeout=10,
                    verify=False
                )
                response.raise_for_status()
                data = response.json()
                prices = data.get("prices", {})
                # Build a list of (symbol, volume, pct_change for compatibility)
                high_volume_symbols = []
                for symbol, price_data in prices.items():
                    pct_change = float(price_data.get("pc", 0))  # Keep for compatibility but don't use for sorting
                    volume = float(price_data.get("v", 0))
                    if volume >= min_volume:
                        high_volume_symbols.append((symbol, volume, pct_change))
                # Sort by volume only, descending - REMOVED 24h change sorting
                high_volume_symbols.sort(key=lambda x: x[1], reverse=True)
                # Return top N symbols (with volume and pct_change for compatibility)
                return [(symbol, pct_change, volume) for symbol, volume, pct_change in high_volume_symbols[:top_n]]
            except Exception as e:
                self.logger.error(f"Attempt {attempt + 1}/{retries} - Error fetching high volume symbols: {e}")
                if attempt < retries - 1:
                    time.sleep(1)
                continue
        self.logger.error("Failed to fetch high volume symbols after retries")
        return []

    def _extract_currencies_from_symbol(self, symbol):
        """Extract base and quote currencies from CoinDCX symbol format.
        
        Args:
            symbol: CoinDCX symbol (e.g., 'B-BTC_USDT')
            
        Returns:
            Tuple of (base_currency, quote_currency) or (None, None) if parsing fails
        """
        try:
            original_symbol = symbol
            # Remove 'B-' prefix if present
            if symbol.startswith('B-'):
                symbol = symbol[2:]
            
            # Split by underscore
            parts = symbol.split('_')
            if len(parts) >= 2:
                base = parts[0]
                quote = parts[1]
                self.logger.debug(f"Symbol currency extraction: {original_symbol} -> {base}/{quote}")
                return base, quote
            else:
                self.logger.warning(f"Cannot extract currencies from symbol: {symbol}")
                return None, None
        except Exception as e:
            self.logger.error(f"Error extracting currencies from symbol {symbol}: {e}")
            return None, None

    def _is_likely_cryptocompare_supported(self, symbol):
        """Check if a symbol is likely to be supported by CryptoCompare.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Boolean indicating if likely supported
        """
        # CryptoCompare supports most major cryptocurrencies
        definitely_supported = [
            'BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'DOT', 'AVAX', 'MATIC', 
            'LINK', 'UNI', 'LTC', 'BCH', 'XRP', 'DOGE', 'SHIB', 'TRX',
            'ALGO', 'FTM', 'VET', 'ICP', 'FIL', 'APT', 'NEAR', 'ATOM'
        ]
        
        # Remove B- prefix if present
        if symbol.startswith('B-'):
            symbol = symbol[2:]
        
        # Extract base currency
        parts = symbol.split('_')
        if len(parts) >= 2:
            base = parts[0]
            return base in definitely_supported
        
        return False

    def fetch_cryptocompare_candles(self, symbol, timeframe, limit=100, retries=3):
        """Fetch candlestick data from CryptoCompare API.
        
        Args:
            symbol: Trading symbol (e.g., 'B-BTC_USDT')
            timeframe: Timeframe in minutes ('1', '5', '15', '30', '60')
            limit: Number of candles to fetch
            retries: Number of retry attempts
            
        Returns:
            List of candle dictionaries with OHLCV data
        """
        # Check if CryptoCompare API is enabled
        if not self.cryptocompare_enabled:
            self.logger.debug(f"CryptoCompare API is disabled, skipping {symbol}")
            return []
        
        # Check if we've hit too many errors globally
        if self._cryptocompare_error_count >= self._max_cryptocompare_errors:
            self.logger.debug(f"CryptoCompare API disabled due to too many errors ({self._cryptocompare_error_count})")
            return []
        
        # Check if this specific symbol failed before
        if symbol in self._cryptocompare_failed_symbols:
            self.logger.debug(f"Symbol {symbol} previously failed CryptoCompare API, skipping")
            return []
        
        # Check if symbol is likely supported before making API call
        if not self._is_likely_cryptocompare_supported(symbol):
            self.logger.debug(f"Symbol {symbol} is unlikely to be supported by CryptoCompare, skipping")
            return []
        
        # Use configured retries
        max_retries = self.cryptocompare_config.get("max_retries", 3)
        for attempt in range(max_retries):
            try:
                # Extract base and quote currency from symbol (e.g., B-BTC_USDT -> BTC, USDT)
                fsym, tsym = self._extract_currencies_from_symbol(symbol)
                if not fsym or not tsym:
                    self.logger.warning(f"Could not extract currencies from symbol {symbol}")
                    return []
                
                # Get base URL and API key from config
                base_url = self.cryptocompare_config.get("base_url", "https://min-api.cryptocompare.com")
                api_key = self.cryptocompare_config.get("api_key")
                
                # Build API URL for histominute endpoint
                url = f"{base_url}/data/v2/histominute"
                params = {
                    "fsym": fsym,
                    "tsym": tsym,
                    "limit": limit
                }
                
                # Add aggregate parameter if timeframe is not 1 minute
                timeframe_map = {
                    "1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60,
                    "1": 1, "5": 5, "15": 15, "30": 30, "60": 60
                }
                aggregate = timeframe_map.get(timeframe, 1)
                if aggregate > 1:
                    params["aggregate"] = aggregate
                
                # Set up headers with API key
                headers = {}
                if api_key:
                    headers["authorization"] = f"Apikey {api_key}"
                
                self.logger.debug(f"Fetching CryptoCompare data: {url} with params {params}")
                
                response = self.session.get(url, params=params, headers=headers, timeout=15)
                
                # Check for specific error responses
                if response.status_code == 400:
                    # Track this symbol as failed
                    self._cryptocompare_failed_symbols.add(symbol)
                    self._cryptocompare_error_count += 1
                    
                    # Only log the first few errors to avoid spam
                    if self._cryptocompare_error_count <= 3:
                        error_data = response.json() if response.content else {}
                        error_msg = error_data.get('Message', 'Bad Request')
                        self.logger.warning(f"CryptoCompare API 400 error for {symbol} ({fsym}/{tsym}): {error_msg}")
                    elif self._cryptocompare_error_count == self._max_cryptocompare_errors:
                        self.logger.warning(f"CryptoCompare API has failed {self._max_cryptocompare_errors} times, disabling for this session")
                    
                    # This symbol is not supported by CryptoCompare, return empty list
                    return []
                
                response.raise_for_status()
                data = response.json()
                
                # Parse CryptoCompare response format
                if "Data" in data and "Data" in data["Data"] and isinstance(data["Data"]["Data"], list):
                    candles = []
                    for item in data["Data"]["Data"]:
                        # Skip empty candles (volume = 0 indicates no trading)
                        if item.get("volumefrom", 0) == 0:
                            continue
                            
                        candle = {
                            "open": float(item.get("open", 0)),
                            "high": float(item.get("high", 0)),
                            "low": float(item.get("low", 0)),
                            "close": float(item.get("close", 0)),
                            "volume": float(item.get("volumefrom", 0)),
                            "timestamp": int(item.get("time", 0))
                        }
                        candles.append(candle)
                    
                    self.logger.debug(f"Fetched {len(candles)} candles from CryptoCompare for {symbol} {timeframe}")
                    return candles
                else:
                    self.logger.warning(f"Unexpected CryptoCompare response format for {symbol}: {data}")
                    return []
                    
            except requests.exceptions.ConnectionError as e:
                self.logger.error(f"Connection error on attempt {attempt + 1}/{retries} for {symbol}: {e}")
                if attempt < retries - 1:
                    time.sleep(3)
                continue
            except requests.exceptions.Timeout as e:
                self.logger.warning(f"Timeout on attempt {attempt + 1}/{retries} for {symbol}: {e}")
                if attempt < retries - 1:
                    time.sleep(2)
                continue
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 400:
                    # Track as failed and reduce spam
                    self._cryptocompare_failed_symbols.add(symbol)
                    self._cryptocompare_error_count += 1
                    if self._cryptocompare_error_count <= 3:
                        self.logger.warning(f"CryptoCompare API 400 error for {symbol} - symbol not supported")
                    return []  # Don't retry for 400 errors
                else:
                    self._cryptocompare_error_count += 1
                    if self._cryptocompare_error_count <= 3:
                        self.logger.error(f"HTTP error on attempt {attempt + 1}/{max_retries} for {symbol}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(1)
                    continue
            except Exception as e:
                self._cryptocompare_error_count += 1
                if self._cryptocompare_error_count <= 3:
                    self.logger.error(f"Unexpected error on attempt {attempt + 1}/{max_retries} for {symbol}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                continue
                
        # Track as failed after all retries exhausted
        self._cryptocompare_failed_symbols.add(symbol)
        if self._cryptocompare_error_count <= 3:
            self.logger.error(f"Failed to fetch CryptoCompare candlestick data for {symbol} after {max_retries} attempts")
        return []

    def fetch_multi_timeframe_data(self, symbol, timeframes, limit=100):
        """Fetch candlestick data for multiple timeframes.
        
        Args:
            symbol: Trading symbol
            timeframes: List of timeframes to fetch (e.g., ['5m', '15m', '30m'])
            limit: Number of candles per timeframe
            
        Returns:
            Dictionary with timeframe as key and candle data as value
        """
        multi_tf_data = {}
        
        for tf in timeframes:
            try:
                # Try CryptoCompare first for 15m and 30m
                if tf in ['15m', '30m']:
                    candles = self.fetch_cryptocompare_candles(symbol, tf, limit)
                    if candles:
                        multi_tf_data[tf] = candles
                        self.logger.debug(f"Successfully fetched {tf} data from CryptoCompare for {symbol}")
                        continue
                    else:
                        self.logger.debug(f"CryptoCompare returned no data for {symbol} {tf}, trying CoinDCX")
                
                # Fallback to CoinDCX for other timeframes or if CryptoCompare fails
                candles = self.fetch_candlestick_data(symbol, tf, limit)
                if candles:
                    multi_tf_data[tf] = candles
                    self.logger.debug(f"Successfully fetched {tf} data from CoinDCX for {symbol}")
                else:
                    self.logger.warning(f"Failed to fetch {tf} data for {symbol}")
                    
            except Exception as e:
                self.logger.error(f"Error fetching {tf} data for {symbol}: {e}")
                continue
        
        return multi_tf_data