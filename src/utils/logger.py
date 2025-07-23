import logging
import os
import sys
from datetime import datetime

# Global logger instance
_logger_instance = None

def setup_logger(name="trading_bot", log_file="logs/trading_bot.log", level=logging.INFO):
    """
    Setup logger with singleton pattern to prevent multiple initializations.
    
    Args:
        name: Logger name
        log_file: Log file path
        level: Logging level
        
    Returns:
        Logger instance
    """
    global _logger_instance
    
    # Return existing logger if already initialized
    if _logger_instance is not None:
        return _logger_instance
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear any existing handlers to prevent duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create file handler with UTF-8 encoding
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    
    # Create console handler with proper encoding for Windows
    if sys.platform.startswith('win'):
        # Use UTF-8 for Windows console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Store the logger instance
    _logger_instance = logger
    
    # Log initialization message
    logger.info("=== Logger initialized successfully ===")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Log level: {logging.getLevelName(level)}")
    
    return logger