"""
Logging utility for bit-flip attack framework
Provides centralized logging configuration
"""
import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name=None, level=logging.INFO, log_file=None, console=True):
    """
    Set up a logger with both file and console handlers
    
    Args:
        name: Logger name (defaults to root logger if None)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        console: Whether to output to console (default True)
    
    Returns:
        logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name or __name__)
    logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if log_file specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_attack_logger(attack_name, log_dir='logs', level=logging.INFO):
    """
    Get a logger configured for a specific attack
    
    Args:
        attack_name: Name of the attack (e.g., 'lfw_face_attack')
        log_dir: Directory to save logs
        level: Logging level
    
    Returns:
        logger: Configured logger
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"{attack_name}_{timestamp}.log"
    
    return setup_logger(
        name=attack_name,
        level=level,
        log_file=log_file,
        console=True
    )


# Convenience functions for different log levels
def debug_mode_logger(name=None):
    """Create a logger with DEBUG level (shows all messages)"""
    return setup_logger(name, level=logging.DEBUG)


def quiet_mode_logger(name=None):
    """Create a logger with WARNING level (only warnings and errors)"""
    return setup_logger(name, level=logging.WARNING)


# Pre-configured loggers for common modules
def get_bit_manipulation_logger():
    """Get logger for bit manipulation operations"""
    return logging.getLogger('bitflip_attack.bit_manipulation')


def get_optimization_logger():
    """Get logger for optimization operations"""
    return logging.getLogger('bitflip_attack.optimization')


def get_evaluation_logger():
    """Get logger for evaluation operations"""
    return logging.getLogger('bitflip_attack.evaluation')

