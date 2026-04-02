import logging
import sys

def get_core_logger(name: str) -> logging.Logger:
    """
    Returns a configured, production-ready logger instances.
    In a real production environment, this would integrate with LangSmith, 
    Datadog, or ELK stacks.
    """
    logger = logging.getLogger(name)
    
    # Prevent duplicate logs if the logger is already initialized
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        
    return logger

# Example usage available on import:
# from src.utils.logger import get_core_logger
# logger = get_core_logger("LangChainArchitecture")
