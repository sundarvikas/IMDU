# logger_config.py
import logging
import sys

def setup_logger():
    """
    Configures a logger to write structured events to a file.
    """
    # Create a custom logger
    logger = logging.getLogger("IMDU_APP")
    logger.setLevel(logging.INFO)

    # Prevent logs from propagating to the root logger (and showing up in the console)
    logger.propagate = False

    # Create handlers if they don't exist
    if not logger.handlers:
        # Create a file handler to write logs to a file
        file_handler = logging.FileHandler("imdu_usage.log", mode='a')
        file_handler.setLevel(logging.INFO)

        # Create a logging format. The "EVENT:" prefix makes parsing easy.
        formatter = logging.Formatter('%(asctime)s - EVENT:%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)

        # Add the handlers to the logger
        logger.addHandler(file_handler)

    return logger

# Create a logger instance that can be imported by other modules
app_logger = setup_logger()