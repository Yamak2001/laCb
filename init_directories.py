# init_directories.py
import logging
import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_directories():
    """Create all required directories for the application."""
    # Use config's initialize_directories
    config.initialize_directories()
    logger.info("All directories initialized from config")

if __name__ == "__main__":
    initialize_directories()
