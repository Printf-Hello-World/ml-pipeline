import logging

logging.basicConfig(level=logging.INFO,  # Set the default log level
                    format='%(asctime)s - %(levelname)s - %(message)s')

logger=logging.getLogger("Basic Logger")
