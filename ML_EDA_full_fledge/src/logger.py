import logging
from pathlib import Path

from src.config import LOG_FILE, LOGS_DIR

# LOG_FILE = "logs/app.log"
# LOGS_DIR = Path("logs")

LOGS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format=
    "%(asctime)s - %(levelname)s - %(message)s"
)

logger=logging.getLogger()
