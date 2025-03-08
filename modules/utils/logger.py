import logging
from typing import Optional


def get_logger(name: Optional[str] = None):
    if name is None:
        name = "Whisper-WebUI"
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        handler = logging.StreamHandler()
        # handler.setFormatter(formatter)

        logger.addHandler(handler)

    return logger