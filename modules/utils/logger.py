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

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        file_handler = logging.FileHandler("webui.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger