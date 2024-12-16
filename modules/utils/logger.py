import logging


def get_gradio_logger():
    return get_logger("Whisper-WebUI")


def get_backend_logger():
    return get_logger("Whisper-WebUI-Backend")


def get_logger(name: str):
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        logger.addHandler(handler)

    return logger