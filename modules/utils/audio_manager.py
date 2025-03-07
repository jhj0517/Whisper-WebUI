from typing import Optional, Union
import soundfile as sf
import os
import numpy as np
from faster_whisper.audio import decode_audio

from modules.utils.files_manager import is_video
from modules.utils.logger import get_logger

logger = get_logger()


def validate_audio(audio: Optional[str] = None):
    """Validate audio file and check if it's corrupted"""
    if isinstance(audio, np.ndarray):
        return True

    if not os.path.exists(audio):
        logger.info(f"The file {audio} does not exist. Please check the path.")
        return False

    try:
        audio = decode_audio(audio)
        return True
    except Exception as e:
        logger.info(f"The file {audio} is not able to open or corrupted. Please check the file. {e}")
        return False
