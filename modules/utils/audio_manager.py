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

    if is_video(audio):
        try:
            audio = decode_audio(audio)
            return True
        except Exception as e:
            logger.info(f"The file {audio} is not able to open or corrupted. Please check the file. {e}")
            return False

    if not os.path.exists(audio):
        return False

    try:
        with sf.SoundFile(audio) as f:
            if f.frames > 0:
                return True
            else:
                return False
    except Exception as e:
        logger.info(f"The file {audio} is not able to open or corrupted. Please check the file. {e}")
        return False
