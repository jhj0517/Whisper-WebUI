from typing import Optional, Union
import soundfile as sf
import os
import numpy as np


def validate_audio(audio: Optional[str] = None):
    """Validate audio file and check if it's corrupted"""
    if isinstance(audio, np.ndarray):
        return True

    if not os.path.exists(audio):
        return False

    try:
        with sf.SoundFile(audio) as f:
            if f.frames > 0:
                return True
            else:
                return False
    except Exception as e:
        return False
