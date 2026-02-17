from typing import Optional
import os
import torch

from modules.utils.paths import (FASTER_WHISPER_MODELS_DIR, DIARIZATION_MODELS_DIR, OUTPUT_DIR,
                                 INSANELY_FAST_WHISPER_MODELS_DIR, WHISPER_MODELS_DIR, UVR_MODELS_DIR)
from modules.whisper.faster_whisper_inference import FasterWhisperInference
from modules.whisper.whisper_Inference import WhisperInference
from modules.whisper.insanely_fast_whisper_inference import InsanelyFastWhisperInference
from modules.whisper.base_transcription_pipeline import BaseTranscriptionPipeline
from modules.whisper.data_classes import *
from modules.utils.logger import get_logger


logger = get_logger()


class WhisperFactory:
    @staticmethod
    def create_whisper_inference(
        whisper_type: str,
        whisper_model_dir: str = WHISPER_MODELS_DIR,
        faster_whisper_model_dir: str = FASTER_WHISPER_MODELS_DIR,
        insanely_fast_whisper_model_dir: str = INSANELY_FAST_WHISPER_MODELS_DIR,
        diarization_model_dir: str = DIARIZATION_MODELS_DIR,
        uvr_model_dir: str = UVR_MODELS_DIR,
        output_dir: str = OUTPUT_DIR,
    ) -> "BaseTranscriptionPipeline":
        """
        Create a whisper inference class based on the provided whisper_type.

        Parameters
        ----------
        whisper_type : str
            The type of Whisper implementation to use. Supported values (case-insensitive):
            - "faster-whisper": https://github.com/openai/whisper
            - "whisper": https://github.com/openai/whisper
            - "insanely-fast-whisper": https://github.com/Vaibhavs10/insanely-fast-whisper
        whisper_model_dir : str
            Directory path for the Whisper model.
        faster_whisper_model_dir : str
            Directory path for the Faster Whisper model.
        insanely_fast_whisper_model_dir : str
            Directory path for the Insanely Fast Whisper model.
        diarization_model_dir : str
            Directory path for the diarization model.
        uvr_model_dir : str
            Directory path for the UVR model.
        output_dir : str
            Directory path where output files will be saved.

        Returns
        -------
        BaseTranscriptionPipeline
            An instance of the appropriate whisper inference class based on the whisper_type.
        """
        # Temporal fix of the bug : https://github.com/jhj0517/Whisper-WebUI/issues/144
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

        whisper_type = whisper_type.strip().lower()

        if whisper_type == WhisperImpl.FASTER_WHISPER.value:
            if torch.xpu.is_available():
                logger.warning("XPU is detected but faster-whisper only supports CUDA. "
                               "Automatically switching to insanely-whisper implementation.")
                return InsanelyFastWhisperInference(
                    model_dir=insanely_fast_whisper_model_dir,
                    output_dir=output_dir,
                    diarization_model_dir=diarization_model_dir,
                    uvr_model_dir=uvr_model_dir
                )

            return FasterWhisperInference(
                model_dir=faster_whisper_model_dir,
                output_dir=output_dir,
                diarization_model_dir=diarization_model_dir,
                uvr_model_dir=uvr_model_dir
            )
        elif whisper_type == WhisperImpl.WHISPER.value:
            return WhisperInference(
                model_dir=whisper_model_dir,
                output_dir=output_dir,
                diarization_model_dir=diarization_model_dir,
                uvr_model_dir=uvr_model_dir
            )
        elif whisper_type == WhisperImpl.INSANELY_FAST_WHISPER.value:
            return InsanelyFastWhisperInference(
                model_dir=insanely_fast_whisper_model_dir,
                output_dir=output_dir,
                diarization_model_dir=diarization_model_dir,
                uvr_model_dir=uvr_model_dir
            )
        else:
            return FasterWhisperInference(
                model_dir=faster_whisper_model_dir,
                output_dir=output_dir,
                diarization_model_dir=diarization_model_dir,
                uvr_model_dir=uvr_model_dir
            )
