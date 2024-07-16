from argparse import Namespace
import os

from modules.whisper.faster_whisper_inference import FasterWhisperInference
from modules.whisper.whisper_Inference import WhisperInference
from modules.whisper.insanely_fast_whisper_inference import InsanelyFastWhisperInference
from modules.whisper.whisper_base import WhisperBase


class WhisperFactory:
    @staticmethod
    def create_whisper_inference(
        whisper_type: str,
        model_dir: str,
        output_dir: str,
        args: Namespace
    ) -> "WhisperBase":
        """
        Create a whisper inference class based on the provided whisper_type.

        Parameters
        ----------
        whisper_type: str
            The repository name of whisper inference to use. Supported values are:
            - "faster-whisper" from
            - "whisper"
            - insanely-fast-whisper", "insanely_fast_whisper", "insanelyfastwhisper",
              "insanely-faster-whisper", "insanely_faster_whisper", "insanelyfasterwhisper"
        model_dir: str
            The directory path where the whisper model is located.
        output_dir: str
            The directory path where the output files will be saved.
        args: Any
            Additional arguments to be passed to the whisper inference object.

        Returns
        -------
        WhisperBase
            An instance of the appropriate whisper inference class based on the whisper_type.
        """
        # Temporal fix of the bug : https://github.com/jhj0517/Whisper-WebUI/issues/144
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

        whisper_type = whisper_type.lower().strip()

        faster_whisper_typos = ["faster_whisper", "faster-whisper", "fasterwhisper"]
        whisper_typos = ["whisper"]
        insanely_fast_whisper_typos = [
            "insanely_fast_whisper", "insanely-fast-whisper", "insanelyfastwhisper",
            "insanely_faster_whisper", "insanely-faster-whisper", "insanelyfasterwhisper"
        ]

        if whisper_type in faster_whisper_typos:
            return FasterWhisperInference(model_dir, output_dir, args)
        elif whisper_type in whisper_typos:
            return WhisperInference(model_dir, output_dir, args)
        elif whisper_type in insanely_fast_whisper_typos:
            return InsanelyFastWhisperInference(model_dir, output_dir, args)
        else:
            return FasterWhisperInference(model_dir, output_dir, args)
