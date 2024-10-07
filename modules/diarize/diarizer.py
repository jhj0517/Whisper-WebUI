import os
import torch
try:
    import intel_extension_for_pytorch as ipex
    if torch.xpu.is_available():
        xpu_available = True
except:
    pass
from typing import List, Union, BinaryIO, Optional
import numpy as np
import time
import logging

from modules.utils.paths import DIARIZATION_MODELS_DIR
from modules.diarize.diarize_pipeline import DiarizationPipeline, assign_word_speakers
from modules.diarize.audio_loader import load_audio


class Diarizer:
    def __init__(self,
                 model_dir: str = DIARIZATION_MODELS_DIR
                 ):
        self.device = self.get_device()
        self.available_device = self.get_available_device()
        self.compute_type = "float16"
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.pipe = None

    def run(self,
            audio: Union[str, BinaryIO, np.ndarray],
            transcribed_result: List[dict],
            use_auth_token: str,
            device: Optional[str] = None
            ):
        """
        Diarize transcribed result as a post-processing

        Parameters
        ----------
        audio: Union[str, BinaryIO, np.ndarray]
            Audio input. This can be file path or binary type.
        transcribed_result: List[dict]
            transcribed result through whisper.
        use_auth_token: str
            Huggingface token with READ permission. This is only needed the first time you download the model.
            You must manually go to the website https://huggingface.co/pyannote/speaker-diarization-3.1 and agree to their TOS to download the model.
        device: Optional[str]
            Device for diarization.

        Returns
        ----------
        segments_result: List[dict]
            list of dicts that includes start, end timestamps and transcribed text
        elapsed_time: float
            elapsed time for running
        """
        start_time = time.time()

        if device is None:
            device = self.device

        if device != self.device or self.pipe is None:
            self.update_pipe(
                device=device,
                use_auth_token=use_auth_token
            )

        audio = load_audio(audio)

        diarization_segments = self.pipe(audio)
        diarized_result = assign_word_speakers(
            diarization_segments,
            {"segments": transcribed_result}
        )

        for segment in diarized_result["segments"]:
            speaker = "None"
            if "speaker" in segment:
                speaker = segment["speaker"]
            segment["text"] = speaker + "|" + segment["text"].strip()

        elapsed_time = time.time() - start_time
        return diarized_result["segments"], elapsed_time

    def update_pipe(self,
                    use_auth_token: str,
                    device: str
                    ):
        """
        Set pipeline for diarization

        Parameters
        ----------
        use_auth_token: str
            Huggingface token with READ permission. This is only needed the first time you download the model.
            You must manually go to the website https://huggingface.co/pyannote/speaker-diarization-3.1 and agree to their TOS to download the model.
        device: str
            Device for diarization.
        """
        self.device = device

        os.makedirs(self.model_dir, exist_ok=True)

        if (not os.listdir(self.model_dir) and
                not use_auth_token):
            print(
                "\nFailed to diarize. You need huggingface token and agree to their requirements to download the diarization model.\n"
                "Go to \"https://huggingface.co/pyannote/speaker-diarization-3.1\" and follow their instructions to download the model.\n"
            )
            return

        logger = logging.getLogger("speechbrain.utils.train_logger")
        # Disable redundant torchvision warning message
        logger.disabled = True
        self.pipe = DiarizationPipeline(
            use_auth_token=use_auth_token,
            device=device,
            cache_dir=self.model_dir
        )
        logger.disabled = False

    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        elif torch.xpu.is_available():
            return "xpu"
        else:
            return "cpu"

    @staticmethod
    def get_available_device():
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        elif torch.backends.mps.is_available():
            devices.append("mps")
        elif torch.xpu.is_available():
            devices.append("xpu")
        return devices
