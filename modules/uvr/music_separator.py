from typing import Optional, Union
import numpy as np
import torchaudio
import soundfile as sf
import os
import torch
import gc
import gradio as gr
from datetime import datetime

from uvr.models import MDX, Demucs, VrNetwork, MDXC
from modules.utils.files_manager import is_video
from modules.diarize.audio_loader import load_audio


class MusicSeparator:
    def __init__(self,
                 model_dir: Optional[str] = None,
                 output_dir: Optional[str] = None):
        self.model = None
        self.device = self.get_device()
        self.available_devices = ["cpu", "cuda"]
        self.model_dir = model_dir
        self.output_dir = output_dir
        instrumental_output_dir = os.path.join(self.output_dir, "instrumental")
        vocals_output_dir = os.path.join(self.output_dir, "vocals")
        os.makedirs(instrumental_output_dir, exist_ok=True)
        os.makedirs(vocals_output_dir, exist_ok=True)
        self.audio_info = None
        self.available_models = ["UVR-MDX-NET-Inst_HQ_4", "UVR-MDX-NET-Inst_3"]
        self.default_model = self.available_models[0]
        self.current_model_size = self.default_model
        self.model_config = {
            "segment": 256,
            "split": True
        }

    def update_model(self,
                     model_name: str = "UVR-MDX-NET-Inst_1",
                     device: Optional[str] = None,
                     segment_size: int = 256):
        """
        Update model with the given model name

        Args:
            model_name (str): Model name.
            device (str): Device to use for the model.
            segment_size (int): Segment size for the prediction.
        """
        if device is None:
            device = self.device

        self.device = device
        self.model_config = {
            "segment": segment_size,
            "split": True
        }
        self.model = MDX(name=model_name,
                         other_metadata=self.model_config,
                         device=self.device,
                         logger=None,
                         model_dir=self.model_dir)

    def separate(self,
                 audio: Union[str, np.ndarray],
                 model_name: str,
                 device: Optional[str] = None,
                 segment_size: int = 256,
                 save_file: bool = False,
                 progress: gr.Progress = gr.Progress()) -> tuple[np.ndarray, np.ndarray]:
        """
        Separate the background music from the audio.

        Args:
            audio (Union[str, np.ndarray]): Audio path or numpy array.
            model_name (str): Model name.
            device (str): Device to use for the model.
            segment_size (int): Segment size for the prediction.
            save_file (bool): Whether to save the separated audio to output path or not.
            progress (gr.Progress): Gradio progress indicator.

        Returns:
            tuple[np.ndarray, np.ndarray]: Instrumental and vocals numpy arrays.
        """
        if isinstance(audio, str):
            output_filename, ext = os.path.basename(audio), ".wav"

            if is_video(audio):
                audio = load_audio(audio)
                sample_rate = 16000
            else:
                self.audio_info = torchaudio.info(audio)
                sample_rate = self.audio_info.sample_rate
        else:
            timestamp = datetime.now().strftime("%m%d%H%M%S")
            output_filename, ext = f"UVR-{timestamp}", ".wav"
            sample_rate = 16000

        model_config = {
            "segment": segment_size,
            "split": True
        }

        if (self.model is None or
                self.current_model_size != model_name or
                self.model_config != model_config or
                self.model.sample_rate != sample_rate or
                self.device != device):
            progress(0, desc="Initializing UVR Model..")
            self.update_model(
                model_name=model_name,
                device=device,
                segment_size=segment_size
            )
            self.model.sample_rate = sample_rate

        progress(0, desc="Separating background music from the audio..")
        result = self.model(audio)
        instrumental, vocals = result["instrumental"].T, result["vocals"].T

        if save_file:
            instrumental_output_path = os.path.join(self.output_dir, "instrumental", f"{output_filename}-instrumental{ext}")
            vocals_output_path = os.path.join(self.output_dir, "vocals", f"{output_filename}-vocals{ext}")
            sf.write(instrumental_output_path, instrumental, sample_rate, format="WAV")
            sf.write(vocals_output_path, vocals, sample_rate, format="WAV")

        return instrumental, vocals

    @staticmethod
    def get_device():
        """Get device for the model"""
        return "cuda" if torch.cuda.is_available() else "cpu"

    def offload(self):
        """Offload the model and free up the memory"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        self.audio_info = None
