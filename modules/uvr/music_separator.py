# Credit to Team UVR : https://github.com/Anjok07/ultimatevocalremovergui
from typing import Optional
import torchaudio
import soundfile as sf
import os
import torch
import gc
import gradio as gr

from uvr.models import MDX, Demucs, VrNetwork, MDXC


class MusicSeparator:
    def __init__(self,
                 model_dir: Optional[str] = None,
                 output_dir: Optional[str] = None):
        self.model = None
        self.device = self.get_device()
        self.available_devices = ["cpu", "cuda"]
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.audio_info = None
        self.available_models = ["UVR-MDX-NET-Inst_1", "UVR-MDX-NET-Inst_HQ_1"]
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
                 audio_file_path: str,
                 model_name: str,
                 device: Optional[str] = None,
                 segment_size: int = 256,
                 progress: gr.Progress = gr.Progress()):
        if device is None:
            device = self.device

        self.audio_info = torchaudio.info(audio_file_path)
        sample_rate = self.audio_info.sample_rate

        filename, ext = os.path.splitext(audio_file_path)
        filename, ext = os.path.basename(filename), ".wav"
        instrumental_output_path = os.path.join(self.output_dir, "instrumental", f"{filename}-instrumental{ext}")
        vocals_output_path = os.path.join(self.output_dir, "vocals", f"{filename}-vocals{ext}")

        model_config = {
            "segment": segment_size,
            "split": True
        }

        if (self.model is None or
                self.current_model_size != model_name or
                self.model_config != model_config or
                self.audio_info.sample_rate != sample_rate):
            progress(0, desc="Initializing UVR Model..")
            self.update_model(
                model_name=model_name,
                device=device,
                segment_size=segment_size
            )
            self.model.sample_rate = sample_rate

        progress(0, desc="Separating background music from the audio..")
        result = self.model(audio_file_path)
        instrumental, vocals = result["instrumental"].T, result["vocals"].T

        sf.write(instrumental_output_path, instrumental, sample_rate, format="WAV")
        sf.write(vocals_output_path, vocals, sample_rate, format="WAV")

        return instrumental_output_path, vocals_output_path

    @staticmethod
    def get_device():
        return "cuda" if torch.cuda.is_available() else "cpu"

    def offload(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        self.audio_info = None
