# Credit to Team UVR : https://github.com/Anjok07/ultimatevocalremovergui
from typing import Optional
import soundfile as sf
import os
import torch

from uvr.models import MDX, Demucs, VrNetwork, MDXC


class MusicSeparator:
    def __init__(self,
                 model_dir: Optional[str] = None,
                 output_dir: Optional[str] = None):
        self.model = None
        self.device = self.get_device()
        self.model_dir = model_dir
        self.output_dir = output_dir

    def update_model(self,
                     model_name: str = "UVR-MDX-NET-Inst_1",
                     segment_size: int = 256):
        self.model = MDX(name="UVR-MDX-NET-Inst_1",
                         other_metadata={"segment": segment_size, "split": True},
                         device=self.device,
                         logger=None,
                         model_path="models\UVR\MDX_Net_Models\UVR-MDX-NET-Inst_HQ_1.onnx")

    def separate(self,
                 audio_file_path: str,
                 sample_rate: int = 44100):
        if self.model is None:
            self.model = self.update_model()

        filename = audio_file_path
        instrumental_output_path = os.path.join(self.output_dir, "instrumental", filename)
        vocals_output_path = os.path.join(self.output_dir, "vocals", filename)

        result = self.model(audio_file_path)
        instrumental, vocals = result["instrumental"], result["vocals"]

        sf.write('instrumental.wav', instrumental.T, sample_rate, format="WAV")
        sf.write('vocals.wav', vocals.T, sample_rate, format="WAV")

        return instrumental_output_path, vocals_output_path

    @staticmethod
    def get_device():
        return "cuda" if torch.cuda.is_available() else "cpu"