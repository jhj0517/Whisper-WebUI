import os
import time
import numpy as np
from typing import BinaryIO, Union, Tuple, List
import torch
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available
import gradio as gr
import wget

from modules.whisper_parameter import *
from modules.whisper_base import WhisperBase


class InsanelyFastWhisperInference(WhisperBase):
    def __init__(self):
        super().__init__(
            model_dir=os.path.join("models", "Whisper", "insanely_fast_whisper")
        )
        self.available_compute_types = ["float16"]

    def transcribe(self,
                   audio: Union[str, np.ndarray, torch.Tensor],
                   progress: gr.Progress,
                   *whisper_params,
                   ) -> Tuple[List[dict], float]:
        """
        transcribe method for faster-whisper.

        Parameters
        ----------
        audio: Union[str, BinaryIO, np.ndarray]
            Audio path or file binary or Audio numpy array
        progress: gr.Progress
            Indicator to show progress directly in gradio.
        *whisper_params: tuple
            Gradio components related to Whisper. see whisper_data_class.py for details.

        Returns
        ----------
        segments_result: List[dict]
            list of dicts that includes start, end timestamps and transcribed text
        elapsed_time: float
            elapsed time for transcription
        """
        start_time = time.time()
        params = WhisperValues(*whisper_params)

        if params.model_size != self.current_model_size or self.model is None or self.current_compute_type != params.compute_type:
            self.update_model(params.model_size, params.compute_type, progress)

        if params.lang == "Automatic Detection":
            params.lang = None

        progress(0, desc="Transcribing...")
        segments = self.model(
            inputs=audio,
            chunk_length_s=30,
            batch_size=24,
            return_timestamps=True,
        )
        segments_result = self.format_result(
            transcribed_result=segments,
        )
        elapsed_time = time.time() - start_time
        return segments_result, elapsed_time

    def update_model(self,
                     model_size: str,
                     compute_type: str,
                     progress: gr.Progress,
                     ):
        """
        Update current model setting

        Parameters
        ----------
        model_size: str
            Size of whisper model
        compute_type: str
            Compute type for transcription.
            see more info : https://opennmt.net/CTranslate2/quantization.html
        progress: gr.Progress
            Indicator to show progress directly in gradio.
        """
        progress(0, desc="Initializing Model..")
        model_path = os.path.join(self.model_dir, model_size)
        if not os.path.isdir(model_path) or not os.listdir(model_path):
            self.download_model(
                model_size=model_size,
                download_root=model_path,
                progress=progress
            )

        self.current_compute_type = compute_type
        self.current_model_size = model_size

        self.model = pipeline(
            "automatic-speech-recognition",
            model=os.path.join(self.model_dir, model_size),
            torch_dtype=self.current_compute_type,
            device=self.device,
            model_kwargs={"attn_implementation": "flash_attention_2"} if is_flash_attn_2_available() else {"attn_implementation": "sdpa"},
        )

    @staticmethod
    def format_result(
        transcribed_result: dict
    ) -> List[dict]:
        """
        Format the transcription result of insanely_fast_whisper as the same with other implementation.

        Parameters
        ----------
        transcribed_result: dict
            Transcription result of the insanely_fast_whisper
        progress: gr.Progress
            Indicator to show progress directly in gradio.

        Returns
        ----------
        result: List[dict]
            Formatted result as the same with other implementation
        """
        result = transcribed_result["chunks"]
        for item in result:
            start, end = item["timestamp"][0], item["timestamp"][1]
            item["start"] = start
            item["end"] = end
        return result

    @staticmethod
    def download_model(
        model_size: str,
        download_root: str,
        progress: gr.Progress
    ):
        progress(0, 'Initializing model..')
        print(f'Downloading {model_size} to "{download_root}"....')

        os.makedirs(download_root, exist_ok=True)
        download_list = [
            "model.safetensors",
            "config.json",
            "generation_config.json",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "added_tokens.json",
            "special_tokens_map.json",
            "vocab.json",
        ]

        download_host = f"https://huggingface.co/openai/whisper-{model_size}/resolve/main"
        for item in download_list:
            wget.download(
                download_host+"/"+item,
                download_root
            )
