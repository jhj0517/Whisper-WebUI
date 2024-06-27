import os
import time
import numpy as np
from typing import BinaryIO, Union, Tuple, List
import torch
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available
import gradio as gr
from huggingface_hub import hf_hub_download
import whisper
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn
from argparse import Namespace

from modules.whisper.whisper_parameter import *
from modules.whisper.whisper_base import WhisperBase


class InsanelyFastWhisperInference(WhisperBase):
    def __init__(self,
                 model_dir: str,
                 output_dir: str,
                 args: Namespace
                 ):
        super().__init__(
            model_dir=model_dir,
            output_dir=output_dir,
            args=args
        )
        openai_models = whisper.available_models()
        distil_models = ["distil-large-v2", "distil-large-v3", "distil-medium.en", "distil-small.en"]
        self.available_models = openai_models + distil_models
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
            Parameters related with whisper. This will be dealt with "WhisperParameters" data class

        Returns
        ----------
        segments_result: List[dict]
            list of dicts that includes start, end timestamps and transcribed text
        elapsed_time: float
            elapsed time for transcription
        """
        start_time = time.time()
        params = WhisperParameters.as_value(*whisper_params)

        if params.model_size != self.current_model_size or self.model is None or self.current_compute_type != params.compute_type:
            self.update_model(params.model_size, params.compute_type, progress)

        progress(0, desc="Transcribing...Progress is not shown in insanely-fast-whisper.")
        with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(style="yellow1", pulse_style="white"),
                TimeElapsedColumn(),
        ) as progress:
            progress.add_task("[yellow]Transcribing...", total=None)

            segments = self.model(
                inputs=audio,
                return_timestamps=True,
                chunk_length_s=params.chunk_length_s,
                batch_size=params.batch_size,
                generate_kwargs={
                    "language": params.lang,
                    "task": "translate" if params.is_translate and self.current_model_size in self.translatable_models else "transcribe",
                    "no_speech_threshold": params.no_speech_threshold,
                    "temperature": params.temperature,
                    "compression_ratio_threshold": params.compression_ratio_threshold
                }
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

        Returns
        ----------
        result: List[dict]
            Formatted result as the same with other implementation
        """
        result = transcribed_result["chunks"]
        for item in result:
            start, end = item["timestamp"][0], item["timestamp"][1]
            if end is None:
                end = start
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

        if model_size.startswith("distil"):
            repo_id = f"distil-whisper/{model_size}"
        else:
            repo_id = f"openai/whisper-{model_size}"
        for item in download_list:
            hf_hub_download(repo_id=repo_id, filename=item, local_dir=download_root)
