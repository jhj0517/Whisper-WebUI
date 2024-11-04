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

from modules.utils.paths import (INSANELY_FAST_WHISPER_MODELS_DIR, DIARIZATION_MODELS_DIR, UVR_MODELS_DIR, OUTPUT_DIR)
from modules.whisper.data_classes import *
from modules.whisper.base_transcription_pipeline import BaseTranscriptionPipeline


class InsanelyFastWhisperInference(BaseTranscriptionPipeline):
    def __init__(self,
                 model_dir: str = INSANELY_FAST_WHISPER_MODELS_DIR,
                 diarization_model_dir: str = DIARIZATION_MODELS_DIR,
                 uvr_model_dir: str = UVR_MODELS_DIR,
                 output_dir: str = OUTPUT_DIR,
                 ):
        super().__init__(
            model_dir=model_dir,
            output_dir=output_dir,
            diarization_model_dir=diarization_model_dir,
            uvr_model_dir=uvr_model_dir
        )
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        self.available_models = self.get_model_paths()

    def transcribe(self,
                   audio: Union[str, np.ndarray, torch.Tensor],
                   progress: gr.Progress = gr.Progress(),
                   *whisper_params,
                   ) -> Tuple[List[Segment], float]:
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
        segments_result: List[Segment]
            list of Segment that includes start, end timestamps and transcribed text
        elapsed_time: float
            elapsed time for transcription
        """
        start_time = time.time()
        params = WhisperParams.from_list(list(whisper_params))

        if params.model_size != self.current_model_size or self.model is None or self.current_compute_type != params.compute_type:
            self.update_model(params.model_size, params.compute_type, progress)

        progress(0, desc="Transcribing...Progress is not shown in insanely-fast-whisper.")
        with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(style="yellow1", pulse_style="white"),
                TimeElapsedColumn(),
        ) as progress:
            progress.add_task("[yellow]Transcribing...", total=None)

            kwargs = {
                "no_speech_threshold": params.no_speech_threshold,
                "temperature": params.temperature,
                "compression_ratio_threshold": params.compression_ratio_threshold,
                "logprob_threshold": params.log_prob_threshold,
            }

            if self.current_model_size.endswith(".en"):
                pass
            else:
                kwargs["language"] = params.lang
                kwargs["task"] = "translate" if params.is_translate else "transcribe"

            segments = self.model(
                inputs=audio,
                return_timestamps=True,
                chunk_length_s=params.chunk_length,
                batch_size=params.batch_size,
                generate_kwargs=kwargs
            )

        segments_result = []
        for item in segments["chunks"]:
            start, end = item["timestamp"][0], item["timestamp"][1]
            if end is None:
                end = start
            segments_result.append(Segment(
                text=item["text"],
                start=start,
                end=end
            ))

        elapsed_time = time.time() - start_time
        return segments_result, elapsed_time

    def update_model(self,
                     model_size: str,
                     compute_type: str,
                     progress: gr.Progress = gr.Progress(),
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

    def get_model_paths(self):
        """
        Get available models from models path including fine-tuned model.

        Returns
        ----------
        Name set of models
        """
        openai_models = whisper.available_models()
        distil_models = ["distil-large-v2", "distil-large-v3", "distil-medium.en", "distil-small.en"]
        default_models = openai_models + distil_models

        existing_models = os.listdir(self.model_dir)
        wrong_dirs = [".locks"]

        available_models = default_models + existing_models
        available_models = [model for model in available_models if model not in wrong_dirs]
        available_models = sorted(set(available_models), key=available_models.index)

        return available_models

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
