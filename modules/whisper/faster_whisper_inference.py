import os
import time
import numpy as np
import torch
from typing import BinaryIO, Union, Tuple, List
import faster_whisper
from faster_whisper.vad import VadOptions
import ast
import ctranslate2
import whisper
import gradio as gr
from argparse import Namespace

from modules.whisper.whisper_parameter import *
from modules.whisper.whisper_base import WhisperBase


class FasterWhisperInference(WhisperBase):
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
        self.model_paths = self.get_model_paths()
        self.device = self.get_device()
        self.available_models = self.model_paths.keys()
        self.available_compute_types = ctranslate2.get_supported_compute_types(
            "cuda") if self.device == "cuda" else ctranslate2.get_supported_compute_types("cpu")

    def transcribe(self,
                   audio: Union[str, BinaryIO, np.ndarray],
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

        params.suppress_tokens = self.format_suppress_tokens_str(params.suppress_tokens)

        segments, info = self.model.transcribe(
            audio=audio,
            language=params.lang,
            task="translate" if params.is_translate and self.current_model_size in self.translatable_models else "transcribe",
            beam_size=params.beam_size,
            log_prob_threshold=params.log_prob_threshold,
            no_speech_threshold=params.no_speech_threshold,
            best_of=params.best_of,
            patience=params.patience,
            temperature=params.temperature,
            compression_ratio_threshold=params.compression_ratio_threshold,
            length_penalty=params.length_penalty,
            repetition_penalty=params.repetition_penalty,
            no_repeat_ngram_size=params.no_repeat_ngram_size,
            prefix=params.prefix,
            suppress_blank=params.suppress_blank,
            suppress_tokens=params.suppress_tokens,
            max_initial_timestamp=params.max_initial_timestamp,
            word_timestamps=params.word_timestamps,
            prepend_punctuations=params.prepend_punctuations,
            append_punctuations=params.append_punctuations,
            max_new_tokens=params.max_new_tokens,
            chunk_length=params.chunk_length,
            hallucination_silence_threshold=params.hallucination_silence_threshold,
            hotwords=params.hotwords,
            language_detection_threshold=params.language_detection_threshold,
            language_detection_segments=params.language_detection_segments,
            prompt_reset_on_temperature=params.prompt_reset_on_temperature
        )
        progress(0, desc="Loading audio..")

        segments_result = []
        for segment in segments:
            progress(segment.start / info.duration, desc="Transcribing..")
            segments_result.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text
            })

        elapsed_time = time.time() - start_time
        return segments_result, elapsed_time

    def update_model(self,
                     model_size: str,
                     compute_type: str,
                     progress: gr.Progress
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
        self.current_model_size = self.model_paths[model_size]
        self.current_compute_type = compute_type
        self.model = faster_whisper.WhisperModel(
            device=self.device,
            model_size_or_path=self.current_model_size,
            download_root=self.model_dir,
            compute_type=self.current_compute_type
        )

    def get_model_paths(self):
        """
        Get available models from models path including fine-tuned model.

        Returns
        ----------
        Name list of models
        """
        model_paths = {model:model for model in whisper.available_models()}
        faster_whisper_prefix = "models--Systran--faster-whisper-"

        existing_models = os.listdir(self.model_dir)
        wrong_dirs = [".locks"]
        existing_models = list(set(existing_models) - set(wrong_dirs))

        webui_dir = os.getcwd()

        for model_name in existing_models:
            if faster_whisper_prefix in model_name:
                model_name = model_name[len(faster_whisper_prefix):]

            if model_name not in whisper.available_models():
                model_paths[model_name] = os.path.join(webui_dir, self.model_dir, model_name)
        return model_paths

    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "auto"

    @staticmethod
    def format_suppress_tokens_str(suppress_tokens_str: str) -> List[int]:
        try:
            suppress_tokens = ast.literal_eval(suppress_tokens_str)
            if not isinstance(suppress_tokens, list) or not all(isinstance(item, int) for item in suppress_tokens):
                raise ValueError("Invalid Suppress Tokens. The value must be type of List[int]")
            return suppress_tokens
        except Exception as e:
            raise ValueError("Invalid Suppress Tokens. The value must be type of List[int]")
