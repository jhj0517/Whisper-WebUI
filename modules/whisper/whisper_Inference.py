import whisper
import gradio as gr
import time
from typing import BinaryIO, Union, Tuple, List, Callable, Optional
import numpy as np
import torch
import os
from argparse import Namespace

from modules.utils.paths import (WHISPER_MODELS_DIR, DIARIZATION_MODELS_DIR, OUTPUT_DIR, UVR_MODELS_DIR)
from modules.whisper.base_transcription_pipeline import BaseTranscriptionPipeline
from modules.whisper.data_classes import *


class WhisperInference(BaseTranscriptionPipeline):
    def __init__(self,
                 model_dir: str = WHISPER_MODELS_DIR,
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

    def transcribe(self,
                   audio: Union[str, np.ndarray, torch.Tensor],
                   progress: gr.Progress = gr.Progress(),
                   progress_callback: Optional[Callable] = None,
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
        progress_callback: Optional[Callable]
            callback function to show progress. Can be used to update progress in the backend.
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

        def progress_callback(progress_value):
            progress(progress_value, desc="Transcribing..")

        result = self.model.transcribe(audio=audio,
                                       language=params.lang,
                                       verbose=False,
                                       beam_size=params.beam_size,
                                       logprob_threshold=params.log_prob_threshold,
                                       no_speech_threshold=params.no_speech_threshold,
                                       task="translate" if params.is_translate else "transcribe",
                                       fp16=True if params.compute_type == "float16" else False,
                                       best_of=params.best_of,
                                       patience=params.patience,
                                       temperature=params.temperature,
                                       compression_ratio_threshold=params.compression_ratio_threshold,
                                       progress_callback=progress_callback,)["segments"]
        segments_result = []
        for segment in result:
            segments_result.append(Segment(
                start=segment["start"],
                end=segment["end"],
                text=segment["text"]
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
        self.current_compute_type = compute_type
        self.current_model_size = model_size
        self.model = whisper.load_model(
            name=model_size,
            device=self.device,
            download_root=self.model_dir
        )