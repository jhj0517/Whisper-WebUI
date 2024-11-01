import functools

import numpy as np
from fastapi import (
    File,
    UploadFile,
)
import gradio as gr
from fastapi import APIRouter, BackgroundTasks, Depends, Response, status
from typing import List, Dict

from modules.whisper.data_classes import *
from modules.whisper.faster_whisper_inference import FasterWhisperInference
from ..util.audio import read_audio
from ..util.schemas import QueueResponse
from ..util.config_loader import load_server_config

@functools.lru_cache
def init_pipeline() -> 'FasterWhisperInference':
    config = load_server_config()["whisper"]
    transcription_pipeline = FasterWhisperInference()
    transcription_pipeline.update_model(
        model_size=config["model_size"],
        compute_type=config["compute_type"]
    )
    return transcription_pipeline

transcription_router = APIRouter()
transcription_pipeline = init_pipeline()


async def run_transcription(
    audio: np.ndarray,
    params: TranscriptionPipelineParams
) -> List[Segment]:
    segments, elapsed_time = transcription_pipeline.run(
        audio=audio,
        progress=gr.Progress(),
        add_timestamp=False,
        *params.to_list()
    )
    return segments


@transcription_router.post("/transcription", tags=["transcription"])
async def vad(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Audio or video file to transcribe."),
    params: TranscriptionPipelineParams = Depends()
) -> QueueResponse:
    if not isinstance(file, np.ndarray):
        audio = await read_audio(file=file)
    else:
        audio = file

    background_tasks.add_task(run_transcription, audio=audio, params=params)

    return QueueResponse(message="Transcription task queued")


