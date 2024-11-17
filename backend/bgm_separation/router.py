import functools
import numpy as np
from fastapi import (
    File,
    UploadFile,
)
import gradio as gr
from fastapi import APIRouter, BackgroundTasks, Depends, Response, status
from typing import List, Dict, Tuple

from modules.whisper.data_classes import *
from modules.uvr.music_separator import MusicSeparator
from ..common.audio import read_audio
from ..common.models import QueueResponse
from ..common.config_loader import load_server_config


bgm_separation_router = APIRouter(prefix="bgm-separation", tags=["BGM Separation"])


@functools.lru_cache
def get_bgm_separation_inferencer() -> 'MusicSeparator':
    config = load_server_config()["bgm_separation"]
    inferencer = MusicSeparator()
    inferencer.update_model(
        model_name=config["model_size"],
        device=config["compute_type"]
    )
    return inferencer


async def run_bgm_separation(
    audio: np.ndarray,
    params: BGMSeparationParams
) -> Tuple[np.ndarray, np.ndarray]:
    instrumental, vocal, filepaths = get_bgm_separation_inferencer().separate(
        audio=audio,
        model_name=params.model_size,
        device=params.device,
        segment_size=params.segment_size,
        save_file=False,
        progress=gr.Progress()
    )
    return instrumental, vocal


@bgm_separation_router.post(
    "/",
    response_model=QueueResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Separate Background BGM abd vocal",
    description="Separate background music and vocal from an uploaded audio or video file.",
)
async def bgm_separation(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Audio or video file to separate background music."),
    params: BGMSeparationParams = Depends()
) -> QueueResponse:
    if not isinstance(file, np.ndarray):
        audio = await read_audio(file=file)
    else:
        audio = file

    background_tasks.add_task(run_bgm_separation, audio=audio, params=params)

    return QueueResponse(message="Transcription task queued")


