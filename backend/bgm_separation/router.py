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
from ..util.audio import read_audio
from ..util.schemas import QueueResponse
from ..util.config_loader import load_server_config


bgm_separation_router = APIRouter()


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
    "/bgm",
    tags=["bgm-separation"],
    status_code=status.HTTP_201_CREATED,
    summary="Separate Background BGM",
    description="Separate Background Music and Vocal"
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


