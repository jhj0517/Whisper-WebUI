import functools
import numpy as np
from fastapi import (
    File,
    UploadFile,
)
import gradio as gr
from fastapi import APIRouter, BackgroundTasks, Depends, Response, status
from fastapi.responses import FileResponse
from typing import List, Dict, Tuple
from datetime import datetime
import os

from modules.whisper.data_classes import *
from modules.uvr.music_separator import MusicSeparator
from modules.utils.paths import BACKEND_CACHE_DIR
from backend.common.audio import read_audio
from backend.common.models import QueueResponse
from backend.common.config_loader import load_server_config
from backend.common.compresser import get_file_hash, find_file_by_hash
from backend.db.task.models import TaskStatus, TaskType, ResultType
from backend.db.task.dao import add_task_to_db, update_task_status_in_db
from .models import BGMSeparationResult


bgm_separation_router = APIRouter(prefix="/bgm-separation", tags=["BGM Separation"])


@functools.lru_cache
def get_bgm_separation_inferencer() -> 'MusicSeparator':
    config = load_server_config()["bgm_separation"]
    inferencer = MusicSeparator(
        output_dir=os.path.join(BACKEND_CACHE_DIR, "UVR")
    )
    inferencer.update_model(
        model_name=config["model_size"],
        device=config["device"]
    )
    return inferencer


def run_bgm_separation(
    audio: np.ndarray,
    params: BGMSeparationParams,
    identifier: str,
) -> Tuple[np.ndarray, np.ndarray]:
    update_task_status_in_db(
        identifier=identifier,
        update_data={
            "uuid": identifier,
            "status": TaskStatus.IN_PROGRESS,
            "updated_at": datetime.utcnow()
        }
    )

    start_time = datetime.utcnow()
    instrumental, vocal, filepaths = get_bgm_separation_inferencer().separate(
        audio=audio,
        model_name=params.uvr_model_size,
        device=params.uvr_device,
        segment_size=params.segment_size,
        save_file=True,
        progress=gr.Progress()
    )
    instrumental_path, vocal_path = filepaths
    elapsed_time = (datetime.utcnow() - start_time).total_seconds()

    update_task_status_in_db(
        identifier=identifier,
        update_data={
            "uuid": identifier,
            "status": TaskStatus.COMPLETED,
            "result": BGMSeparationResult(
                instrumental_hash=get_file_hash(instrumental_path),
                vocal_hash=get_file_hash(vocal_path)
            ).model_dump(),
            "result_type": ResultType.FILEPATH,
            "updated_at": datetime.utcnow(),
            "duration": elapsed_time
        }
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
        audio, info = await read_audio(file=file)
    else:
        audio, info = file, None

    identifier = add_task_to_db(
        status=TaskStatus.QUEUED,
        file_name=file.filename,
        audio_duration=info.duration if info else None,
        task_type=TaskType.BGM_SEPARATION,
        task_params=params.model_dump(),
    )

    background_tasks.add_task(
        run_bgm_separation,
        audio=audio,
        params=params,
        identifier=identifier
    )

    return QueueResponse(identifier=identifier, status=TaskStatus.QUEUED, message="BGM Separation task has queued")


