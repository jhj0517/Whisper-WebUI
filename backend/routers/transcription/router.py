import functools
import uuid
import numpy as np
from fastapi import (
    File,
    UploadFile,
)
import gradio as gr
from fastapi import APIRouter, BackgroundTasks, Depends, Response, status
from typing import List, Dict
from sqlalchemy.orm import Session
from datetime import datetime
from modules.whisper.data_classes import *
from modules.utils.paths import BACKEND_CACHE_DIR
from modules.whisper.faster_whisper_inference import FasterWhisperInference
from backend.common.audio import read_audio
from backend.common.models import QueueResponse
from backend.common.config_loader import load_server_config
from backend.db.task.dao import (
    add_task_to_db,
    get_db_session,
    update_task_status_in_db
)
from backend.db.task.models import TaskStatus, TaskType

transcription_router = APIRouter(prefix="/transcription", tags=["Transcription"])


@functools.lru_cache
def get_pipeline() -> 'FasterWhisperInference':
    config = load_server_config()["whisper"]
    inferencer = FasterWhisperInference(
        output_dir=BACKEND_CACHE_DIR
    )
    inferencer.update_model(
        model_size=config["model_size"],
        compute_type=config["compute_type"]
    )
    return inferencer


def run_transcription(
    audio: np.ndarray,
    params: TranscriptionPipelineParams,
    identifier: str,
) -> List[Segment]:
    update_task_status_in_db(
        identifier=identifier,
        update_data={
            "uuid": identifier,
            "status": TaskStatus.IN_PROGRESS,
            "updated_at": datetime.utcnow()
        },
    )

    segments, elapsed_time = get_pipeline().run(
        audio,
        gr.Progress(),
        "SRT",
        False,
        *params.to_list()
    )
    segments = [seg.model_dump() for seg in segments]

    update_task_status_in_db(
        identifier=identifier,
        update_data={
            "uuid": identifier,
            "status": TaskStatus.COMPLETED,
            "result": segments,
            "updated_at": datetime.utcnow(),
            "duration": elapsed_time
        },
    )
    return segments


@transcription_router.post(
    "/",
    response_model=QueueResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Transcribe Audio",
    description="Process the provided audio or video file to generate a transcription.",
)
async def transcription(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Audio or video file to transcribe."),
    whisper_params: WhisperParams = Depends(),
    vad_params: VadParams = Depends(),
    bgm_separation_params: BGMSeparationParams = Depends(),
    diarization_params: DiarizationParams = Depends(),
) -> QueueResponse:
    if not isinstance(file, np.ndarray):
        audio, info = await read_audio(file=file)
    else:
        audio, info = file, None

    params = TranscriptionPipelineParams(
        whisper=whisper_params,
        vad=vad_params,
        bgm_separation=bgm_separation_params,
        diarization=diarization_params
    )

    identifier = add_task_to_db(
        status=TaskStatus.QUEUED,
        file_name=file.filename,
        audio_duration=info.duration if info else None,
        language=params.whisper.lang,
        task_type=TaskType.TRANSCRIPTION,
        task_params=params.to_dict(),
    )

    background_tasks.add_task(
        run_transcription,
        audio=audio,
        params=params,
        identifier=identifier,
    )

    return QueueResponse(identifier=identifier, status=TaskStatus.QUEUED, message="Transcription task has queued")


