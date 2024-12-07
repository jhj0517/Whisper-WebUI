import functools
import numpy as np
from faster_whisper.vad import VadOptions
from fastapi import (
    File,
    UploadFile,
)
from fastapi import APIRouter, BackgroundTasks, Depends, Response, status
from typing import List, Dict
from datetime import datetime

from modules.vad.silero_vad import SileroVAD
from modules.whisper.data_classes import VadParams
from backend.common.audio import read_audio
from backend.common.models import QueueResponse
from backend.db.task.dao import add_task_to_db, update_task_status_in_db
from backend.db.task.models import TaskStatus, TaskType

vad_router = APIRouter(prefix="/vad", tags=["Voice Activity Detection"])


@functools.lru_cache
def get_vad_model() -> SileroVAD:
    inferencer = SileroVAD()
    inferencer.update_model()
    return inferencer


def run_vad(
    audio: np.ndarray,
    params: VadOptions,
    identifier: str,
) -> List[Dict]:
    update_task_status_in_db(
        identifier=identifier,
        update_data={
            "uuid": identifier,
            "status": TaskStatus.IN_PROGRESS,
            "updated_at": datetime.utcnow()
        }
    )

    start_time = datetime.utcnow()
    audio, speech_chunks = get_vad_model().run(
        audio=audio,
        vad_parameters=params
    )
    elapsed_time = (datetime.utcnow() - start_time).total_seconds()

    update_task_status_in_db(
        identifier=identifier,
        update_data={
            "uuid": identifier,
            "status": TaskStatus.COMPLETED,
            "updated_at": datetime.utcnow(),
            "result": speech_chunks,
            "duration": elapsed_time
        }
    )

    return speech_chunks


@vad_router.post(
    "/",
    response_model=QueueResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Voice Activity Detection",
    description="Detect voice parts in the provided audio or video file to generate a timeline of speech segments.",
)
async def vad(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Audio or video file to detect voices."),
    params: VadParams = Depends()
) -> QueueResponse:
    if not isinstance(file, np.ndarray):
        audio, info = await read_audio(file=file)
    else:
        audio, info = file, None

    vad_options = VadOptions(
        threshold=params.threshold,
        min_speech_duration_ms=params.min_speech_duration_ms,
        max_speech_duration_s=params.max_speech_duration_s,
        min_silence_duration_ms=params.min_silence_duration_ms,
        speech_pad_ms=params.speech_pad_ms
    )

    identifier = add_task_to_db(
        status=TaskStatus.QUEUED,
        file_name=file.filename,
        audio_duration=info.duration if info else None,
        task_type=TaskType.VAD,
        task_params=params.model_dump(),
    )

    background_tasks.add_task(run_vad, audio=audio, params=vad_options, identifier=identifier)

    return QueueResponse(identifier=identifier, status=TaskStatus.QUEUED, message="VAD task has queued")


