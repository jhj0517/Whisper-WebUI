import functools
import numpy as np
from faster_whisper.vad import VadOptions
from fastapi import (
    File,
    UploadFile,
)
from fastapi import APIRouter, BackgroundTasks, Depends, Response, status
from typing import List, Dict

from modules.vad.silero_vad import SileroVAD
from modules.whisper.data_classes import VadParams
from ..common.audio import read_audio
from ..common.models import QueueResponse

vad_router = APIRouter(prefix="/vad", tags=["Voice Activity Detection"])


@functools.lru_cache
def get_vad_model() -> SileroVAD:
    inferencer = SileroVAD()
    inferencer.update_model()
    return inferencer


async def run_vad(
    audio: np.ndarray,
    params: VadOptions
) -> List[Dict]:
    audio, speech_chunks = get_vad_model().run(
        audio=audio,
        vad_parameters=params
    )
    return speech_chunks


@vad_router.post(
    "/",
    status_code=status.HTTP_201_CREATED,
    summary="Voice Activity Detection",
    description="Detect voice parts in the provided audio or video file to generate a timeline of speech segments.",
    response_model=QueueResponse,
)
async def vad(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Audio or video file to detect voices."),
    params: VadParams = Depends()
) -> QueueResponse:
    if not isinstance(file, np.ndarray):
        audio = await read_audio(file=file)
    else:
        audio = file

    vad_options = VadOptions(
        threshold=params.threshold,
        min_speech_duration_ms=params.min_speech_duration_ms,
        max_speech_duration_s=params.max_speech_duration_s,
        min_silence_duration_ms=params.min_silence_duration_ms,
        speech_pad_ms=params.speech_pad_ms
    )

    background_tasks.add_task(run_vad, audio=audio, params=vad_options)

    return QueueResponse(message="VAD task queued")


