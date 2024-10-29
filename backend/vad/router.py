import numpy as np
from faster_whisper.vad import VadOptions
from fastapi import (
    File,
    UploadFile,
)
from fastapi import APIRouter, BackgroundTasks, Depends, Response, status

from modules.vad.silero_vad import SileroVAD
from modules.whisper.data_classes import VadParams
from ..util.audio import read_audio
from ..util.schemas import QueueResponse


vad_router = APIRouter()
vad_inferencer = SileroVAD()


async def run_vad(
    audio: np.ndarray,
    params: VadOptions
):
    audio, speech_chunks = vad_inferencer.run(
        audio=audio,
        vad_parameters=params
    )
    return speech_chunks


@vad_router.post("/vad", tags=["vad"])
async def vad(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Audio or video file to transcribe."),
    params: VadParams = Depends()
):
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


