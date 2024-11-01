import numpy as np
from fastapi import (
    File,
    UploadFile,
)
import gradio as gr
from fastapi import APIRouter, BackgroundTasks, Depends, Response, status

from modules.whisper.data_classes import WhisperParams
from modules.whisper.faster_whisper_inference import FasterWhisperInference
from ..util.audio import read_audio
from ..util.schemas import QueueResponse


transcription_router = APIRouter()
whisper_inferencer = FasterWhisperInference()


async def run_transcription(
    audio: np.ndarray,
    params: WhisperParams
):
    segments, elapsed_time = whisper_inferencer.transcribe(
        audio=audio,
        progress=gr.Progress(),
        *params.to_list()
    )
    return segments


@transcription_router.post("/transcription", tags=["transcription"])
async def vad(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Audio or video file to transcribe."),
    params: WhisperParams = Depends()
):
    if not isinstance(file, np.ndarray):
        audio = await read_audio(file=file)
    else:
        audio = file

    background_tasks.add_task(run_transcription, audio=audio, params=params)

    return QueueResponse(message="Transcription task queued")


