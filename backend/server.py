# This mounts gradio app to fast api according to https://www.gradio.app/guides/fastapi-app-with-the-gradio-client
import os
import argparse
import json
from io import BytesIO
import faster_whisper
from fastapi import (
    File,
    HTTPException,
    Query,
    UploadFile,
    Form,
    FastAPI,
    Request,
    WebSocket,
)
from typing import Annotated, Any, BinaryIO, Literal, Generator
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
import requests

from modules.whisper.faster_whisper_inference import FasterWhisperInference


backend_app = FastAPI()
backend_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
inferencer = None


def format_stream_result(generator: Generator[dict[str, Any], Any, None]):
    for seg in generator:
        yield json.dumps({
            "seek": seg.seek,
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
            "tokens": seg.tokens
        }, ensure_ascii=False) + "\n\n"
    yield "[DONE]\n\n"


def format_json_result(
    generator: Generator[dict[str, Any], Any, None]
) -> dict[str, Any]:
    segments = []
    for seg in generator:
        segments.append({
            "seek": seg.seek,
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
            "tokens": seg.tokens
        })
    text = "\n".join([seg["text"] for seg in segments])
    return {
        "text": text,
        "segments": segments,
    }


async def read_audio(
        file: UploadFile = File(None),
        file_url: str = Form(None)
        ):
    if (file and file_url) or (not file and not file_url):
        raise HTTPException(status_code=400, detail="Provide only one of file or file_url")

    if file:
        file_content = await file.read()
    elif file_url:
        file_response = requests.get(file_url)
        if file_response.status_code != 200:
            raise HTTPException(status_code=400, detail="Could not download the file")
        file_content = file_response.content
    file_bytes = BytesIO(file_content)
    return faster_whisper.audio.decode_audio(file_bytes)


@backend_app.post("/transcription")
async def transcription(
    file: UploadFile = File(None),
    file_url: str = Form(None),
    response_format: str = Form("json"),
    model_size: str = Form("large-v2"),
    task: str = Form("transcribe"),
    language: str = Form(None),
    vad_filter: bool = Form(False),
):
    global inferencer

    if model_size != inferencer.current_model_size or inferencer.model is None:
        inferencer.update_model(model_size, "float16")
        print("model loaded")

    audio = await read_audio(file=file, file_url=file_url)

    segments, info = inferencer.model.transcribe(
        audio=audio,
    )

    if response_format == "stream":
        return StreamingResponse(
            format_stream_result(segments),
            media_type="text/event-stream",
        )
    elif response_format == "json":
        return format_json_result(segments)

    raise HTTPException(400, "Invailed response_format")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", type=str)
    parser.add_argument("--port", default=5000, type=int)
    parser.add_argument('--faster_whisper_model_dir', type=str,
                        default=os.path.join("models", "Whisper", "faster-whisper"),
                        help='Directory path of the faster-whisper model')
    parser.add_argument('--diarization_model_dir', type=str, default=os.path.join("models", "Diarization"),
                        help='Directory path of the diarization model')
    args = parser.parse_args()

    inferencer = FasterWhisperInference(
        model_dir=args.faster_whisper_model_dir,
        output_dir=os.path.join("outputs"),
        args=args
    )

    uvicorn.run(backend_app, host=args.host, port=args.port)