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
from typing import Annotated, Any, BinaryIO, Literal, Generator, Union, Optional, List, Tuple
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
import requests

from modules.whisper.faster_whisper_inference import FasterWhisperInference
from modules.utils.logger import get_logger


backend_app = FastAPI()
backend_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
inferencer = None
logger = get_logger("Whisper-WebUI-Backend")


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
            raise HTTPException(status_code=422, detail="Could not download the file")
        file_content = file_response.content
    file_bytes = BytesIO(file_content)
    return faster_whisper.audio.decode_audio(file_bytes)


@backend_app.post("/transcription")
async def transcription(
    file: UploadFile = File(default=None, description="Input file for video or audio. This will be pre-processed with ffmpeg in the server"),
    file_url: str = Form(default=None, description="Input file url for video or audio. You have to provide only one of file or file URL"),
    response_format: str = Form("json"),
    model_size: str = Form("large-v2"),
    language: Optional[str] = Form(None),
    task: str = Form("transcribe"),
    beam_size: int = Form(5),
    best_of: int = Form(5),
    patience: float = Form(1),
    length_penalty: float = Form(1),
    repetition_penalty: float = Form(1),
    no_repeat_ngram_size: int = Form(0),
    temperature: Union[float, List[float], Tuple[float, ...]] = Form(0.0),
    compression_ratio_threshold: Optional[float] = Form(2.4),
    log_prob_threshold: Optional[float] = Form(-1.0),
    no_speech_threshold: Optional[float] = Form(0.6),
    condition_on_previous_text: bool = Form(True),
    prompt_reset_on_temperature: float = Form(0.5),
    initial_prompt: Optional[Union[str, List[int]]] = Form(None),
    prefix: Optional[str] = Form(None),
    suppress_blank: bool = Form(True),
    suppress_tokens: Optional[List[int]] = Form(None),
    without_timestamps: bool = Form(False),
    max_initial_timestamp: float = Form(1.0),
    word_timestamps: bool = Form(False),
    prepend_punctuations: str = Form("\"'“¿([{-"),
    append_punctuations: str = Form("\"'.。,，!！?？:：”)]}、"),
    vad_filter: bool = Form(False),
    vad_parameters: Optional[Union[dict, str]] = Form(None),
    max_new_tokens: Optional[int] = Form(None),
    chunk_length: Optional[int] = Form(None),
    clip_timestamps: Union[str, List[float]] = Form("0"),
    hallucination_silence_threshold: Optional[float] = Form(None),
    hotwords: Optional[str] = Form(None),
    language_detection_threshold: Optional[float] = Form(None),
    language_detection_segments: int = Form(1)
):
    global inferencer

    if model_size != inferencer.current_model_size or inferencer.model is None:
        inferencer.update_model(model_size, inferencer.current_compute_type)
        logger.info("Model loaded")
    audio = await read_audio(file=file, file_url=file_url)
    segments, info = inferencer.model.transcribe(
        audio=audio,
        language=language,
        task=task,
        beam_size=beam_size,
        best_of=best_of,
        patience=patience,
        length_penalty=length_penalty,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        temperature=temperature,
        compression_ratio_threshold=compression_ratio_threshold,
        log_prob_threshold=log_prob_threshold,
        no_speech_threshold=no_speech_threshold,
        condition_on_previous_text=condition_on_previous_text,
        prompt_reset_on_temperature=prompt_reset_on_temperature,
        initial_prompt=initial_prompt,
        prefix=prefix,
        suppress_blank=suppress_blank,
        suppress_tokens=suppress_tokens,
        without_timestamps=without_timestamps,
        max_initial_timestamp=max_initial_timestamp,
        word_timestamps=word_timestamps,
        prepend_punctuations=prepend_punctuations,
        append_punctuations=append_punctuations,
        vad_filter=vad_filter,
        vad_parameters=vad_parameters,
        max_new_tokens=max_new_tokens,
        chunk_length=chunk_length,
        clip_timestamps=clip_timestamps,
        hallucination_silence_threshold=hallucination_silence_threshold,
        hotwords=hotwords,
        language_detection_threshold=language_detection_threshold,
        language_detection_segments=language_detection_segments
    )

    if response_format == "stream":
        return StreamingResponse(
            format_stream_result(segments),
            media_type="text/event-stream",
        )
    elif response_format == "json":
        return format_json_result(segments)

    raise HTTPException(400, "Invalid response_format")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", type=str)
    parser.add_argument("--port", default=5000, type=int)
    parser.add_argument("--device", type=str, help="Device between ['cuda', 'cpu', 'auto']. It will use cuda if it's enabled by default")
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
    inferencer.update_model(model_size="large-v2", compute_type=inferencer.current_compute_type)
    if args.device is not None:
        inferencer.device = args.device

    uvicorn.run(backend_app, host=args.host, port=args.port)