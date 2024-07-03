import os
import argparse
import json
from io import BytesIO
import numpy as np
import faster_whisper
from faster_whisper.vad import VadOptions
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
from scipy.io.wavfile import write
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
import requests
import io

from modules.whisper.faster_whisper_inference import FasterWhisperInference
from modules.utils.logger import get_logger
from modules.vad.silero_vad import SileroVAD
from modules.diarize.diarizer import Diarizer
from modules.diarize.audio_loader import SAMPLE_RATE


backend_app = FastAPI()
backend_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
vad_inferencer = None
whisper_inferencer = None
diarization_inferencer = None
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
    segments: Union[Generator[dict[str, Any], Any, None], List[dict]]
) -> dict[str, Any]:
    result = []
    for seg in segments:
        result.append({
            "seek": seg.seek,
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
            "tokens": seg.tokens
        })
    text = "\n".join([seg["text"] for seg in result])
    return {
        "text": text,
        "segments": result,
    }


async def read_audio(
    file: Optional[UploadFile] = None,
    file_url: Optional[str] = None
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


@backend_app.post("/vad")
async def vad(
    file: UploadFile = File(None),
    threshold: float = Form(0.5),
    min_speech_duration_ms: int = Form(250),
    max_speech_duration_s: Optional[int] = Form(999),
    min_silence_duration_ms: int = Form(2000),
    window_size_samples: int = Form(1024),
    speech_pad_ms: int = Form(400)
):
    global vad_inferencer

    if not isinstance(file, np.ndarray):
        audio = await read_audio(file=file)
    else:
        audio = file

    vad_options = VadOptions(
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        max_speech_duration_s=max_speech_duration_s,
        min_silence_duration_ms=min_silence_duration_ms,
        window_size_samples=window_size_samples,
        speech_pad_ms=speech_pad_ms
    )

    preprocessed_audio = vad_inferencer.run(
        audio=audio,
        vad_parameters=vad_options
    )

    audio_output = io.BytesIO()
    write(audio_output, SAMPLE_RATE, preprocessed_audio)
    audio_output.seek(0)
    return StreamingResponse(audio_output, media_type="audio/wav")


@backend_app.post("/diarization")
async def diarization(
    file: UploadFile = File(None),
    use_auth_token: str = Form(None),
    transcript: Optional[List[dict]] = None,
):
    global diarization_inferencer
    global whisper_inferencer

    if not isinstance(file, np.ndarray):
        audio = await read_audio(file=file)
    else:
        audio = file

    if transcript is None:
        generator, info = whisper_inferencer.model.transcribe(
            audio=audio
        )
        transcript = format_json_result(generator)["segments"]

    diarized_transcript, elapsed_time = diarization_inferencer.run(
        audio=audio,
        transcribed_result=transcript,
        use_auth_token=use_auth_token,
        device=diarization_inferencer.device
    )
    return diarized_transcript


@backend_app.post("/transcription")
async def transcription(
    file: UploadFile = File(default=None, description="Input file for video or audio. This will be pre-processed with ffmpeg in the server"),
    file_url: str = Form(default=None, description="Input file url for video or audio. You need to provide either the file or the file URL, but not both"),
    response_format: str = Form(default="json"),
    model_size: str = Form(default="large-v2"),
    language: Optional[str] = Form(default=None),
    task: str = Form(default="transcribe"),
    beam_size: int = Form(default=5),
    best_of: int = Form(default=5),
    patience: float = Form(default=1),
    length_penalty: float = Form(default=1),
    repetition_penalty: float = Form(default=1),
    no_repeat_ngram_size: int = Form(default=0),
    temperature: Union[float, List[float], Tuple[float, ...]] = Form(default=0.0),
    compression_ratio_threshold: Optional[float] = Form(default=2.4),
    log_prob_threshold: Optional[float] = Form(default=-1.0),
    no_speech_threshold: Optional[float] = Form(default=0.6),
    condition_on_previous_text: bool = Form(default=True),
    prompt_reset_on_temperature: float = Form(default=0.5),
    initial_prompt: Optional[Union[str, List[int]]] = Form(default=None),
    prefix: Optional[str] = Form(default=None),
    suppress_blank: bool = Form(default=True),
    suppress_tokens: Optional[List[int]] = Form(default=None),
    without_timestamps: bool = Form(default=False),
    max_initial_timestamp: float = Form(default=1.0),
    word_timestamps: bool = Form(default=False),
    prepend_punctuations: str = Form(default="\"'“¿([{-"),
    append_punctuations: str = Form(default="\"'.。,，!！?？:：”)]}、"),
    max_new_tokens: Optional[int] = Form(default=None),
    chunk_length: Optional[int] = Form(default=None),
    clip_timestamps: Union[str, List[float]] = Form(default="0"),
    hallucination_silence_threshold: Optional[float] = Form(default=None),
    hotwords: Optional[str] = Form(default=None),
    language_detection_threshold: Optional[float] = Form(default=None),
    language_detection_segments: int = Form(default=1),

    vad_filter: bool = Form(default=False),
    threshold: float = Form(default=0.5),
    min_speech_duration_ms: int = Form(default=250),
    max_speech_duration_s: Optional[int] = Form(default=999),
    min_silence_duration_ms: int = Form(default=2000),
    window_size_samples: int = Form(default=1024),
    speech_pad_ms: int = Form(default=400),

    is_diarization: bool = Form(default=False),
    use_auth_token: str = Form(default=None),
):
    global whisper_inferencer
    global vad_inferencer
    global diarization_inferencer

    if model_size != whisper_inferencer.current_model_size or whisper_inferencer.model is None:
        whisper_inferencer.update_model(model_size, whisper_inferencer.current_compute_type)
        logger.info("Model loaded")

    audio = await read_audio(file=file, file_url=file_url)

    if vad_filter:
        vad_options = VadOptions(
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            max_speech_duration_s=max_speech_duration_s,
            min_silence_duration_ms=min_silence_duration_ms,
            window_size_samples=window_size_samples,
            speech_pad_ms=speech_pad_ms
        )
        audio = vad_inferencer.run(
            audio=audio,
            vad_parameters=vad_options
        )

    segments, info = whisper_inferencer.model.transcribe(
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

    if is_diarization:
        segments = [seg for seg in segments]
        segments = diarization_inferencer.run(
            audio=audio,
            transcribed_result=segments,
            use_auth_token=use_auth_token,
            device=diarization_inferencer.device
        )

    elif response_format == "json":
        return format_json_result(segments)

    raise HTTPException(400, "Invalid response_format")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", type=str)
    parser.add_argument("--port", default=5000, type=int)
    parser.add_argument("--device", type=str, help="Device for the whisper models between ['cuda', 'cpu', 'auto']. It will use cuda if it's enabled by default")
    parser.add_argument("--diarization_device", type=str,
                        help="Device for the diarization models between ['cuda', 'cpu', 'mps']. It will use cuda if it's enabled by default")
    parser.add_argument("--initial_model", type=str,
                        default="large-v2", help="The whisper model to load initially when server start")
    parser.add_argument('--faster_whisper_model_dir', type=str,
                        default=os.path.join("models", "Whisper", "faster-whisper"),
                        help='Directory path of the faster-whisper model')
    parser.add_argument('--diarization_model_dir', type=str, default=os.path.join("models", "Diarization"),
                        help='Directory path of the diarization model')

    args = parser.parse_args()

    whisper_inferencer = FasterWhisperInference(
        model_dir=args.faster_whisper_model_dir,
        output_dir=os.path.join("outputs"),
        args=args
    )

    if not (args.initial_model in whisper_inferencer.available_models):
        raise HTTPException(400, f"The initial model you set \"{args.initial_model}\" is not available.")
    if args.device is not None:
        whisper_inferencer.device = args.device

    whisper_inferencer.update_model(model_size="large-v2", compute_type=whisper_inferencer.current_compute_type)
    vad_inferencer = SileroVAD()
    diarization_inferencer = Diarizer(
        model_dir=args.diarization_model_dir
    )
    if args.diarization_device is not None:
        diarization_inferencer.device = args.diarization_device

    uvicorn.run(backend_app, host=args.host, port=args.port)