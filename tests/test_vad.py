import gradio as gr
import pytest
import os

from modules.whisper.data_classes import *
from modules.vad.silero_vad import SileroVAD
from test_config import *
from test_transcription import download_file, test_transcribe
from faster_whisper.vad import VadOptions, get_speech_timestamps


@pytest.mark.parametrize(
    "whisper_type,vad_filter,bgm_separation,diarization",
    [
        (WhisperImpl.WHISPER.value, True, False, False),
        (WhisperImpl.FASTER_WHISPER.value, True, False, False),
        (WhisperImpl.INSANELY_FAST_WHISPER.value, True, False, False)
    ]
)
def test_vad_pipeline(
    whisper_type: str,
    vad_filter: bool,
    bgm_separation: bool,
    diarization: bool,
):
    test_transcribe(whisper_type, vad_filter, bgm_separation, diarization)


@pytest.mark.parametrize(
    "threshold,min_speech_duration_ms,min_silence_duration_ms",
    [
        (0.5, 250, 2000),
    ]
)
def test_vad(
    threshold: float,
    min_speech_duration_ms: int,
    min_silence_duration_ms: int
):
    audio_path_dir = os.path.join(WEBUI_DIR, "tests")
    audio_path = os.path.join(audio_path_dir, "jfk.wav")

    if not os.path.exists(audio_path):
        download_file(TEST_FILE_DOWNLOAD_URL, audio_path_dir)

    vad_model = SileroVAD()
    vad_model.update_model()

    audio, speech_chunks = vad_model.run(
        audio=audio_path,
        vad_parameters=VadOptions(
            threshold=threshold,
            min_silence_duration_ms=min_silence_duration_ms,
            min_speech_duration_ms=min_speech_duration_ms
        )
    )

    assert speech_chunks
