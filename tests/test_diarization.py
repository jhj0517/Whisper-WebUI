from modules.utils.paths import *
from modules.whisper.whisper_factory import WhisperFactory
from modules.whisper.whisper_parameter import WhisperValues
from test_config import *
from test_transcription import download_file, test_transcribe

import gradio as gr
import pytest
import os


@pytest.mark.parametrize(
    "whisper_type,vad_filter,bgm_separation,diarization",
    [
        ("whisper", False, False, True),
        ("faster-whisper", False, False, True),
        ("insanely_fast_whisper", False, False, True)
    ]
)
def test_diarization_pipeline(
    whisper_type: str,
    vad_filter: bool,
    bgm_separation: bool,
    diarization: bool,
):
    test_transcribe(whisper_type, vad_filter, bgm_separation, diarization)

