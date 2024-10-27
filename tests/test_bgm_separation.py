from modules.utils.paths import *
from modules.whisper.whisper_factory import WhisperFactory
from modules.whisper.data_classes import WhisperValues
from test_config import *
from test_transcription import download_file, test_transcribe

import gradio as gr
import pytest
import torch
import os


@pytest.mark.skipif(
    not is_cuda_available(),
    reason="Skipping because the test only works on GPU"
)
@pytest.mark.parametrize(
    "whisper_type,vad_filter,bgm_separation,diarization",
    [
        ("whisper", False, True, False),
        ("faster-whisper", False, True, False),
        ("insanely_fast_whisper", False, True, False)
    ]
)
def test_bgm_separation_pipeline(
    whisper_type: str,
    vad_filter: bool,
    bgm_separation: bool,
    diarization: bool,
):
    test_transcribe(whisper_type, vad_filter, bgm_separation, diarization)


@pytest.mark.skipif(
    not is_cuda_available(),
    reason="Skipping because the test only works on GPU"
)
@pytest.mark.parametrize(
    "whisper_type,vad_filter,bgm_separation,diarization",
    [
        ("whisper", True, True, False),
        ("faster-whisper", True, True, False),
        ("insanely_fast_whisper", True, True, False)
    ]
)
def test_bgm_separation_with_vad_pipeline(
    whisper_type: str,
    vad_filter: bool,
    bgm_separation: bool,
    diarization: bool,
):
    test_transcribe(whisper_type, vad_filter, bgm_separation, diarization)

