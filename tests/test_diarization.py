import gradio as gr
import pytest
import os

from modules.utils.paths import *
from modules.whisper.whisper_factory import WhisperFactory
from modules.whisper.data_classes import *
from test_config import *
from test_transcription import download_file, run_asr_pipeline


@pytest.mark.skipif(
    not is_cuda_available(),
    reason="Skipping because the test only works on GPU"
)
@pytest.mark.parametrize(
    "whisper_type,vad_filter,bgm_separation,diarization",
    [
        (WhisperImpl.WHISPER.value, False, False, True),
        (WhisperImpl.INSANELY_FAST_WHISPER.value, False, False, True)
    ]
)
def test_diarization_pipeline(
    whisper_type: str,
    vad_filter: bool,
    bgm_separation: bool,
    diarization: bool,
):
    run_asr_pipeline(whisper_type, vad_filter, bgm_separation, diarization)

