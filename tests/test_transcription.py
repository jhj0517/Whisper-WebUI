from modules.whisper.whisper_factory import WhisperFactory
from modules.whisper.data_classes import *
from modules.utils.subtitle_manager import read_file
from modules.utils.paths import WEBUI_DIR
from test_config import *

import requests
import pytest
import gradio as gr
import os


@pytest.mark.parametrize(
    "whisper_type,vad_filter,bgm_separation,diarization",
    [
        (WhisperImpl.WHISPER.value, False, False, False),
        (WhisperImpl.FASTER_WHISPER.value, False, False, False),
        (WhisperImpl.INSANELY_FAST_WHISPER.value, False, False, False)
    ]
)
def test_transcribe(
    whisper_type: str,
    vad_filter: bool,
    bgm_separation: bool,
    diarization: bool,
):
    audio_path = TEST_FILE_PATH

    answer = TEST_ANSWER
    if diarization:
        answer = "SPEAKER_00|"+TEST_ANSWER

    whisper_inferencer = WhisperFactory.create_whisper_inference(
        whisper_type=whisper_type,
    )
    print(
        f"""Whisper Device : {whisper_inferencer.device}\n"""
        f"""BGM Separation Device: {whisper_inferencer.music_separator.device}\n"""
        f"""Diarization Device: {whisper_inferencer.diarizer.device}"""
    )

    hparams = TranscriptionPipelineParams(
        whisper=WhisperParams(
            model_size=TEST_WHISPER_MODEL,
            compute_type=whisper_inferencer.current_compute_type
        ),
        vad=VadParams(
            vad_filter=vad_filter
        ),
        bgm_separation=BGMSeparationParams(
            is_separate_bgm=bgm_separation,
            enable_offload=True
        ),
        diarization=DiarizationParams(
            is_diarize=diarization
        ),
    ).to_list()

    subtitle_str, file_paths = whisper_inferencer.transcribe_file(
        [audio_path],
        None,
        None,
        None,
        "SRT",
        False,
        gr.Progress(),
        *hparams,
    )
    subtitle = read_file(file_paths[0]).split("\n")
    assert calculate_wer(answer, subtitle[2].strip().replace(",", "").replace(".", "")) < 0.1

    if not is_pytube_detected_bot():
        subtitle_str, file_path = whisper_inferencer.transcribe_youtube(
            TEST_YOUTUBE_URL,
            "SRT",
            False,
            gr.Progress(),
            *hparams,
        )
        assert isinstance(subtitle_str, str) and subtitle_str
        assert os.path.exists(file_path)

    subtitle_str, file_path = whisper_inferencer.transcribe_mic(
        audio_path,
        "SRT",
        False,
        gr.Progress(),
        *hparams,
    )
    subtitle = read_file(file_path).split("\n")
    wer = calculate_wer(answer, subtitle[2].strip().replace(",", "").replace(".", ""))
    assert wer < 0.1, f"WER is too high, it's {wer}"


