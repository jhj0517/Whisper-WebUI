from modules.whisper.whisper_factory import WhisperFactory
from modules.whisper.data_classes import *
from modules.utils.paths import WEBUI_DIR
from test_config import *

import requests
import pytest
import gradio as gr
import os


@pytest.mark.parametrize(
    "whisper_type,vad_filter,bgm_separation,diarization",
    [
        ("whisper", False, False, False),
        ("faster-whisper", False, False, False),
        ("insanely_fast_whisper", False, False, False)
    ]
)
def test_transcribe(
    whisper_type: str,
    vad_filter: bool,
    bgm_separation: bool,
    diarization: bool,
):
    audio_path_dir = os.path.join(WEBUI_DIR, "tests")
    audio_path = os.path.join(audio_path_dir, "jfk.wav")
    if not os.path.exists(audio_path):
        download_file(TEST_FILE_DOWNLOAD_URL, audio_path_dir)

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

    subtitle_str, file_path = whisper_inferencer.transcribe_file(
        [audio_path],
        None,
        "SRT",
        False,
        gr.Progress(),
        *hparams,
    )

    assert isinstance(subtitle_str, str) and subtitle_str
    assert isinstance(file_path[0], str) and file_path

    whisper_inferencer.transcribe_youtube(
        TEST_YOUTUBE_URL,
        "SRT",
        False,
        gr.Progress(),
        *hparams,
    )
    assert isinstance(subtitle_str, str) and subtitle_str
    assert isinstance(file_path[0], str) and file_path

    whisper_inferencer.transcribe_mic(
        audio_path,
        "SRT",
        False,
        gr.Progress(),
        *hparams,
    )
    assert isinstance(subtitle_str, str) and subtitle_str
    assert isinstance(file_path[0], str) and file_path


def download_file(url, save_dir):
    if os.path.exists(TEST_FILE_PATH):
        return

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_name = url.split("/")[-1]
    file_path = os.path.join(save_dir, file_name)

    response = requests.get(url)

    with open(file_path, "wb") as file:
        file.write(response.content)

    print(f"File downloaded to: {file_path}")
