from modules.whisper.whisper_factory import WhisperFactory
from modules.whisper.whisper_parameter import WhisperValues
from test_config import *

import pytest
import gradio as gr
import os


@pytest.mark.parametrize("whisper_type", ["whisper", "faster-whisper", "insanely_fast_whisper"])
def test_transcribe(whisper_type: str):
    audio_path = os.path.join("test.wav")
    if not os.path.exists(audio_path):
        download_file(TEST_FILE_DOWNLOAD_URL, audio_path)

    whisper_inferencer = WhisperFactory.create_whisper_inference(
        whisper_type=whisper_type,
    )

    print("Device : ", whisper_inferencer.device)

    hparams = WhisperValues(
        model_size=TEST_WHISPER_MODEL,
    ).as_list()

    whisper_inferencer.transcribe_file(
        files=[audio_path],
        progress=gr.Progress(),
        *hparams,
    )

    whisper_inferencer.transcribe_youtube(
        youtube_link=TEST_YOUTUBE_URL,
        progress=gr.Progress(),
        *hparams,
    )

    whisper_inferencer.transcribe_mic(
        mic_audio=audio_path,
        progress=gr.Progress(),
        *hparams,
    )


def download_file(url: str, path: str):
    if not os.path.exists(path):
        os.system(f"wget {url} -O {path}")
