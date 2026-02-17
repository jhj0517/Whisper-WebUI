import functools
import jiwer
import os
import pytest
import requests
import torch

from modules.utils.paths import *
from modules.utils.youtube_manager import *

TEST_FILE_DOWNLOAD_URL = "https://github.com/jhj0517/whisper_flutter_new/raw/main/example/assets/jfk.wav"
TEST_FILE_PATH = os.path.join(WEBUI_DIR, "tests", "jfk.wav")
TEST_ANSWER = "And so my fellow Americans ask not what your country can do for you ask what you can do for your country"
TEST_YOUTUBE_URL = "https://www.youtube.com/watch?v=4WEQtgnBu0I&ab_channel=AndriaFitzer"
TEST_WHISPER_MODEL = "tiny"
TEST_UVR_MODEL = "UVR-MDX-NET-Inst_HQ_4"
TEST_NLLB_MODEL = "facebook/nllb-200-distilled-600M"
TEST_SUBTITLE_SRT_PATH = os.path.join(WEBUI_DIR, "tests", "test_srt.srt")
TEST_SUBTITLE_VTT_PATH = os.path.join(WEBUI_DIR, "tests", "test_vtt.vtt")


@functools.lru_cache
def is_xpu_available():
    return torch.xpu.is_available()


@functools.lru_cache
def is_cuda_available():
    return torch.cuda.is_available()


@functools.lru_cache
def is_pytube_detected_bot(url: str = TEST_YOUTUBE_URL):
    try:
        yt_temp_path = os.path.join("modules", "yt_tmp.wav")
        if os.path.exists(yt_temp_path):
            return False
        yt = get_ytdata(url)
        audio = get_ytaudio(yt)
        return False
    except Exception as e:
        print(f"Pytube has detected as a bot: {e}")
        return True


@pytest.fixture(autouse=True)
def download_file(url=TEST_FILE_DOWNLOAD_URL, file_path=TEST_FILE_PATH):
    if os.path.exists(file_path):
        return

    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))

    response = requests.get(url)

    with open(file_path, "wb") as file:
        file.write(response.content)

    print(f"File downloaded to: {file_path}")


def calculate_wer(answer, prediction):
    return jiwer.wer(answer, prediction)
