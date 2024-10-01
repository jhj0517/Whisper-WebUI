from modules.utils.paths import *

import os

TEST_FILE_DOWNLOAD_URL = "https://github.com/jhj0517/whisper_flutter_new/raw/main/example/assets/jfk.wav"
TEST_FILE_PATH = os.path.join(WEBUI_DIR, "tests", "jfk.wav")
TEST_YOUTUBE_URL = "https://www.youtube.com/watch?v=4WEQtgnBu0I&ab_channel=AndriaFitzer"
TEST_WHISPER_MODEL = "tiny"
TEST_UVR_MODEL = "UVR-MDX-NET-Inst_HQ_4"
TEST_NLLB_MODEL = "facebook/nllb-200-distilled-600M"
TEST_SUBTITLE_SRT_PATH = os.path.join(WEBUI_DIR, "tests", "test_srt.srt")
TEST_SUBTITLE_VTT_PATH = os.path.join(WEBUI_DIR, "tests", "test_vtt.vtt")

