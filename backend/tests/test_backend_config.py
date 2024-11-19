import functools
from fastapi import FastAPI, UploadFile
from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile as StarletteUploadFile
from io import BytesIO
import os
import requests
import pytest
import yaml

from backend.main import backend_app
from modules.whisper.data_classes import *
from modules.utils.paths import *
from modules.utils.files_manager import load_yaml, save_yaml

TEST_PIPELINE_PARAMS = {**WhisperParams(model_size="tiny").model_dump(exclude_none=True),
                        **VadParams().model_dump(exclude_none=True),
                        **BGMSeparationParams().model_dump(exclude_none=True),
                        **DiarizationParams().model_dump(exclude_none=True)}
TEST_VAD_PARAMS = VadParams()
TEST_BGM_SEPARATION_PARAMS = BGMSeparationParams()
TEST_FILE_DOWNLOAD_URL = "https://github.com/jhj0517/whisper_flutter_new/raw/main/example/assets/jfk.wav"
TEST_FILE_PATH = os.path.join(WEBUI_DIR, "backend", "tests", "jfk.wav")
TEST_WHISPER_MODEL = "tiny"
TEST_COMPUTE_TYPE = "float32"


@pytest.fixture(autouse=True)
@functools.lru_cache
def setup_test_file():
    def download_file(url=TEST_FILE_DOWNLOAD_URL, file_path=TEST_FILE_PATH):
        if os.path.exists(file_path):
            return

        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))

        response = requests.get(url)

        with open(file_path, "wb") as file:
            file.write(response.content)

        print(f"File downloaded to: {file_path}")

    download_file(TEST_FILE_DOWNLOAD_URL, TEST_FILE_PATH)

    server_config = load_yaml(SERVER_CONFIG_PATH)
    server_config["whisper"]["model_size"] = TEST_WHISPER_MODEL
    server_config["whisper"]["compute_type"] = TEST_COMPUTE_TYPE
    save_yaml(server_config, SERVER_CONFIG_PATH)


@pytest.fixture
def upload_file_instance(filepath: str = TEST_FILE_PATH) -> UploadFile:
    with open(filepath, "rb") as f:
        file_contents = BytesIO(f.read())
        filename = os.path.basename(filepath)
        upload_file = StarletteUploadFile(file=file_contents, filename=filename)
    return upload_file


@functools.lru_cache
def get_client(app: FastAPI = backend_app):
    return TestClient(app)



