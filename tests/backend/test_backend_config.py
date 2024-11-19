import functools
from fastapi import FastAPI, UploadFile
from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile as StarletteUploadFile
from io import BytesIO
import os
import requests

from backend.main import backend_app
from modules.whisper.data_classes import *
from modules.utils.paths import *

TEST_PIPELINE_PARAMS = {**WhisperParams().model_dump(exclude_none=True), **VadParams().model_dump(exclude_none=True),
                        **BGMSeparationParams().model_dump(exclude_none=True),
                        **DiarizationParams().model_dump(exclude_none=True)}
TEST_VAD_PARAMS = VadParams()
TEST_BGM_SEPARATION_PARAMS = BGMSeparationParams()
TEST_FILE_PATH = os.path.join(WEBUI_DIR, "tests", "jfk.wav")


@functools.lru_cache
def upload_file_instance(filepath: str = TEST_FILE_PATH) -> UploadFile:
    with open(filepath, "rb") as f:
        file_contents = BytesIO(f.read())
        filename = os.path.basename(filepath)
        upload_file = StarletteUploadFile(file=file_contents, filename=filename)
    return upload_file


TEST_UPLOAD_FILE = upload_file_instance()


@functools.lru_cache
def get_client(app: FastAPI = backend_app):
    return TestClient(app)



