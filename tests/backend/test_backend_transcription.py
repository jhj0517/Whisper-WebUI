import pytest
from fastapi import UploadFile
from io import BytesIO

from modules.whisper.data_classes import *
from backend.db.task.models import TaskStatus
from test_backend_config import get_client, TEST_UPLOAD_FILE, TEST_PIPELINE_PARAMS


@pytest.mark.parametrize(
    "upload_file,pipeline_params",
    [
        (TEST_UPLOAD_FILE, TEST_PIPELINE_PARAMS)
    ]
)
@pytest.mark.asyncio
async def test_transcription_endpoint(
    upload_file: UploadFile,
    pipeline_params: dict
):
    file_content = BytesIO(upload_file.file.read())
    upload_file.file.seek(0)

    client = get_client()
    response = client.post(
        "/transcription",
        files={"file": (upload_file.filename, file_content, "audio/mpeg")},
        params=pipeline_params
    )

    assert response.status_code == 201
    assert response.json()["status"] == TaskStatus.QUEUED
    assert response.json()["message"] == "Transcription task has queued"
