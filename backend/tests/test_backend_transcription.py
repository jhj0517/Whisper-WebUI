import pytest
from fastapi import UploadFile
from io import BytesIO

from backend.db.task.models import TaskStatus
from backend.tests.test_backend_config import get_client, setup_test_file, upload_file_instance, TEST_PIPELINE_PARAMS


@pytest.mark.parametrize(
    "pipeline_params",
    [
        TEST_PIPELINE_PARAMS
    ]
)
@pytest.mark.asyncio
async def test_transcription_endpoint(
    upload_file_instance,
    pipeline_params: dict
):
    file_content = BytesIO(upload_file_instance.file.read())
    upload_file_instance.file.seek(0)

    client = get_client()
    response = client.post(
        "/transcription",
        files={"file": (upload_file_instance.filename, file_content, "audio/mpeg")},
        params=pipeline_params
    )

    assert response.status_code == 201
    assert response.json()["status"] == TaskStatus.QUEUED
    assert response.json()["message"] == "Transcription task has queued"
