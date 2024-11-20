import pytest
from fastapi import UploadFile
from io import BytesIO

from backend.db.task.models import TaskStatus
from backend.tests.test_task_status import wait_for_task_completion
from backend.tests.test_backend_config import (
    get_client, setup_test_file, get_upload_file_instance, calculate_wer,
    TEST_VAD_PARAMS, TEST_ANSWER
)


@pytest.mark.parametrize(
    "vad_params",
    [
        TEST_VAD_PARAMS
    ]
)
def test_transcription_endpoint(
    get_upload_file_instance,
    vad_params: dict
):
    client = get_client()
    file_content = BytesIO(get_upload_file_instance.file.read())
    get_upload_file_instance.file.seek(0)

    response = client.post(
        "/vad",
        files={"file": (get_upload_file_instance.filename, file_content, "audio/mpeg")},
        params=vad_params
    )

    assert response.status_code == 201
    assert response.json()["status"] == TaskStatus.QUEUED
    task_identifier = response.json()["identifier"]
    assert isinstance(task_identifier, str) and task_identifier

    completed_task = wait_for_task_completion(
        identifier=task_identifier
    )

    assert completed_task is not None, f"Task with identifier {task_identifier} did not complete within the " \
                                       f"expected time."

    result = completed_task.json()["result"]
    assert result and "start" in result[0] and "end" in result[0]

