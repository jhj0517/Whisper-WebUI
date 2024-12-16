import pytest
from fastapi import UploadFile
from io import BytesIO
import os
import torch

from backend.db.task.models import TaskStatus
from backend.tests.test_task_status import wait_for_task_completion, fetch_file_response
from backend.tests.test_backend_config import (
    get_client, setup_test_file, get_upload_file_instance, calculate_wer,
    TEST_BGM_SEPARATION_PARAMS, TEST_ANSWER, TEST_BGM_SEPARATION_OUTPUT_PATH
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip the test because CUDA is not available")
@pytest.mark.parametrize(
    "bgm_separation_params",
    [
        TEST_BGM_SEPARATION_PARAMS
    ]
)
def test_transcription_endpoint(
    get_upload_file_instance,
    bgm_separation_params: dict
):
    client = get_client()
    file_content = BytesIO(get_upload_file_instance.file.read())
    get_upload_file_instance.file.seek(0)

    response = client.post(
        "/bgm-separation",
        files={"file": (get_upload_file_instance.filename, file_content, "audio/mpeg")},
        params=bgm_separation_params
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
    assert "instrumental_hash" in result and result["instrumental_hash"]
    assert "vocal_hash" in result and result["vocal_hash"]

    file_response = fetch_file_response(task_identifier)
    assert file_response.status_code == 200, f"Fetching File Response has failed. Response is: {file_response}"

    with open(TEST_BGM_SEPARATION_OUTPUT_PATH, "wb") as file:
        file.write(file_response.content)

    assert os.path.exists(TEST_BGM_SEPARATION_OUTPUT_PATH)

