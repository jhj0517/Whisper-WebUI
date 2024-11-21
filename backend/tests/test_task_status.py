import time
import pytest
from typing import Optional, Union
import httpx

from backend.db.task.models import TaskStatus, Task
from backend.tests.test_backend_config import get_client


def fetch_task(identifier: str):
    """Get task status"""
    client = get_client()
    response = client.get(
        f"/task/{identifier}"
    )
    if response.status_code == 200:
        return response
    return None


def fetch_file_response(identifier: str):
    """Get task status"""
    client = get_client()
    response = client.get(
        f"/task/file/{identifier}"
    )
    if response.status_code == 200:
        return response
    return None


def wait_for_task_completion(identifier: str,
                             max_attempts: int = 20,
                             frequency: int = 3) -> httpx.Response:
    """
    Polls the task status until it is completed, failed, or the maximum attempts are reached.

    Args:
        identifier (str): The unique identifier of the task to monitor.
        max_attempts (int): The maximum number of polling attempts..
        frequency (int): The time (in seconds) to wait between polling attempts.

    Returns:
        bool: Returns json if the task completes successfully within the allowed attempts.
    """
    attempts = 0
    while attempts < max_attempts:
        task = fetch_task(identifier)
        status = task.json()["status"]
        if status == TaskStatus.COMPLETED:
            return task
        if status == TaskStatus.FAILED:
            raise Exception("Task polling failed")
        time.sleep(frequency)
        attempts += 1
    return None
