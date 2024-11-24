# Whisper-WebUI REST API
REST API for Whisper-WebUI. 
Documentation is available via [redoc](https://github.com/Redocly/redoc) or root URL with redirection.
# Setup and Installation

This assumes that you are in the root directory of Whisper-WebUI

1. Create `.env` in `backend/configs/.env`
```
HF_TOKEN="YOUR_HF_TOKEN FOR DIARIZATION MODEL (READ PERMISSION)"
DB_URL="sqlite:///backend/records.db"
```
`HF_TOKEN` is used for diarization model, `DB_URL` indicates where your db file is located. It is stored in `backend/` by default.

2. Install dependency
```
pip install -r backend/requirements-backend.txt
```

3. Deploy the server with `uvicorn` or whatever. 
```
uvicorn backend.main:app --host 0.0.0.0 --port 8000
``` 

# Architecture

![diagram](https://github.com/user-attachments/assets/37d2ab2d-4eb4-4513-bb7b-027d0d631971)

The response can be obtained through [the polling API](https://docs.oracle.com/en/cloud/saas/marketing/responsys-develop/API/REST/Async/asyncApi-v1.3-requests-requestId-get.htm).
Each task is stored in the DB whenever the task is queued or updated by the process.

When the client first sends the `POST` request, the server returns an `identifier` to the client that can be used to track the status of the task. The task status is updated by the processes, and once the task is completed,  the client can finally obtain the result.

The client needs to implement manual API polling to do this, this is the example for the python:
```python
def wait_for_task_completion(identifier: str,
                             max_attempts: int = 20,
                             frequency: int = 3) -> httpx.Response:
    """
    Polls the task status every `frequency` until it is completed, failed, or the `max_attempts` are reached.
    """
    attempts = 0
    while attempts < max_attempts:
        task = fetch_task(identifier)
        status = task.json()["status"]
        if status == "COMPLETED":
            return task["result"]
        if status == "FAILED":
            raise Exception("Task polling failed")
        time.sleep(frequency)
        attempts += 1
    return None
```
