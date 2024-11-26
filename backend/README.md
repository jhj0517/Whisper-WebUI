# Whisper-WebUI REST API
REST API for Whisper-WebUI. Documentation is auto-generated upon deploying the app.
<br>[Swagger UI](https://github.com/swagger-api/swagger-ui) is available at `app/docs`. [Redoc](https://github.com/Redocly/redoc) is available at `app/redoc` or root URL with redirection.

# Setup and Installation

Installation assumes that you are in the root directory of Whisper-WebUI

1. Create `.env` in `backend/configs/.env`
```
HF_TOKEN="YOUR_HF_TOKEN FOR DIARIZATION MODEL (READ PERMISSION)"
DB_URL="sqlite:///backend/records.db"
```
`HF_TOKEN` is used to download diarization model, `DB_URL` indicates where your db file is located. It is stored in `backend/` by default.

2. Install dependency
```
pip install -r backend/requirements-backend.txt
```

3. Deploy the server with `uvicorn` or whatever. 
```
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

## Configuration
You can set some server configurations in [config.yaml](https://github.com/jhj0517/Whisper-WebUI/blob/feature/add-api/backend/configs/config.yaml).
<br>For example, initial model size for Whisper or the cleanup frequency and TTL for cached files.
<br>All output files are stored in the `cache` directory, e.g. separated vocal/instrument files for `/bgm-separation` are saved in `cache` directory.

## Docker
The Dockerfile should be built when you're in the root directory of Whisper-WebUI.

1. git clone this repository
```
git clone https://github.com/jhj0517/AdvancedLivePortrait-WebUI.git
```
2. Mount volume paths with your local paths in `docker-compose.yaml`
https://github.com/jhj0517/Whisper-WebUI/blob/d13d773be5e9c1a19f829e31dc10c3c6a6329bc8/backend/docker-compose.yaml#L13-L16
3. Build the image
```
docker compose -f backend/docker-compose.yaml build
```
4. Run the container
```
docker compose -f backend/docker-compose.yaml up
```

5. Then you can read docs at `localhost:8000` (default port is set to `8000` in `docker-compose.yaml`) and run your own tests. 


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
