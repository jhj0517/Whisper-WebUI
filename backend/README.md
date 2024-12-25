# Whisper-WebUI REST API
REST API for Whisper-WebUI. Documentation is auto-generated upon deploying the app.
<br>[Swagger UI](https://github.com/swagger-api/swagger-ui) is available at `app/docs` or root URL with redirection. [Redoc](https://github.com/Redocly/redoc) is available at `app/redoc`.

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

### Deploy with your domain name
You can deploy the server with your domain name by setting up a reverse proxy with Nginx.

1. Install Nginx if you don't already have it.
- Linux : https://nginx.org/en/docs/install.html
- Windows : https://nginx.org/en/docs/windows.html

2. Edit [`nginx.conf`](https://github.com/jhj0517/Whisper-WebUI/blob/master/backend/nginx/nginx.conf) for your domain name.
https://github.com/jhj0517/Whisper-WebUI/blob/895cafe400944396ad8be5b1cc793b54fecc8bbe/backend/nginx/nginx.conf#L12

3. Add an A type record of your public IPv4 address in your domain provider. (you can get it by searching "What is my IP" in Google)

4. Open a terminal and go to the location of [`nginx.conf`](https://github.com/jhj0517/Whisper-WebUI/blob/master/backend/nginx/nginx.conf), then start the nginx server, so that you can manage nginx-related logs there.
```shell
cd backend/nginx
nginx -c "/path/to/Whisper-WebUI/backend/nginx/nginx.conf"
```

5. Open another terminal in the root project location `/Whisper-WebUI`, and deploy the app with `uvicorn` or whatever. Now the app will be available at your domain.
```shell
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

6. When you turn off nginx, you can use `nginx -s stop`.
```shell
cd backend/nginx
nginx -s stop -c "/path/to/Whisper-WebUI/backend/nginx/nginx.conf"
```


## Configuration
You can set some server configurations in [config.yaml](https://github.com/jhj0517/Whisper-WebUI/blob/master/backend/configs/config.yaml). 
<br>For example, initial model size for Whisper or the cleanup frequency and TTL for cached files.
<br>If the endpoint generates and saves the file, all output files are stored in the `cache` directory, e.g. separated vocal/instrument files for `/bgm-separation` are saved in `cache` directory.

## Docker
The Dockerfile should be built when you're in the root directory of Whisper-WebUI.

1. git clone this repository
```
git clone https://github.com/jhj0517/Whisper-WebUI.git
```
2. Mount volume paths with your local paths in `docker-compose.yaml`
https://github.com/jhj0517/Whisper-WebUI/blob/1dd708ec3844dbf0c1f77de9ef5764e883dd4c78/backend/docker-compose.yaml#L12-L15
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

The client needs to implement manual API polling to do this, this is the example for the python client:
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
