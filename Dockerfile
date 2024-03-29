FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04 AS runtime

VOLUME [ "/Whisper-WebUI/models" ]
VOLUME [ "/Whisper-WebUI/outputs" ]

RUN apt-get update && \
    apt-get install -y curl ffmpeg git python3 python3-pip python3-venv && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* && \
    mkdir -p /Whisper-WebUI

WORKDIR /Whisper-WebUI

COPY . .

RUN python3 -m venv venv && \
    . venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt

ENV PATH="/Whisper-WebUI/venv/bin:$PATH"
ENV LD_LIBRARY_PATH=/Whisper-WebUI/venv/lib64/python3.10/site-packages/nvidia/cublas/lib:/Whisper-WebUI/venv/lib64/python3.10/site-packages/nvidia/cudnn/lib

ENTRYPOINT [ "python", "app.py" ]
