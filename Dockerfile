FROM debian:bookworm-slim AS builder

RUN apt-get update && \
  apt-get install -y curl git python3 python3-pip python3-venv && \
  rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* && \
  mkdir -p /Whisper-WebUI

WORKDIR /Whisper-WebUI

COPY requirements.txt .

#RUN python3 -m venv venv && \
#    . venv/bin/activate && \
#    pip install -U -r requirements.txt

RUN python3 -m venv venv && \
  . venv/bin/activate && \
  # 1. 创建一个约束文件，强制限制 setuptools 版本小于 70
  echo "setuptools<70" > pip-constraints.txt && \
  # 2. 将该文件设置为环境变量，这会强制 pip 的构建环境也遵守此限制
  export PIP_CONSTRAINT=$(pwd)/pip-constraints.txt && \
  # 3. 升级 pip 并安装基础包（此时会自动遵守上面的约束）
  pip install -U pip wheel setuptools && \
  # 4. 安装报错的 git 包
  pip install git+https://github.com/jhj0517/jhj0517-whisper.git && \
  # 5. 安装剩余依赖
  pip install -U -r requirements.txt

FROM debian:bookworm-slim AS runtime

RUN apt-get update && \
  apt-get install -y curl ffmpeg python3 && \
  rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

WORKDIR /Whisper-WebUI

COPY . .
COPY --from=builder /Whisper-WebUI/venv /Whisper-WebUI/venv

VOLUME [ "/Whisper-WebUI/models" ]
VOLUME [ "/Whisper-WebUI/outputs" ]

ENV PATH="/Whisper-WebUI/venv/bin:$PATH"
ENV LD_LIBRARY_PATH=/Whisper-WebUI/venv/lib64/python3.11/site-packages/nvidia/cublas/lib:/Whisper-WebUI/venv/lib64/python3.11/site-packages/nvidia/cudnn/lib

ENTRYPOINT [ "python", "app.py" ]
