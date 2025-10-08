FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel
ARG DEBIAN_FRONTEND=noninteractive

ENV TORCH_CUDA_ARCH_LIST="7.5" \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# 시스템 패키지
RUN apt-get update && apt-get install -y --no-install-recommends \
      git wget ffmpeg ca-certificates build-essential \
      ninja-build cmake pkg-config \
      libglib2.0-0 libsm6 libxext6 libxrender1 v4l-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/program

# GroundingDINO 소스/가중치
RUN git clone https://github.com/IDEA-Research/GroundingDINO.git
RUN mkdir -p /opt/program/GroundingDINO/weights && \
    wget -q -O /opt/program/GroundingDINO/weights/groundingdino_swint_ogc.pth \
      https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# 파이썬 의존성
COPY inference/requirements.txt requirements.txt
COPY inference/constraints.txt constraints.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt -c constraints.txt --no-cache-dir
RUN pip install -e GroundingDINO/ --no-build-isolation -c constraints.txt --no-cache-dir
RUN ln -s /opt/program/GroundingDINO/weights /opt/program/weights

# FastAPI 앱 및 프로젝트 코드
COPY inference/app.py app.py
COPY api api
COPY config config
COPY src src
COPY cli cli
COPY data data

EXPOSE 8000
# 워커 1개 권장(GPU 공유/초기화 이슈 방지). 필요 시 --workers 늘리기.
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000"]
