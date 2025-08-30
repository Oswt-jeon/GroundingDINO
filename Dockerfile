FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel
ARG DEBIAN_FRONTEND=noninteractive

# Turing(2060, SM 7.5)
ENV TORCH_CUDA_ARCH_LIST="7.5" \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
      git wget build-essential ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/program
RUN git clone https://github.com/IDEA-Research/GroundingDINO.git

# 필요 가중치
RUN mkdir -p weights && cd weights && \
    wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# 파이썬 의존성 & 빌드 도구
RUN pip install --upgrade pip \
 && pip install "setuptools<70" wheel ninja packaging \
 && pip install "opencv-python-headless<5" "numpy<2"

# CUDA 확장 빌드: build isolation 끄고 editable 설치
RUN pip install -e GroundingDINO/ --no-build-isolation

# (옵션) 간단 헬스체크 스크립트
COPY docker_test.py docker_test.py
CMD ["python","docker_test.py"]
