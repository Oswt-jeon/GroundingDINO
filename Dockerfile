FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel
ARG DEBIAN_FRONTEND=noninteractive

# RTX 2060 (Turing, SM 7.5)
ENV TORCH_CUDA_ARCH_LIST="7.5" \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# 시스템 패키지 (빌드/코덱/필수 라이브러리)
RUN apt-get update && apt-get install -y --no-install-recommends \
      git wget ffmpeg ca-certificates build-essential \
      ninja-build cmake pkg-config \
      libglib2.0-0 libsm6 libxext6 libxrender1 v4l-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/program

# 소스/가중치
RUN git clone https://github.com/IDEA-Research/GroundingDINO.git
RUN mkdir -p weights && cd weights && \
    wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# pip/빌드 도구 업그레이드
RUN pip install --upgrade pip \
 && pip install "setuptools<70" wheel ninja packaging

# (중요) 전역 constraints 고정: numpy<2, transformers==4.38.2
# -> 이후 모든 pip 설치에 -c /tmp/constraints.txt 적용
RUN printf "numpy<2\ntransformers==4.38.2\n" > /tmp/constraints.txt


# 1) TORCH 스택 먼저 (CUDA 12.1)
RUN pip install -c /tmp/constraints.txt --index-url https://download.pytorch.org/whl/cu121 \
      torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --upgrade

# 2) 혹시 끌려온 numpy 제거 후 1.26.x 재설치 (constraints로 강제)
RUN pip uninstall -y numpy || true \
 && pip install -c /tmp/constraints.txt "numpy>=1.26.4,<2.0"

# 3) 나머지 런타임 의존성 (항상 -c 사용)
RUN pip install -c /tmp/constraints.txt "opencv-python-headless<5"

# 4) GroundingDINO 설치 (build isolation 끄고, constraints 유지)
#    필요시 --no-deps를 추가해 종속성 재해결을 완전히 차단할 수 있음
RUN pip install -c /tmp/constraints.txt -e GroundingDINO/ --no-build-isolation
# RUN pip install -c /tmp/constraints.txt -e GroundingDINO/ --no-deps --no-build-isolation

# (옵션) 설치 검증
RUN python -c "import transformers, numpy; print('TRANSFORMERS', transformers.__version__, 'NUMPY', numpy.__version__)"

# 가중치 파일 옮기기
CMD ["mv", "weights", "GroundingDINO/"]

# (옵션) 간단 헬스체크 스크립트
COPY docker_test.py docker_test.py
CMD ["python","docker_test.py"]
