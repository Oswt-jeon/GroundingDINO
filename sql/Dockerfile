FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

WORKDIR /opt/program

COPY sql/requirements.txt /tmp/requirements.txt

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r /tmp/requirements.txt

# 프로젝트 코드는 볼륨으로 주입하므로 여기서는 추가 복사가 필요하지 않습니다.
