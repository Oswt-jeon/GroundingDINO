FROM python:3.10-slim

ARG DEBIAN_FRONTEND=noninteractive

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      libgl1 \
      libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY front/streamlit_requirements.txt requirements.txt

RUN python -m venv /opt/venv \
 && pip install --upgrade pip \
 && pip install -r requirements.txt

# Copy only the frontend source (exclude local virtualenvs)
COPY front/streamlit_app.py front/streamlit_app.py
COPY front/streamlit_direct.py front/streamlit_direct.py
COPY front/STREAMLIT_README.md front/STREAMLIT_README.md
COPY front/front.py front/front.py

ENV BACKEND_URL=http://inference:8000

EXPOSE 8501

CMD ["streamlit", "run", "front/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

