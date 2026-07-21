FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    libegl1 \
    libgles2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    numpy \
    scipy \
    matplotlib \
    opencv-python \
    mediapipe \
    scikit-learn \
    transformers \
    torch \
    torchvision \
    pillow

RUN python -c "from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation; m='jonathandinu/face-parsing'; SegformerImageProcessor.from_pretrained(m); SegformerForSemanticSegmentation.from_pretrained(m)" \
 && python -c "import urllib.request as u; u.urlretrieve('https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task', 'face_landmarker.task')"
