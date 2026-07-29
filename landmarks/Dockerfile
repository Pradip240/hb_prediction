FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Runtime libraries required by OpenCV and MediaPipe GPU.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libegl1 libgles2 libglvnd0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the MediaPipe FaceLandmarker model into the image.
ADD https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task \
    /app/face_landmarker.task

COPY landmarks.py .

ENTRYPOINT ["python", "landmarks.py"]

# docker build -t rppg-landmarks .