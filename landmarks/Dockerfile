FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip \
        libglib2.0-0 libgl1 libegl1 libgles2 libglvnd0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Bake the FaceLandmarker model so nothing downloads at run time
ADD https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task /app/face_landmarker.task

COPY landmarks.py .

ENTRYPOINT ["python3", "landmarks.py"]

# docker build -t rppg-landmarks .