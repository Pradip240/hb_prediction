FROM pytorch/pytorch:2.12.0-cuda13.2-cudnn9-runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/

COPY requirements.txt .

RUN pip install  --break-system-packages --no-cache-dir -r requirements.txt

# docker build -t rppg-base .