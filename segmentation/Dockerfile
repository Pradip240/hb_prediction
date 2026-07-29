FROM pytorch/pytorch:2.12.0-cuda13.2-cudnn9-runtime

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12-venv libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --break-system-packages --no-cache-dir -r requirements.txt

# Bake the SegFormer weights INTO the image so nothing is downloaded at run time.
RUN python -c "from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor; \
    SegformerImageProcessor.from_pretrained('jonathandinu/face-parsing'); \
    SegformerForSemanticSegmentation.from_pretrained('jonathandinu/face-parsing')"

COPY segmentation.py .

ENTRYPOINT ["python", "segmentation.py"]

# docker build -t rppg-segmentation .