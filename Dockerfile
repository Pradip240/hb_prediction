# Use an official PyTorch image with Python 3.12 and CUDA support
# If you don't have a GPU, you can use: python:3.12-slim
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for OpenCV and Image processing
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt


# Set the default command to run training
# You can override this to run data_prep.py if needed
CMD ["python", "train.py"]


# docker build -t hb-predictor .
# docker run -d --gpus all --name hb_train_v1 --shm-size=8g -v ./:/app hb-predictor

# docker logs -f hb_train_v1                    # Stream live output
# docker exec -it hb_train_v1 bash              # Open a shell inside if needed
# docker exec hb_train_v1 cat /app/output/training_log.csv | tail -20  # Peek at training log

# python plot_training.py
# python diagnose.py