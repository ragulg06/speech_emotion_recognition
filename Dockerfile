# Use a base image with Python
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Install system dependencies (PortAudio + others for audio processing)
RUN apt-get update && apt-get install -y \
    libportaudio2 \
    libsndfile1 \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy all project files into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose the port Flask uses
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
