ARG BUILD_FROM=ghcr.io/hassio-addons/base-debian:12.2.6
FROM ${BUILD_FROM}

ENV LANG C.UTF-8

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    python3 python3-pip \
    libgl1 \
    tesseract-ocr \
    tesseract-ocr-deu \
    && apt-get clean

# Install Python packages
RUN pip3 install --no-cache-dir ultralytics opencv-python-headless numpy

# Copy files
COPY run.py /run.py

CMD ["python3", "/run.py"]
