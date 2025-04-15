ARG BUILD_FROM=ghcr.io/hassio-addons/base:14.1.3
FROM ${BUILD_FROM}

ENV LANG C.UTF-8

# Install system packages
RUN apk add --no-cache \
    python3 py3-pip \
    tesseract-ocr tesseract-ocr-data-deu \
    libstdc++ libjpeg-turbo libpng

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    ultralytics \
    opencv-python-headless \
    numpy

# Copy script
COPY run.py /run.py

CMD ["python3", "/run.py"]
