ARG BUILD_FROM
FROM $BUILD_FROM

ENV LANG C.UTF-8

# Systemabhängigkeiten installieren
RUN apk add --no-cache \
    python3 \
    py3-pip \
    tesseract-ocr \
    libstdc++ \
    jpeg-dev \
    zlib-dev \
    libpng-dev \
    && pip3 install --no-cache-dir \
    ultralytics \
    opencv-python-headless \
    numpy
