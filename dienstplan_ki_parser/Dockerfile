ARG BUILD_FROM
FROM $BUILD_FROM

ENV LANG C.UTF-8

# Installiere benötigte Systempakete
RUN apk add --no-cache \
    python3 \
    py3-pip \
    tesseract-ocr \
    tesseract-ocr-data-deu \
    libstdc++ \
    jpeg-dev \
    zlib-dev \
    libpng-dev \
    && pip3 install --no-cache-dir \
    ultralytics \
    opencv-python-headless \
    numpy
