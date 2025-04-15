ARG BUILD_FROM
FROM $BUILD_FROM

ENV LANG C.UTF-8

# Installiere benötigte Pakete
RUN apk add --no-cache \
    python3 \
    py3-pip \
    tesseract-ocr \
    tesseract-ocr-deu \
    libstdc++ \
    libjpeg-turbo \
    libpng \
    && pip3 install --no-cache-dir ultralytics opencv-python-headless
