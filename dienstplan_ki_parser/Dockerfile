ARG BUILD_FROM=ghcr.io/hassio-addons/base:14.2.2
FROM ${BUILD_FROM}

ENV LANG C.UTF-8

# Systempakete installieren
RUN apk add --no-cache \
    python3 \
    py3-pip \
    tesseract-ocr \
    tesseract-ocr-data-deu \
    libjpeg-turbo \
    zlib \
    libpng \
    libstdc++

# Python-Bibliotheken installieren
RUN pip3 install --no-cache-dir \
    ultralytics==8.0.20 \
    opencv-python-headless==4.7.0.72 \
    numpy

# Skript kopieren
COPY run.py /run.py

CMD ["python3", "/run.py"]
