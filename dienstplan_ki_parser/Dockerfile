
FROM ghcr.io/home-assistant/amd64-base-debian:latest

ENV LANG C.UTF-8

# Systempakete installieren
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgl1 \
    tesseract-ocr \
    && pip3 install ultralytics opencv-python-headless

# Arbeitsverzeichnis
WORKDIR /app
COPY run.py /app/run.py
COPY best.pt /app/best.pt

CMD ["python3", "/app/run.py"]
