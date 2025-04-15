# === Dockerfile für Home Assistant Add-on mit ONNX-Inferenz ===
ARG BUILD_FROM=ghcr.io/hassio-addons/base-python:3.12.2
FROM ${BUILD_FROM}

# Installiere Systemabhängigkeiten
RUN apk add --no-cache libstdc++ libjpeg-turbo libpng

# Python-Pakete installieren
RUN pip3 install --no-cache-dir onnxruntime opencv-python-headless numpy

# Dateien kopieren
COPY run.py /run.py

CMD [ "python3", "/run.py" ]
