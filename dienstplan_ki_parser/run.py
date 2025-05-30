# === run.py: Hauptskript für das Home Assistant Dienstplan-Add-on mit ONNX-Modell ===
import cv2
import numpy as np
from datetime import datetime, timedelta
import os
import re
import onnxruntime as ort

MODEL_PATH = "/config/dienstplan_ki_parser/best.onnx"
INPUT_DIR = "/config/www"

# Neuestes Bild im Format dienstplan_MM.JJJJ.jpg finden
jpg_files = [f for f in os.listdir(INPUT_DIR) if f.startswith("dienstplan_") and f.endswith(".jpg")]
if not jpg_files:
    raise FileNotFoundError("Kein passendes Bild im Ordner /config/www gefunden (Format: dienstplan_MM.JJJJ.jpg)")
IMAGE_PATH = os.path.join(INPUT_DIR, max(jpg_files, key=lambda f: os.path.getctime(os.path.join(INPUT_DIR, f))))

# Monat und Jahr aus Dateinamen extrahieren
date_match = re.search(r'dienstplan_(\d{2})\.(\d{4})', IMAGE_PATH)
if not date_match:
    raise ValueError("Dateiname enthält kein gültiges Datum im Format dienstplan_MM.JJJJ")
monat, jahr = int(date_match.group(1)), int(date_match.group(2))
startdatum = datetime(jahr, monat, 1)
anzahl_tage = (datetime(jahr, monat % 12 + 1, 1) - timedelta(days=1)).day

RASTER_SPALTEN, RASTER_ZEILEN = 7, 6
SCHICHTZEITEN = {
    "F01": ("06:45", "14:00"),
    "F04": ("06:45", "10:30"),
    "F06": ("07:00", "14:00"),
    "F07": ("07:00", "13:30"),
    "F09": ("07:00", "13:00"),
    "F10": ("07:00", "12:30"),
    "F13": ("07:00", "10:30"),
    "F14": ("07:00", "10:00"),
    "S01": ("13:45", "21:00"),
    "S04": ("13:45", "20:30"),
}

# Hilfsfunktion für Raster-Position
def get_raster_position(xc, yc, width, height):
    cell_w, cell_h = width / RASTER_SPALTEN, height / RASTER_ZEILEN
    col, row = int(xc // cell_w), int(yc // cell_h)
    return (row, col) if 0 <= row < RASTER_ZEILEN and 0 <= col < RASTER_SPALTEN else (None, None)

# ONNX-Modell laden und Bild vorbereiten
img = cv2.imread(IMAGE_PATH)
img_resized = cv2.resize(img, (640, 640))
img_input = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
img_input = np.transpose(img_input, (2, 0, 1))[np.newaxis, :].astype(np.float32)

# Inferenz
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
outputs = session.run(None, {session.get_inputs()[0].name: img_input})
detections = outputs[0][0]  # [N, 85] z.B. box + 11 Klassen

# Bilddimensionen zurückholen
h, w = img.shape[:2]
felder = []
for det in detections:
    x, y, bw, bh = det[:4]
    scores = det[4:]
    cls = int(np.argmax(scores))
    conf = float(scores[cls])
    if conf < 0.5:
        continue
    xc, yc = x * w, y * h
    row, col = get_raster_position(xc, yc, w, h)
    if row is not None:
        felder.append((row, col, list(SCHICHTZEITEN.keys())[cls], conf))
felder.sort(key=lambda x: (x[0], x[1]))

# ICS-Datei erzeugen
with open("/config/dienstplan.ics", "w", encoding="utf-8") as f:
    f.write("BEGIN:VCALENDAR\nVERSION:2.0\nCALSCALE:GREGORIAN\n")
    for i, (row, col, label, conf) in enumerate(felder):
        if i >= anzahl_tage:
            continue
        datum = startdatum + timedelta(days=i)
        if label in SCHICHTZEITEN:
            start, end = SCHICHTZEITEN[label]
            f.write(f"BEGIN:VEVENT\nSUMMARY:{label}\nDTSTART;TZID=Europe/Berlin:{datum.strftime('%Y%m%d')}T{start.replace(':','')}00\nDTEND;TZID=Europe/Berlin:{datum.strftime('%Y%m%d')}T{end.replace(':','')}00\nEND:VEVENT\n")
    f.write("END:VCALENDAR\n")

print("✅ Dienstplan verarbeitet und als dienstplan.ics gespeichert.")
