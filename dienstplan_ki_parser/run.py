# === run.py: Hauptskript für das Home Assistant Dienstplan-Add-on mit ONNX-Inferenz ===
import cv2
import numpy as np
import onnxruntime as ort
from datetime import datetime, timedelta
import os
import re

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

# Bild laden
img = cv2.imread(IMAGE_PATH)
h, w = img.shape[:2]

# ONNX-Modell laden
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# Bild vorbereiten
img_resized = cv2.resize(img, (100, 100))
img_input = img_resized.astype(np.float32) / 255.0
img_input = np.transpose(img_input, (2, 0, 1))  # CHW
img_input = np.expand_dims(img_input, axis=0)  # Batch

# Inferenz
outputs = session.run(None, {input_name: img_input})
pred = np.argmax(outputs[0], axis=1)[0]
class_names = list(SCHICHTZEITEN.keys())
label = class_names[pred] if pred < len(class_names) else "Unbekannt"

# Nur ein Feld erkannt → für Beispielzweck
felder = [(0, 0, label, 1.0)]  # Dummy-Koordinaten, echte YOLO-Ausgabe nötig

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
