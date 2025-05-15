import cv2
import pytesseract
import re
import numpy as np
import json
import os
from ultralytics import YOLO

# Cargar el modelo YOLO
model = YOLO("license_plate_detector.pt")

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# --- FUENTE DE VIDEO ---

# Opción 1: RTSP desde cámara cenital (descomenta estas líneas si quieres usarla)
# rtsp_url = "rtsp://admin:IESgc14!@192.168.12.253:554"
# cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

# Opción 2: Archivo de video local (comentado si usas RTSP)
video_path = './R192_168_12_254_80_CH01_17_14_08.webm'
cap = cv2.VideoCapture(video_path)

# Verificación de apertura
#if not cap.isOpened():
#    raise Exception("Error: No se pudo conectar a la cámara RTSP o abrir el video.")

# Conjunto para almacenar matrículas únicas
plates_detected = set()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fin del video o no se pudo leer el frame.")
        break

    results = model(frame)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        roi = frame[y1:y2, x1:x2]

        # Preprocesamiento de la ROI
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        _, roi_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        roi_thresh = cv2.morphologyEx(roi_thresh, cv2.MORPH_CLOSE, kernel)

        # OCR con Tesseract
        text = pytesseract.image_to_string(
            roi_thresh,
            config='--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        ).strip().upper()

        print("Texto OCR:", text)

        match = re.search(r'\b\d{4}[A-Z]{3}\b', text.replace(" ", ""))
        if match:
            plate = match.group()
            print("PLACA DETECTADA:", plate)

            plates_detected.add(plate)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, plate, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow("ROI", roi_thresh)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# --- GUARDAR MATRÍCULAS EN JSON ---
json_file = "registro_matriculas.json"

if os.path.exists(json_file):
    with open(json_file, "r") as f:
        try:
            existing_data = json.load(f)
        except json.JSONDecodeError:
            existing_data = []
else:
    existing_data = []

existing_plates = {entry["matricula"] for entry in existing_data}

for plate in plates_detected:
    if plate not in existing_plates:
        existing_data.append({"matricula": plate})
        existing_plates.add(plate)

if existing_data:
    with open(json_file, "w") as f:
        json.dump(existing_data, f, indent=4)

print("Matrículas actualizadas en el JSON sin perder datos anteriores.")
