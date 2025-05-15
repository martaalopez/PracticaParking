# from ultralytics import YOLO
import cv2
import datetime
import numpy as np
import json
from ultralytics import YOLO  

# Cargar el modelo YOLO
model = YOLO('yolov8m.pt')

class ParkingSystem:
    def __init__(self, linea_conteo, total_plazas=24):
        self.linea_conteo = linea_conteo
        self.total_plazas = total_plazas
        self.entradas = 0
        self.salidas = 0
        self.coches_en_parking = set()
        self.tiempos_cruce = {}
        self.historial = {}
        self.coches_movimiento = 0
        self.eventos = []
        self.tiempos = []

    def actualizar(self, frame, tracker):
        current_time = datetime.datetime.now()

        for caja in tracker[0].boxes:
            if not caja.id:
                continue

            id = int(caja.id)
            x, y = float(caja.xywh[0][0]), float(caja.xywh[0][1])

            if id not in self.historial:
                self.historial[id] = []
            self.historial[id].append((x, y, current_time))

            if len(self.historial[id]) > 5:
                self.historial[id] = self.historial[id][-5:]

            if (self.linea_conteo[0][0] <= x <= self.linea_conteo[1][0] and 
                abs(y - self.linea_conteo[0][1]) < 10):

                if len(self.historial[id]) >= 2:
                    delta_y = y - self.historial[id][-2][1]

                    if delta_y > 0:
                        if id not in self.coches_en_parking:
                            self.entradas += 1
                            self.coches_en_parking.add(id)
                            self.tiempos_cruce[id] = (current_time, None)
                            self.eventos.append({
                                "id": id,
                                "tipo": "entrada",
                                "hora": current_time.isoformat()
                            })
                            cv2.putText(frame, "ENTRADA", (int(x), int(y)-15), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    elif delta_y <= 0:    
                        if id in self.coches_en_parking:  
                            self.salidas += 1
                            self.coches_en_parking.remove(id)
                            self.tiempos_cruce[id] = (current_time)
                            self.eventos.append({
                                "id": id,
                                "tipo": "salida",
                                "hora": current_time.isoformat()
                            })
                            cv2.putText(frame, "SALIDA", (int(x), int(y)-15), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        ids_actuales = {int(box.id) for box in tracker[0].boxes if box.id}
        self.coches_en_parking = {id for id in self.coches_en_parking if id in ids_actuales}
        self.historial = {k: v for k, v in self.historial.items() if k in ids_actuales}

        self._mostrar_info(frame, tracker)

    def _mostrar_info(self, frame, tracker):
        alto, ancho, _ = frame.shape
        cv2.rectangle(frame, (0, alto - 150), (400, alto), (40, 40, 40), -1)

        coches_actuales = len({int(box.id) for box in tracker[0].boxes if box.id})
        plazas_libres = max(self.total_plazas - coches_actuales, 0)

        info_base = [
            f"COCHES ACTUALES EN PARKING: {coches_actuales}",
            f"PLAZAS LIBRES: {plazas_libres}/{self.total_plazas}",
            f"TOTAL ENTRADAS: {self.entradas}",
            f"TOTAL SALIDAS: {self.salidas}",
        ]

        for i, texto in enumerate(info_base):
            cv2.putText(frame, texto, (10, alto - 130 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        cv2.line(frame, self.linea_conteo[0], self.linea_conteo[1], (0, 255, 255), 2)



# --- CONFIGURACIÓN DE FUENTE DE VIDEO ---
# Opción 1: Video en vivo desde la cámara RTSP cenital
#rtsp_url = "rtsp://admin:IESgc14!@192.168.12.253:554"
#cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
#if not cap.isOpened():
 #   raise Exception("Error: No se pudo conectar a la cámara. Verifica la URL RTSP.")

# Opción 2: Video desde archivo local (solo si deseas probar sin cámara)
video_path = './R192_168_12_253_80_CH01_16_36_01.webm'
cap = cv2.VideoCapture(video_path)

parking = ParkingSystem(linea_conteo=((650, 30), (900, 30)))

while True:
    start = datetime.datetime.now()

    ret, frame = cap.read()
    if not ret:
        break

    alto, ancho, _ = frame.shape
    mask = np.zeros((alto, ancho), dtype=np.uint8)
    pts = np.array([[0, 0], [ancho // 2, 0], [0, alto // 2]], np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)

    frame_blur = cv2.GaussianBlur(frame, (51, 51), 0)
    frame[mask == 255] = frame_blur[mask == 255]

    resultado = model.track(frame)
    parking.actualizar(frame, resultado)

    end = datetime.datetime.now()
    fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
    cv2.putText(frame, fps, (10, 25), 1, 1, (0, 0, 255), 2)

    cv2.imshow('Video', resultado[0].plot(line_width=1))
    if cv2.waitKey(1) == 27:
        break

with open("eventos_parking.json", "w") as f:
    json.dump(parking.eventos, f, indent=4)

cap.release()
cv2.destroyAllWindows()
