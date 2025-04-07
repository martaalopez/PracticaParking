import cv2

# Configura la URL RTSP de tu cámara EasyP
#rtsp_url = "rtsp://admin:IESgc14@@192.168.127.72:554"
rtsp_url = "rtsp://admin:IESgc14@@192.168.12.253:554"


# Inicializa la captura de video
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise Exception("Error: No se pudo conectar a la cámara. Verifica la URL RTSP.")

# Carga el modelo preentrenado para detección de coches (Haar Cascade)
car_cascade = cv2.CascadeClassifier(r'C:\Users\usuario\Desktop\IAYBIGDATA\DeteccionMatriculas\cars.xml')
if car_cascade.empty():
    raise Exception("Error: No se pudo cargar el archivo cars.xml")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo capturar el frame. Verifica la URL RTSP.")
            break

        # Convierte el frame a escala de grises para la detección de coches
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detecta coches en el frame
        cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in cars:
            # Dibuja un rectángulo alrededor del coche detectado (en el frame original en color)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Recorta la región de interés (ROI) donde se encuentra el coche
            roi = frame[y:y+h, x:x+w]

            # Muestra la ROI en una ventana separada (opcional)
            cv2.imshow('ROI', roi)

        # Muestra el frame original en color con los coches detectados
        cv2.imshow('Frame', frame)

        # Presiona 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("Ejecución interrumpida por el usuario.")
finally:
    # Libera la captura y cierra las ventanas
    cap.release()
    cv2.destroyAllWindows()