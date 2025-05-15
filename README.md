# PracticaParking


# Sistema de Detección Automática de Coches


## ¿Cómo funciona?

1.Carga del modelo

Se carga el modelo preentrenado YOLOv8 (yolov8m.pt) para detectar vehículos en los frames del vídeo.

2.Clase ParkingSystem

Esta clase gestiona el conteo de coches y controla el estado del parking:

Se define una línea virtual (linea_conteo) en la imagen por donde pasan los coches para contar entradas y salidas.

Lleva el registro de coches dentro del parking, entradas, salidas y eventos.

Usa la posición del coche en frames sucesivos para detectar si está entrando o saliendo según el movimiento respecto a la línea.

3.Procesamiento de vídeo

Se lee un vídeo local o cámara en vivo.

Se aplica un desenfoque (blur) parcial para mejorar la detección en ciertas zonas.

Cada frame se procesa con el modelo YOLO para detectar y trackear coches.

El sistema actualiza el estado del parking, identificando cada coche con un ID y determinando si cruza la línea de conteo.

Se muestra información en pantalla: coches actuales, plazas libres, entradas y salidas totales, y el FPS (fotogramas por segundo).

4.Guardado de eventos

Al finalizar, se guardan en un archivo JSON todos los eventos detectados (entradas y salidas con tiempo e ID).

Resultados

Vemos como ahora mismo no ha entrado ningún vehículo

![image](https://github.com/user-attachments/assets/4068800e-fa54-480b-a58c-474e5938e783)

Cuando entra se suma 1 a la entrada y se actualizan los campos ahora hay 14 coches y 10 plazas libres

![image](https://github.com/user-attachments/assets/96956680-5ebb-4d21-8f5a-a17a9aca40ee)

En nuestro archivo JSON se registra el evento

![image](https://github.com/user-attachments/assets/a84e6f50-12b2-4c2c-8370-c8c276b463ba)


# Sistema de Detección Automática de Matrículas Vehiculares
Tecnologías utilizadas
* YOLOv8 (modelo personalizado) para detección de matrículas.
* Tesseract OCR para reconocimiento del texto de la matrícula.
* OpenCV para procesamiento de imagen y visualización.
* Python como lenguaje principal.
* Expresiones regulares para validar matrículas válidas.

## ¿Cómo funciona?
 
Capturamos el video donde usamos o bien el archivo de vídeo local (.webm) o la cámara en tiempo real (RTSP como: rtsp://admin:IESgc14!@192.168.12.253:554).
Para la detección de matrículas usamos (YOLO) en cada frame del vídeo se pasa al modelo YOLOv8 y el modelo intenta  localizar las regiones donde hay una matrícula.
Devuelve las coordenadas (x1, y1, x2, y2) del rectángulo.
Recorta y preprocesa  la matrícula

Se recorta la región (ROI) y después se convierte a escala de grises.

Aplicamos :

* Redimensionado.
* Filtro bilateral.
* Binarización (umbral Otsu).
* Morfología (cerrado).

Esto seria lo que ha cogido 

![image](https://github.com/user-attachments/assets/d5eebfbb-b5ad-4a53-b62b-5625182e3689)

De este coche

![image](https://github.com/user-attachments/assets/a59bf729-fc4e-4070-8810-bb019270b252)

Más tarde se pasa la ROI procesada a Tesseract OCR,se obtiene una cadena de texto y limpiamos el  texto limpia (strip().upper()).

Se usa una expresión regular para validar que el texto coincida con el formato español,si se cumple, se guarda en un archivo .json.

Resultados

![image](https://github.com/user-attachments/assets/d2d279c2-dcb6-428d-8472-1e9a5f175d46)

Como podemos ver en la terminal nos sale un mensaje con PLACA DETECTADA

En el json se ha escrito la matricula que se ha reconocido 

![image](https://github.com/user-attachments/assets/5f3ea080-8ff0-4a83-bcec-a260c69328de)






