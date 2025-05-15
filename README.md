# PracticaParking

"Sistema de Detección Automática de Matrículas Vehiculares"
Tecnologías utilizadas
* YOLOv8 (modelo personalizado) para detección de matrículas.
* Tesseract OCR para reconocimiento del texto de la matrícula.
* OpenCV para procesamiento de imagen y visualización.
* Python como lenguaje principal.
* Expresiones regulares para validar matrículas válidas.

 ¿Cómo funciona?
Capturamos el video donde usamos o bien el archivo de vídeo local (.webm) o la cámara en tiempo real (RTSP como: rtsp://admin:IESgc14!@192.168.12.253:554).
Para la detección de matrículas usamos (YOLO) en cada frame del vídeo se pasa al modelo YOLOv8 y el modelo intenta  localizar las regiones donde hay una matrícula.
Devuelve las coordenadas (x1, y1, x2, y2) del rectángulo.
Recorta y preprocesa  la matrícula
Se recorta la región (ROI) y después se convierte a escala de grises.
Aplicamos :
Redimensionado.
Filtro bilateral.
Binarización (umbral Otsu).
Morfología (cerrado).

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


