# Detector de objetos con YOLOv8 (Node.js + ONNX + OpenCV)

Servicio web que recibe una imagen, la procesa con un modelo YOLOv8
cargado directamente desde un archivo **ONNX** (sin usar la librería
`ultralytics`), guarda las detecciones en **JSON** y genera una copia
de la imagen con las **bounding boxes dibujadas**, mostrando todo en
el navegador.

## Cumplimiento de la consigna

| Requisito | Implementación |
|---|---|
| Objetivo principal (Node.js + YOLO + JSON + navegador) | `server.js` + `src/detector.js` + `public/` |
| 1. OpenCV para lectura/procesamiento | `@techstark/opencv-js` (OpenCV.js) para letterbox, `resize`, `cvtColor`, `rectangle`, `putText` |
| 2. Modelo desde ONNX sin librería YOLO/Ultralytics | `onnxruntime-node` corre `models/yolov8n.onnx` directamente; el pre y post-procesamiento (letterbox, NMS) está escrito desde cero |
| 3. Copia de la imagen con bounding boxes | Se genera en `outputs/<id>_annotated.jpg` |
| 4. Elegir/cargar imagen desde el navegador | Selector de archivo + drag&drop en `public/index.html` |
| 5. Interfaz mejorada | Preview de la imagen, tabla de detecciones ordenada por confianza, descarga del JSON, manejo de errores |

## Instalación y ejecución

Requiere Node.js 18 o superior.

```bash
npm install
npm start
```

Abrir `http://localhost:3000`, elegir una imagen y tocar "Detectar
objetos".

> El modelo `models/yolov8n.onnx` ya está incluido en el proyecto
> (YOLOv8n pre-entrenado en COCO, 80 clases); no hace falta descargar
> ni entrenar nada por separado.

## Estructura del proyecto

```
yolo-detector/
├── server.js              # Express: rutas HTTP, upload, orquestación
├── src/
│   ├── detector.js        # Pipeline completo: preprocesamiento, inferencia, NMS, dibujo
│   ├── opencv.js           # Carga del runtime de OpenCV.js
│   └── labels.js            # 80 clases de COCO
├── models/
│   └── yolov8n.onnx        # Modelo YOLOv8n pre-entrenado, formato ONNX
├── public/                 # Frontend (HTML/CSS/JS)
├── uploads/                 # Imágenes originales subidas (se genera en tiempo de ejecución)
└── outputs/                  # Imágenes anotadas + JSON de resultados (se genera en tiempo de ejecución)
```

## Cómo funciona el pipeline

**1. Modelo.** Se usa YOLOv8n (variante "nano", la más liviana),
pre-entrenada sobre COCO (80 clases), exportada a formato ONNX. La
entrada del modelo es un tensor `[1, 3, 640, 640]` y la salida es un
tensor `[1, 84, 8400]`: 84 = 4 valores de bounding box (`cx, cy, w, h`)
+ 80 puntajes de clase; 8400 = cantidad de anclas candidatas evaluadas
internamente por el modelo. En este export en particular, las
coordenadas de caja vienen normalizadas en el rango [0,1] respecto al
lienzo de 640×640 (se detecta automáticamente y se reescala).

**2. Lectura y preprocesamiento (OpenCV).** La imagen subida se
decodifica con `sharp` a píxeles crudos (la build WASM de OpenCV.js no
incluye los códecs de imagen, ya que en el navegador esa tarea la
resuelve el `<canvas>`), y esos píxeles se cargan en un `cv.Mat` real.
A partir de ahí, todo el procesamiento usa funciones genuinas de
OpenCV: conversión de color (`cv.cvtColor`), y **letterbox**
(`cv.resize` + `cv.copyMakeBorder`) para llevar la imagen a 640×640
preservando la relación de aspecto original, con relleno gris en los
bordes en vez de deformar la imagen.

**3. Inferencia.** `onnxruntime-node` carga el modelo una sola vez y
ejecuta `session.run()` sobre el tensor preprocesado. No se usa
ninguna clase de alto nivel de Ultralytics: la API de ONNX Runtime
solo corre el grafo del modelo; toda la interpretación de la salida
está implementada en este proyecto.

**4. Post-procesamiento manual.** Se recorren las 8400 anclas, se
descartan las que no superan el umbral de confianza (0.35), y se
aplica **NMS (Non-Maximum Suppression)** implementado desde cero para
eliminar cajas duplicadas sobre un mismo objeto, usando el cálculo de
**IoU** (intersección sobre unión) también escrito a mano. Finalmente
se deshace el letterbox para expresar las cajas en las coordenadas de
la imagen original.

**5. Dibujo de resultados.** Sobre una copia de la imagen original se
dibujan las cajas y etiquetas con `cv.rectangle` y `cv.putText`. Las
etiquetas incluyen un algoritmo simple de resolución de colisiones:
si dos detecciones están muy cerca y sus etiquetas de texto se
superpondrían, la etiqueta se corre verticalmente hasta encontrar un
lugar libre, para que el resultado sea legible incluso con objetos
agrupados. La imagen final se re-codifica a JPEG con `sharp`.

**6. Persistencia y API.** `POST /api/detect` recibe la imagen
(`multer`), guarda el original en `uploads/`, ejecuta el pipeline
completo, guarda la imagen anotada y el JSON de resultados en
`outputs/`, y devuelve todo al navegador (detecciones, tiempo de
inferencia, URLs de los archivos generados).

## Decisiones técnicas

- **OpenCV.js (WASM) en vez de `opencv4nodejs`:** evita tener que
  compilar OpenCV nativo en el sistema (proceso pesado y frágil de
  instalar). A cambio, no incluye codecs de imagen, por lo que la
  decodificación/codificación de archivos (JPEG/PNG) se resuelve con
  `sharp`, mientras que todo el procesamiento de píxeles en sí
  (resize, letterbox, dibujo) se hace con OpenCV real.
- **YOLOv8n:** variante más liviana del modelo, adecuada para correr
  en CPU sin GPU dedicada (inferencia típica de 70-150 ms por imagen
  en una notebook estándar). Cambiar a una variante más grande
  (`yolov8s/m/l/x.onnx`) no requiere tocar código, solo reemplazar el
  archivo en `models/`.
- **Umbrales fijos (confianza 0.35, IoU 0.45):** valores estándar
  para YOLOv8, definidos como constantes en `src/detector.js` para
  mantener la interfaz simple.

## Pruebas realizadas

Se validó el pipeline con imágenes de referencia públicas usadas
habitualmente para evaluar modelos YOLO, con distintos tamaños y
relaciones de aspecto (vertical y horizontal), confirmando que el
letterbox y el reescalado de cajas funcionan correctamente en ambos
casos, incluyendo escenas con múltiples objetos cercanos entre sí.