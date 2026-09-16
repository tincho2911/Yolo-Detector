'use strict';

const path = require('path');
const sharp = require('sharp');
const ort = require('onnxruntime-node');

const { getOpenCV } = require('./opencv');
const { COCO_CLASSES } = require('./labels');

const MODEL_PATH = path.join(__dirname, '..', 'models', 'yolov8n.onnx');
const INPUT_SIZE = 640; // yolov8n.onnx espera tensores [1,3,640,640]

let sessionPromise = null;
function getSession() {
  if (!sessionPromise) {
    sessionPromise = ort.InferenceSession.create(MODEL_PATH);
  }
  return sessionPromise;
}

/**
 * Decodifica el archivo de imagen (cualquier formato soportado: jpg, png,
 * webp, etc) a píxeles crudos RGB con sharp, y los vuelca en un cv.Mat real.
 * A partir de acá se trabaja 100% con la API de OpenCV.
 */
async function loadImageAsMat(cv, filePath) {
  const { data, info } = await sharp(filePath)
    .rotate() // respeta la orientación EXIF
    .removeAlpha()
    .toColorspace('srgb')
    .raw()
    .toBuffer({ resolveWithObject: true });

  const rgb = new cv.Mat(info.height, info.width, cv.CV_8UC3);
  rgb.data.set(data);

  // Simulamos el comportamiento estándar de cv2.imread (que lee en BGR),
  // que es el punto de partida habitual de cualquier pipeline de OpenCV.
  const bgr = new cv.Mat();
  cv.cvtColor(rgb, bgr, cv.COLOR_RGB2BGR);
  rgb.delete();
  return bgr; // cv.Mat en formato BGR, tamaño original
}

/**
 * Letterbox: redimensiona preservando el aspect ratio y rellena con
 * padding gris (114,114,114) hasta llegar a un cuadrado size x size.
 * Es el preprocesamiento estándar con el que se entrena/exporta YOLOv8.
 * Devuelve el Mat resultante junto con la info necesaria para des-escalar
 * las cajas detectadas de vuelta a la imagen original.
 */
function letterbox(cv, srcMat, size = INPUT_SIZE) {
  const srcW = srcMat.cols;
  const srcH = srcMat.rows;
  const ratio = Math.min(size / srcW, size / srcH);
  const newW = Math.round(srcW * ratio);
  const newH = Math.round(srcH * ratio);

  const resized = new cv.Mat();
  cv.resize(srcMat, resized, new cv.Size(newW, newH), 0, 0, cv.INTER_LINEAR);

  const padW = size - newW;
  const padH = size - newH;
  const top = Math.floor(padH / 2);
  const bottom = padH - top;
  const left = Math.floor(padW / 2);
  const right = padW - left;

  const padded = new cv.Mat();
  cv.copyMakeBorder(
    resized, padded, top, bottom, left, right,
    cv.BORDER_CONSTANT, new cv.Scalar(114, 114, 114, 0),
  );
  resized.delete();

  return { mat: padded, ratio, padTop: top, padLeft: left };
}

/**
 * Convierte un Mat BGR HWC de 640x640, ya normalizado 0-255, en el
 * Float32Array NCHW [1,3,640,640] normalizado a [0,1] y en orden RGB
 * (formato que espera la entrada "images" del modelo YOLOv8 ONNX).
 */
function matToTensor(cv, paddedBgrMat) {
  const rgb = new cv.Mat();
  cv.cvtColor(paddedBgrMat, rgb, cv.COLOR_BGR2RGB);

  const size = INPUT_SIZE;
  const chw = new Float32Array(3 * size * size);
  const pixels = rgb.data; // Uint8Array HWC, 3 canales

  const plane = size * size;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const pixelIdx = (y * size + x) * 3;
      const outIdx = y * size + x;
      chw[0 * plane + outIdx] = pixels[pixelIdx] / 255; // R
      chw[1 * plane + outIdx] = pixels[pixelIdx + 1] / 255; // G
      chw[2 * plane + outIdx] = pixels[pixelIdx + 2] / 255; // B
    }
  }
  rgb.delete();
  return chw;
}

function iou(a, b) {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  const interW = Math.max(0, x2 - x1);
  const interH = Math.max(0, y2 - y1);
  const inter = interW * interH;
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (areaA + areaB - inter + 1e-6);
}

/** Non-Maximum Suppression clásica, por clase, implementada a mano
 * (no usamos ninguna utilidad de post-proceso de Ultralytics). */
function nms(boxes, iouThreshold) {
  const byClass = new Map();
  for (const b of boxes) {
    if (!byClass.has(b.classId)) byClass.set(b.classId, []);
    byClass.get(b.classId).push(b);
  }

  const kept = [];
  for (const group of byClass.values()) {
    group.sort((a, b) => b.confidence - a.confidence);
    const active = [...group];
    while (active.length) {
      const best = active.shift();
      kept.push(best);
      for (let i = active.length - 1; i >= 0; i--) {
        if (iou(best, active[i]) > iouThreshold) active.splice(i, 1);
      }
    }
  }
  return kept;
}

/**
 * Parsea la salida cruda del modelo [1, 84, 8400] (4 coords de bbox +
 * 80 scores de clase, transpuesto), filtra por confianza y aplica NMS.
 * Devuelve cajas en el espacio de 640x640 (antes de deshacer el letterbox).
 */
function postprocess(outputTensor, confThreshold, iouThreshold) {
  const data = outputTensor.data; // Float32Array
  const numAttrs = outputTensor.dims[1]; // 84
  const numAnchors = outputTensor.dims[2]; // 8400
  const numClasses = numAttrs - 4;

  // Este export en particular devuelve cx,cy,w,h NORMALIZADOS en [0,1]
  // respecto del lienzo de entrada (640x640), en vez de en píxeles
  // absolutos como hacen otros exports de Ultralytics. Lo detectamos
  // una sola vez mirando el primer batch y escalamos a píxeles de INPUT_SIZE.
  let maxCoord = 0;
  for (let i = 0; i < numAnchors; i += 97) { // muestreo rápido, no hace falta recorrer todo
    maxCoord = Math.max(maxCoord, data[0 * numAnchors + i], data[1 * numAnchors + i]);
  }
  const coordScale = maxCoord <= 1.5 ? INPUT_SIZE : 1; // normalizado -> escalar; ya en píxeles -> no tocar

  const candidates = [];
  for (let i = 0; i < numAnchors; i++) {
    let bestScore = 0;
    let bestClass = -1;
    for (let c = 0; c < numClasses; c++) {
      const score = data[(4 + c) * numAnchors + i];
      if (score > bestScore) {
        bestScore = score;
        bestClass = c;
      }
    }
    if (bestScore < confThreshold) continue;

    const cx = data[0 * numAnchors + i] * coordScale;
    const cy = data[1 * numAnchors + i] * coordScale;
    const w = data[2 * numAnchors + i] * coordScale;
    const h = data[3 * numAnchors + i] * coordScale;

    candidates.push({
      x1: cx - w / 2,
      y1: cy - h / 2,
      x2: cx + w / 2,
      y2: cy + h / 2,
      confidence: bestScore,
      classId: bestClass,
    });
  }

  return nms(candidates, iouThreshold);
}

/** Deshace el letterbox y devuelve coordenadas en píxeles de la imagen original. */
function unletterboxBoxes(boxes, { ratio, padLeft, padTop }, origW, origH) {
  return boxes.map((b) => {
    const x1 = Math.max(0, Math.min(origW, (b.x1 - padLeft) / ratio));
    const y1 = Math.max(0, Math.min(origH, (b.y1 - padTop) / ratio));
    const x2 = Math.max(0, Math.min(origW, (b.x2 - padLeft) / ratio));
    const y2 = Math.max(0, Math.min(origH, (b.y2 - padTop) / ratio));
    return {
      class: COCO_CLASSES[b.classId] ?? `class_${b.classId}`,
      classId: b.classId,
      confidence: Number(b.confidence.toFixed(4)),
      box: {
        x: Math.round(x1),
        y: Math.round(y1),
        width: Math.round(x2 - x1),
        height: Math.round(y2 - y1),
      },
    };
  });
}

const BOX_COLORS = [
  [255, 56, 56], [255, 157, 151], [255, 112, 31], [255, 178, 29], [207, 210, 49],
  [72, 249, 10], [146, 204, 23], [61, 219, 134], [26, 147, 52], [0, 212, 187],
  [44, 153, 168], [0, 194, 255], [52, 69, 147], [100, 115, 255], [0, 24, 236],
  [132, 56, 255], [82, 0, 133], [203, 56, 255], [255, 149, 200], [255, 55, 199],
];

/** Dos rectángulos {x1,y1,x2,y2} se superponen. */
function rectsOverlap(a, b) {
  return a.x1 < b.x2 && a.x2 > b.x1 && a.y1 < b.y2 && a.y2 > b.y1;
}

/**
 * Busca una posición vertical libre para el fondo de una etiqueta, evitando
 * que se superponga con etiquetas ya dibujadas. Primero intenta arriba de la
 * caja; si choca con otra etiqueta, va bajando de a un alto de línea por vez
 * (hasta un máximo de intentos) antes de resignarse a superponer.
 */
function findFreeLabelPosition(desiredRect, placedRects, stepDown, maxTries = 15) {
  let rect = desiredRect;
  let tries = 0;
  while (tries < maxTries && placedRects.some((r) => rectsOverlap(rect, r))) {
    rect = { x1: rect.x1, x2: rect.x2, y1: rect.y1 + stepDown, y2: rect.y2 + stepDown };
    tries++;
  }
  return rect;
}

/** Dibuja las bounding boxes + etiquetas sobre una copia de la imagen original,
 * usando exclusivamente primitivas de OpenCV (rectangle, putText). */
function drawDetections(cv, bgrMat, detections) {
  const out = bgrMat.clone();

  // Dibujamos primero todos los rectángulos de caja, y las etiquetas después,
  // en orden de arriba hacia abajo: así, cuando una etiqueta "choca" con otra
  // ya ubicada, sabemos que la de arriba ya quedó fija y solo movemos la nueva.
  const ordered = [...detections].sort((a, b) => a.box.y - b.box.y);
  const placedLabelRects = [];

  for (const det of ordered) {
    const color = BOX_COLORS[det.classId % BOX_COLORS.length];
    const scalar = new cv.Scalar(color[2], color[1], color[0], 255); // BGR
    const p1 = new cv.Point(det.box.x, det.box.y);
    const p2 = new cv.Point(det.box.x + det.box.width, det.box.y + det.box.height);
    cv.rectangle(out, p1, p2, scalar, 2);

    const label = `${det.class} ${(det.confidence * 100).toFixed(1)}%`;
    const fontScale = 0.5;
    const thickness = 1;
    // cv.getTextSize no está expuesto en este build WASM de OpenCV.js;
    // estimamos el ancho del texto con un promedio de ancho de carácter
    // para el font HERSHEY_SIMPLEX (suficiente para dibujar el fondo de la etiqueta).
    const approxCharWidth = 9 * fontScale;
    const textWidth = label.length * approxCharWidth;
    const textHeight = 16 * fontScale;
    const labelHeight = textHeight + 6;

    // Posición "ideal": pegada arriba de la caja.
    const desiredRect = {
      x1: det.box.x,
      x2: det.box.x + textWidth + 4,
      y1: Math.max(0, det.box.y - labelHeight),
      y2: Math.max(0, det.box.y - labelHeight) + labelHeight,
    };

    // Si choca con una etiqueta ya dibujada, la corremos hacia abajo de a
    // un alto de línea, dentro de la propia caja, hasta encontrar hueco libre.
    const finalRect = findFreeLabelPosition(desiredRect, placedLabelRects, labelHeight);
    placedLabelRects.push(finalRect);

    const textBgP1 = new cv.Point(finalRect.x1, finalRect.y1);
    const textBgP2 = new cv.Point(finalRect.x2, finalRect.y2);
    cv.rectangle(out, textBgP1, textBgP2, scalar, -1);
    cv.putText(
      out, label, new cv.Point(finalRect.x1 + 2, finalRect.y2 - 4),
      cv.FONT_HERSHEY_SIMPLEX, fontScale, new cv.Scalar(255, 255, 255, 255), thickness,
    );
  }
  return out;
}

/** Codifica un cv.Mat BGR a un Buffer JPEG usando sharp (OpenCV.js WASM no
 * trae imgcodecs para escritura de archivos). */
async function matToJpegBuffer(cv, bgrMat) {
  const rgb = new cv.Mat();
  cv.cvtColor(bgrMat, rgb, cv.COLOR_BGR2RGB);
  const buf = Buffer.from(rgb.data);
  const jpeg = await sharp(buf, {
    raw: { width: rgb.cols, height: rgb.rows, channels: 3 },
  }).jpeg({ quality: 92 }).toBuffer();
  rgb.delete();
  return jpeg;
}

/**
 * Pipeline completo: imagen en disco -> detecciones + imagen anotada.
 * @param {string} filePath ruta al archivo de imagen subido
 * @param {{confThreshold?: number, iouThreshold?: number}} opts
 */
async function detectObjects(filePath, opts = {}) {
  const confThreshold = opts.confThreshold ?? 0.35;
  const iouThreshold = opts.iouThreshold ?? 0.45;

  console.log('[1/6] Cargando OpenCV...');
  const cv = await getOpenCV();
  console.log('[2/6] OpenCV listo. Cargando modelo y decodificando imagen...');
  const [session, originalMat] = await Promise.all([
    getSession(),
    loadImageAsMat(cv, filePath),
  ]);
  console.log('[3/6] Modelo e imagen listos. Preprocesando (letterbox)...');

  const origW = originalMat.cols;
  const origH = originalMat.rows;

  const lb = letterbox(cv, originalMat, INPUT_SIZE);
  const tensorData = matToTensor(cv, lb.mat);
  lb.mat.delete();

  const inputTensor = new ort.Tensor('float32', tensorData, [1, 3, INPUT_SIZE, INPUT_SIZE]);

  console.log('[4/6] Corriendo inferencia...');
  const t0 = Date.now();
  const results = await session.run({ [session.inputNames[0]]: inputTensor });
  const inferenceMs = Date.now() - t0;
  console.log(`[5/6] Inferencia OK en ${inferenceMs} ms. Postprocesando y dibujando...`);

  const outputTensor = results[session.outputNames[0]];
  const rawBoxes = postprocess(outputTensor, confThreshold, iouThreshold);
  const detections = unletterboxBoxes(rawBoxes, lb, origW, origH);

  const annotatedMat = drawDetections(cv, originalMat, detections);
  const annotatedJpeg = await matToJpegBuffer(cv, annotatedMat);

  originalMat.delete();
  annotatedMat.delete();
  console.log(`[6/6] Listo. ${detections.length} detección(es).`);

  return {
    width: origW,
    height: origH,
    inferenceMs,
    confThreshold,
    iouThreshold,
    detections,
    annotatedJpeg,
  };
}

module.exports = { detectObjects, INPUT_SIZE };