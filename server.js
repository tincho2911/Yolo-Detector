'use strict';

const path = require('path');
const fs = require('fs');
const crypto = require('crypto');

const express = require('express');
const multer = require('multer');

const { detectObjects } = require('./src/detector');

const app = express();
const PORT = process.env.PORT || 3000;

const UPLOADS_DIR = path.join(__dirname, 'uploads');
const OUTPUTS_DIR = path.join(__dirname, 'outputs');
for (const dir of [UPLOADS_DIR, OUTPUTS_DIR]) {
  fs.mkdirSync(dir, { recursive: true });
}

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 15 * 1024 * 1024 }, // 15 MB
  fileFilter: (req, file, cb) => {
    if (!file.mimetype.startsWith('image/')) {
      return cb(new Error('El archivo debe ser una imagen.'));
    }
    cb(null, true);
  },
});

app.use(express.static(path.join(__dirname, 'public')));
app.use('/outputs', express.static(OUTPUTS_DIR));
app.use('/uploads', express.static(UPLOADS_DIR));

app.get('/api/health', (req, res) => {
  res.json({ status: 'ok' });
});

app.post('/api/detect', upload.single('image'), async (req, res) => {
  console.log('--- Nueva request a /api/detect ---');
  if (!req.file) {
    return res.status(400).json({ error: 'No se recibió ninguna imagen (campo "image").' });
  }
  console.log(`Archivo recibido: ${req.file.originalname} (${req.file.size} bytes)`);

  const id = crypto.randomUUID();
  const ext = (path.extname(req.file.originalname) || '.jpg').toLowerCase();
  const originalName = `${id}_original${ext}`;
  const originalPath = path.join(UPLOADS_DIR, originalName);

  try {
    fs.writeFileSync(originalPath, req.file.buffer);

    // Umbrales fijos: 0.35 de confianza mínima, 0.45 de IoU para el NMS
    // (valores típicos usados en YOLOv8). Están documentados en detector.js.
    const result = await detectObjects(originalPath);

    const annotatedName = `${id}_annotated.jpg`;
    const annotatedPath = path.join(OUTPUTS_DIR, annotatedName);
    fs.writeFileSync(annotatedPath, result.annotatedJpeg);
    console.log('[7/8] Imagen anotada guardada en disco.');

    const jsonName = `${id}_result.json`;
    const jsonPath = path.join(OUTPUTS_DIR, jsonName);
    const payload = {
      id,
      originalFilename: req.file.originalname,
      width: result.width,
      height: result.height,
      model: 'yolov8n.onnx',
      confThreshold: result.confThreshold,
      iouThreshold: result.iouThreshold,
      inferenceMs: result.inferenceMs,
      detectionCount: result.detections.length,
      detections: result.detections,
      generatedAt: new Date().toISOString(),
    };
    fs.writeFileSync(jsonPath, JSON.stringify(payload, null, 2));
    console.log('[8/8] JSON guardado. Mandando respuesta al navegador...');

    res.json({
      ...payload,
      originalImageUrl: `/uploads/${originalName}`,
      annotatedImageUrl: `/outputs/${annotatedName}`,
      jsonUrl: `/outputs/${jsonName}`,
    });
    console.log('Respuesta enviada OK.');
  } catch (err) {
    console.error('Error procesando la imagen:', err);
    res.status(500).json({ error: 'Error procesando la imagen.', details: err.message });
  }
});

app.listen(PORT, () => {
  console.log(`Servidor escuchando en http://localhost:${PORT}`);
});