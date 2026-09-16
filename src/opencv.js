'use strict';

// @techstark/opencv-js expone el build WASM "oficial" de OpenCV.js
// (el mismo que se usa en el navegador) empaquetado para Node.
// El módulo exporta una Promise que resuelve cuando el runtime WASM
// terminó de inicializarse; a partir de ahí `cv` tiene la API real de
// OpenCV (Mat, resize, cvtColor, rectangle, putText, copyMakeBorder, etc).
//
// Nota: al ser el build pensado para browser, NO incluye los codecs de
// imgcodecs (imread/imencode), porque en el navegador esas tareas las
// resuelve el <canvas>. Por eso la lectura/escritura de archivos de
// imagen (decode/encode JPEG-PNG) se hace con `sharp`, y esos bytes
// crudos (RGB) se cargan en un cv.Mat real para que TODO el
// procesamiento (resize, letterbox, conversión de color, dibujo de
// bounding boxes) se haga con funciones genuinas de OpenCV.

let cvPromise = null;

function getOpenCV() {
  if (!cvPromise) {
    cvPromise = require('@techstark/opencv-js');
  }
  return cvPromise;
}

module.exports = { getOpenCV };
