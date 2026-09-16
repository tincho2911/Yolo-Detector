'use strict';

// Elementos de la página
const dropzone = document.getElementById('dropzone');
const dropzoneContent = document.getElementById('dropzoneContent');
const fileInput = document.getElementById('fileInput');
const previewImg = document.getElementById('previewImg');
const detectBtn = document.getElementById('detectBtn');
const errorMsg = document.getElementById('errorMsg');

const resultsPanel = document.getElementById('resultsPanel');
const resultImg = document.getElementById('resultImg');
const loadingOverlay = document.getElementById('loadingOverlay');
const detectionsBody = document.getElementById('detectionsBody');
const metaCount = document.getElementById('metaCount');
const metaTime = document.getElementById('metaTime');
const downloadJson = document.getElementById('downloadJson');

let selectedFile = null;

// Elegir imagen: clic en el recuadro o arrastrar y soltar
dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropzone.classList.remove('dragover');
  if (e.dataTransfer.files?.[0]) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files?.[0]) handleFile(fileInput.files[0]);
});

function handleFile(file) {
  if (!file.type.startsWith('image/')) {
    showError('El archivo elegido no es una imagen.');
    return;
  }
  hideError();
  selectedFile = file;
  previewImg.src = URL.createObjectURL(file);
  previewImg.hidden = false;
  dropzoneContent.hidden = true;
  detectBtn.disabled = false;
}

function showError(msg) { errorMsg.textContent = msg; errorMsg.hidden = false; }
function hideError() { errorMsg.hidden = true; }

// Enviar la imagen al servidor y mostrar el resultado
detectBtn.addEventListener('click', async () => {
  if (!selectedFile) return;
  hideError();
  detectBtn.disabled = true;
  resultsPanel.hidden = false;
  loadingOverlay.hidden = false;
  resultImg.removeAttribute('src');

  const formData = new FormData();
  formData.append('image', selectedFile);

  try {
    const res = await fetch('/api/detect', { method: 'POST', body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Error desconocido');

    resultImg.src = data.annotatedImageUrl;
    metaCount.textContent = `${data.detectionCount} objeto(s) detectado(s)`;
    metaTime.textContent = `· inferencia: ${data.inferenceMs} ms`;
    downloadJson.href = data.jsonUrl;
    renderTable(data.detections);
  } catch (err) {
    showError(`No se pudo procesar la imagen: ${err.message}`);
    resultsPanel.hidden = true;
  } finally {
    loadingOverlay.hidden = true;
    detectBtn.disabled = false;
  }
});

function renderTable(detections) {
  detectionsBody.innerHTML = '';
  if (!detections.length) {
    detectionsBody.innerHTML = '<tr><td colspan="4">No se detectaron objetos.</td></tr>';
    return;
  }
  detections
    .slice()
    .sort((a, b) => b.confidence - a.confidence)
    .forEach((d, i) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${i + 1}</td>
        <td>${d.class}</td>
        <td>${(d.confidence * 100).toFixed(1)}%</td>
        <td>${d.box.x}, ${d.box.y}, ${d.box.width}, ${d.box.height}</td>
      `;
      detectionsBody.appendChild(tr);
    });
}
