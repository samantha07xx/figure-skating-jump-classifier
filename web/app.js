const input = document.getElementById('video');
const dropzone = document.getElementById('dropzone');
const preview = document.getElementById('preview');
const button = document.getElementById('predict');
const status = document.getElementById('status');
const error = document.getElementById('error');
const result = document.getElementById('result');
const probabilities = document.getElementById('probabilities');
const labels = ['Axel', 'Flip', 'Loop', 'Lutz', 'Salchow', 'Toeloop'];
let selected = null;
let previewUrl = null;
let maxUploadBytes = 100 * 1024 * 1024;
fetch('/config').then(response => response.json()).then(data => { maxUploadBytes = data.max_upload_bytes; }).catch(() => {});
function showError(message) { error.textContent = message; error.hidden = false; status.textContent = ''; }
function selectFile(file) {
  selected = null; result.hidden = true; error.hidden = true;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  preview.hidden = true; button.disabled = true;
  if (!file) { document.getElementById('file-name').textContent = 'No video selected'; return; }
  if (!file.name.toLowerCase().endsWith('.mp4')) { showError('Choose an MP4 video.'); return; }
  if (!file.size) { showError('The selected file is empty.'); return; }
  if (file.size > maxUploadBytes) { showError(`Choose a video under ${Math.round(maxUploadBytes / 1024 / 1024)} MB.`); return; }
  selected = file; document.getElementById('file-name').textContent = file.name;
  previewUrl = URL.createObjectURL(file); preview.src = previewUrl; preview.hidden = false; button.disabled = false;
}
input.addEventListener('change', () => selectFile(input.files[0]));
['dragenter','dragover'].forEach(name => dropzone.addEventListener(name, event => { event.preventDefault(); dropzone.classList.add('drag'); }));
['dragleave','drop'].forEach(name => dropzone.addEventListener(name, event => { event.preventDefault(); dropzone.classList.remove('drag'); }));
dropzone.addEventListener('drop', event => selectFile(event.dataTransfer.files[0]));
button.addEventListener('click', async () => {
  if (!selected) { showError('Choose one MP4 video first.'); return; }
  button.disabled = true; result.hidden = true; error.hidden = true; status.textContent = 'Analyzing video…';
  try {
    const form = new FormData(); form.append('video', selected);
    const response = await fetch('/predict', { method: 'POST', body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Prediction failed. Please try another MP4.');
    if (!labels.includes(data.predicted_label)) throw new Error('The server returned an invalid prediction.');
    document.getElementById('predicted-label').textContent = data.predicted_label === 'Toeloop' ? 'Toe Loop' : data.predicted_label;
    document.getElementById('confidence').textContent = `${(data.confidence * 100).toFixed(1)}%`;
    probabilities.replaceChildren();
    for (const label of labels) {
      const value = data.class_probabilities[label];
      if (typeof value !== 'number' || !Number.isFinite(value)) throw new Error('The server returned invalid probabilities.');
      const row = document.createElement('div'); row.className = `prob-row${label === data.predicted_label ? ' winner' : ''}`;
      const name = document.createElement('span'); name.textContent = label === 'Toeloop' ? 'Toe Loop' : label;
      const track = document.createElement('div'); track.className = 'track';
      const fill = document.createElement('div'); fill.className = 'fill'; fill.style.width = `${Math.max(0, Math.min(100, value * 100))}%`; track.append(fill);
      const percent = document.createElement('span'); percent.className = 'value'; percent.textContent = `${(value * 100).toFixed(1)}%`;
      row.append(name, track, percent); probabilities.append(row);
    }
    result.hidden = false; status.textContent = 'Analysis complete';
  } catch (exc) { showError(exc.message || 'Network error. Check that the app is running.'); }
  finally { button.disabled = false; }
});
