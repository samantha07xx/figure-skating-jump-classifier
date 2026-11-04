const test = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');

function setup() {
  const elements = new Map();
  const requests = [];
  function element() {
    return { hidden: true, textContent: '', disabled: false, style: {},
      handlers: {}, classList: { add() {}, remove() {} }, append() {}, replaceChildren() {},
      addEventListener(name, callback) { this.handlers[name] = callback; } };
  }
  const document = { getElementById(id) {
    if (!elements.has(id)) elements.set(id, element());
    return elements.get(id);
  }, createElement: element };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../web/app.js'), 'utf8'), {
    document, AbortController,
    URL: { createObjectURL: () => 'blob:test', revokeObjectURL() {} },
    FormData: class { append() {} },
    fetch(url, options) {
      if (url === '/config') return Promise.resolve({ json: async () => ({ max_upload_bytes: 1000 }) });
      return new Promise((resolve, reject) => requests.push({ resolve, reject, options }));
    }
  });
  const get = id => document.getElementById(id);
  return { get, requests, select(name) {
    get('video').files = name ? [{ name, size: 10 }] : [];
    get('video').handlers.change();
  }, predict: () => get('predict').handlers.click() };
}
function response(label) {
  return { ok: true, json: async () => ({ predicted_label: label, confidence: .8,
    class_probabilities: Object.fromEntries(['Axel','Flip','Loop','Lutz','Salchow','Toeloop'].map(x => [x, x === label ? .8 : .04])) }) };
}

test('late result from A cannot overwrite B or enable its pending button', async () => {
  const ui = setup();
  ui.select('A.mp4'); const a = ui.predict();
  ui.select('B.mp4'); const b = ui.predict();
  assert.equal(ui.requests[0].options.signal.aborted, true);
  // Deliberately deliver a response even after cancellation.
  ui.requests[0].resolve(response('Lutz')); await a;
  assert.equal(ui.get('result').hidden, true);
  assert.equal(ui.get('predict').disabled, true);
  ui.requests[1].resolve(response('Axel')); await b;
  assert.equal(ui.get('predicted-label').textContent, 'Axel');
  assert.equal(ui.get('result').hidden, false);
  assert.equal(ui.get('predict').disabled, false);
});

test('late request error cannot replace a new invalid-file message', async () => {
  const ui = setup();
  ui.select('A.mp4'); const a = ui.predict();
  ui.select('B.txt');
  ui.requests[0].reject(new Error('old network error')); await a;
  assert.equal(ui.get('error').textContent, 'Choose an MP4 video.');
  assert.equal(ui.get('predict').disabled, true);
  assert.equal(ui.get('file-name').textContent, 'No video selected');
});
