'use strict';
/* もじ起こしライブ
 * オフラインエンジン: sherpa-onnx WebAssembly (VAD + Zipformer ReazonSpeech)
 * モデルは初回にダウンロードして Cache Storage に保存し、実行時に
 * 公式 .data と同一バイト列を再構成して Emscripten に渡す。
 * 音声はメモリ内でのみ処理し、保存しない。残るのはテキストだけ。
 */

const $ = (s) => document.querySelector(s);

/* ---------- モデル定義（公式WASMパッケージ .data の内訳と同一） ---------- */
const HF_ZIP = 'https://huggingface.co/csukuangfj/reazonspeech-k2-v2/resolve/main';
const MODEL_PARTS = [
  // silero_vad.onnx は GitHub Releases が CORS 非対応のため、同一バイト列を含む
  // 公式HF Space の .data（先頭1076バイトはREADME）から切り出して使う。ハッシュで同一性を保証。
  { fsName: 'silero_vad.onnx', url: 'https://huggingface.co/spaces/k2-fsa/web-assembly-vad-sherpa-onnx/resolve/main/sherpa-onnx-wasm-main-vad.data',
    size: 643854, sha256: '9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6', sliceFrom: 1076 },
  { fsName: 'tokens.txt', url: HF_ZIP + '/tokens.txt',
    size: 45754, sha256: '2c3ac659818a48a0c04010e0593bbc4d7c8a24a054340b01131499c05fd52def' },
  { fsName: 'transducer-decoder.onnx', url: HF_ZIP + '/decoder-epoch-99-avg-1.onnx',
    size: 11767836, sha256: '58b18211ae06265466bfa17172dab574df94f76c8bcb61a3640c28ba860e4124' },
  { fsName: 'transducer-encoder.onnx', url: HF_ZIP + '/encoder-epoch-99-avg-1.int8.onnx',
    size: 154670139, sha256: '2c7bd08a8a99f9ddd0d9e458456577b1f6279214e51426f114f9eced44c54e1d' },
  { fsName: 'transducer-joiner.onnx', url: HF_ZIP + '/joiner-epoch-99-avg-1.int8.onnx',
    size: 2696970, sha256: '49cc7ea1d3d35a40a27442db5e89996da64bf0e683a903dce76e99e57a12e4de' },
];
const MODEL_TOTAL = MODEL_PARTS.reduce((a, p) => a + p.size, 0); // 169824553
/* 高精度エンジン: Parakeet 日本語専用（NeMo TDT-CTC 0.6B, int8）。ベンチマークで最も高精度だった。
 * 公式WASMパッケージは無いが、同梱ランタイムの OfflineRecognizer が nemoCtc 設定を受け付けることを
 * ヘッドレスブラウザで実測済み（2026-11-13、サンプルAを期待文と完全一致でデコード）。 */
const HF_PK = 'https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8/resolve/main';
const PK_PARTS = [
  MODEL_PARTS[0], // silero_vad.onnx（VADは共通）
  { fsName: 'parakeet-ja.onnx', url: HF_PK + '/model.int8.onnx',
    size: 655542604, sha256: '3addd00ef5bd1742078389e540b77394e4a508bdf2f4c9ad1b4a76d93e76598e' },
  { fsName: 'pk-tokens.txt', url: HF_PK + '/tokens.txt',
    size: 28557, sha256: '732f64c53909f2620c713f4106b487d92e6f54a6915b3cd3d1dbd32f9f4f392a' },
];
const PK_TOTAL = PK_PARTS.reduce((a, p) => a + p.size, 0); // 656215015
const LS_ENGINE = 'mojiokoshi.engine.v1';
let engineKind = 'parakeet'; // 'parakeet' | 'zipformer'
try { if (localStorage.getItem(LS_ENGINE) === 'zipformer') engineKind = 'zipformer'; } catch (_) {}
const MODEL_CACHE = 'mojiokoshi-models-v1';
const LS_TEXT = 'mojiokoshi.finals.v1';
const LS_LLM = 'mojiokoshi.llm.v1';
const SR = 16000;

/* ---------- 状態 ---------- */
let engineState = 'idle'; // idle | busy | ready | error
let sherpaVad = null, recognizer = null, circBuffer = null;
let mode = 'offline'; // offline | cloud
let finals = [];
try { finals = JSON.parse(localStorage.getItem(LS_TEXT) || '[]'); if (!Array.isArray(finals)) finals = []; } catch (_) { finals = []; }
let interimText = '';
// セッション音声（メモリのみ。再認識用）
let sessionChunks = [], sessionLength = 0;
// マイク
let audioCtx = null, micStream = null, micSource = null, procNode = null, recordSampleRate = SR;
let recording = false;
let pendingChunks = [], pendingLength = 0; // 現在発話分（途中結果プレビュー用）
let interimTimer = null, interimBusy = false;
// クラウド
let cloudRec = null, cloudWantOn = false;
// ファイル処理中
let fileBusy = false;

/* ---------- UI参照 ---------- */
const els = {
  badge: $('#engineBadge'), prepareBtn: $('#prepareBtn'), engineStatus: $('#engineStatus'),
  dlProgress: $('#dlProgress'), dlBar: $('#dlBar'), dlText: $('#dlText'),
  engineError: $('#engineError'), offlinePanel: $('#offlinePanel'), cloudPanel: $('#cloudPanel'),
  engineKind: $('#engineKind'), engineSizeNote: $('#engineSizeNote'),
  modeOfflineBtn: $('#modeOfflineBtn'), modeCloudBtn: $('#modeCloudBtn'), cloudSupport: $('#cloudSupport'),
  recordBtn: $('#recordBtn'), reDecodeBtn: $('#reDecodeBtn'), fileInput: $('#fileInput'),
  fileProgress: $('#fileProgress'), transcript: $('#transcript'), interim: $('#interimText'),
  copyBtn: $('#copyBtn'), saveBtn: $('#saveBtn'), formatBtn: $('#formatBtn'), aiBtn: $('#aiBtn'),
  clearBtn: $('#clearBtn'), actionMsg: $('#actionMsg'),
  polishCard: $('#polishCard'), polishTitle: $('#polishTitle'), polishText: $('#polishText'),
  polishApplyBtn: $('#polishApplyBtn'), polishCloseBtn: $('#polishCloseBtn'),
  settingsDetails: $('#settingsDetails'), llmEndpoint: $('#llmEndpoint'), llmModel: $('#llmModel'),
  llmKey: $('#llmKey'), llmSaveBtn: $('#llmSaveBtn'), llmClearBtn: $('#llmClearBtn'), llmMsg: $('#llmMsg'),
};

function showMsg(msg, isErr) {
  els.actionMsg.textContent = msg;
  els.actionMsg.classList.remove('hidden');
  els.actionMsg.style.color = isErr ? 'var(--danger)' : '';
  clearTimeout(showMsg._t);
  showMsg._t = setTimeout(() => els.actionMsg.classList.add('hidden'), 5000);
}

/* ---------- テキスト管理 ---------- */
function currentText() { return finals.join('\n'); }
function persistText() { try { localStorage.setItem(LS_TEXT, JSON.stringify(finals)); } catch (_) {} }
function renderText() {
  if (finals.length === 0) {
    els.transcript.innerHTML = '<span class="placeholder">ここに文字起こし結果が表示されます。確定した文は黒、認識中の文はグレーです。</span>';
  } else {
    els.transcript.textContent = currentText();
  }
  els.transcript.scrollTop = els.transcript.scrollHeight;
  els.interim.textContent = interimText;
  els.reDecodeBtn.disabled = !(engineState === 'ready' && sessionLength > 0 && !recording && !fileBusy);
}
function addFinal(text) {
  const t = (text || '').trim();
  if (!t) return;
  finals.push(t);
  persistText();
  interimText = '';
  renderText();
}

/* ---------- モデル取得（進捗つき）＋ ハッシュ検証 ---------- */
async function sha256Hex(buf) {
  const h = await crypto.subtle.digest('SHA-256', buf);
  return [...new Uint8Array(h)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

async function fetchPart(part, onBytes) {
  const cache = await caches.open(MODEL_CACHE);
  const key = new Request(new URL('__model__/' + part.fsName, location.href).toString());
  const hit = await cache.match(key);
  if (hit) {
    const buf = await hit.arrayBuffer();
    if (buf.byteLength === part.size) { onBytes(part.size); return buf; }
    // サイズ不一致の壊れたキャッシュは捨てて取り直す
    await cache.delete(key);
  }
  // 通信が不安定な回線でも通るよう、パート単位で最大3回まで再試行する
  let counted = 0; // このパートで進捗に加算済みのバイト数（再試行時に巻き戻す）
  let lastErr = null;
  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      const resp = await fetch(part.url);
      if (!resp.ok || !resp.body) throw new Error(`モデル取得に失敗しました (${part.fsName}: HTTP ${resp.status})`);
      const reader = resp.body.getReader();
      const chunks = [];
      let got = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value); got += value.byteLength; counted += value.byteLength; onBytes(value.byteLength);
      }
      let buf = new Uint8Array(got);
      let off = 0;
      for (const c of chunks) { buf.set(c, off); off += c.byteLength; }
      if (part.sliceFrom) buf = buf.subarray(part.sliceFrom);
      if (buf.byteLength !== part.size) throw new Error(`${part.fsName} のサイズが想定と違います (${buf.byteLength} != ${part.size})`);
      const hex = await sha256Hex(buf);
      if (hex !== part.sha256) throw new Error(`${part.fsName} のハッシュ検証に失敗しました`);
      try {
        await cache.put(key, new Response(buf, { headers: { 'Content-Type': 'application/octet-stream' } }));
      } catch (cacheErr) {
        // 保存容量不足（QuotaExceeded）でも、メモリ上のバイト列でこのセッションは動作できる。
        // キャッシュできない場合は次回も再ダウンロードになるだけで、致命的ではない。
        console.warn('モデルをキャッシュできませんでした（容量不足の可能性）。このセッションのみ有効です:', part.fsName, cacheErr);
      }
      return buf.slice().buffer; // ビューではなく実バイト列のコピーを返す
    } catch (e) {
      lastErr = e;
      if (counted > 0) { onBytes(-counted); counted = 0; }
      if (attempt < 3) await new Promise((r) => setTimeout(r, 1500 * attempt));
    }
  }
  throw new Error(`${part.fsName} の取得に3回失敗しました。通信環境を確認して再試行してください。（${lastErr && lastErr.message ? lastErr.message : lastErr}）`);
}

/* ---------- sherpa-onnx 読み込み ---------- */
function loadSherpaRuntime(dataUrl, createRecognizer) {
  return new Promise((resolve, reject) => {
    window.Module = {
      locateFile: (path) => (path.endsWith('.data') ? dataUrl : 'vendor/sherpa/' + path),
      setStatus: (s) => { if (s) console.log('[sherpa]', s); },
      print: (...a) => console.log(...a),
      printErr: (...a) => console.warn(...a),
      onRuntimeInitialized: () => {
        try {
          sherpaVad = createVad(window.Module);
          circBuffer = new CircularBuffer(60 * SR, window.Module);
          recognizer = createRecognizer();
          resolve();
        } catch (e) { reject(e); }
      },
      onAbort: (what) => reject(new Error('WASMが異常終了しました: ' + what)),
    };
    const s = document.createElement('script');
    s.src = 'vendor/sherpa/sherpa-onnx-wasm-main-vad-asr.js';
    s.onerror = () => reject(new Error('認識エンジン本体の読み込みに失敗しました'));
    document.body.appendChild(s);
  });
}

async function prepareEngine() {
  if (engineState === 'busy' || engineState === 'ready') return;
  if (!window.crossOriginIsolated) {
    // 初回アクセス直後などで分離モードがまだ有効でない場合は、SW 経由で再読み込みしてから始める
    if (navigator.serviceWorker && navigator.serviceWorker.controller && !sessionStorage.getItem('mojiokoshi.coiReload2')) {
      sessionStorage.setItem('mojiokoshi.coiReload2', '1');
      location.reload();
      return;
    }
    els.engineError.textContent = 'このブラウザではオフライン認識に必要な分離モード（crossOriginIsolated）が有効になっていません。ページを再読み込みするか、最新の Safari / Chrome で開いてください。';
    els.engineError.classList.remove('hidden');
    return;
  }
  engineState = 'busy';
  els.engineError.classList.add('hidden');
  els.prepareBtn.disabled = true;
  els.badge.textContent = '準備中';
  els.dlProgress.classList.remove('hidden');
  const isPk = engineKind === 'parakeet';
  const parts = isPk ? PK_PARTS : MODEL_PARTS;
  const grandTotal = isPk ? PK_TOTAL : MODEL_TOTAL;
  let done = 0;
  const onBytes = (n) => {
    done += n;
    const pct = Math.min(100, (done / grandTotal) * 100);
    els.dlBar.value = pct;
    els.dlText.textContent = `モデルをダウンロード中… ${pct.toFixed(1)}% (${(done / 1048576).toFixed(0)} / ${(grandTotal / 1048576).toFixed(0)} MB)`;
  };
  try {
    els.engineStatus.textContent = 'モデルを確認しています…';
    const buffers = [];
    for (const part of parts) buffers.push(await fetchPart(part, onBytes));
    els.engineStatus.textContent = 'エンジンを初期化しています…';
    els.dlText.textContent = 'ダウンロード完了。エンジン初期化中…';
    let blob;
    if (isPk) {
      // ランタイムの .data は「silero の実データ＋残りはゼロ埋め」で構成する。
      // Parakeet モードでは transducer 用ファイルは一切読まれない（実測で動作確認済み）。
      blob = new Blob([buffers[0], new Uint8Array(MODEL_TOTAL - buffers[0].byteLength)], { type: 'application/octet-stream' });
    } else {
      let total = 0; for (const b of buffers) total += b.byteLength;
      if (total !== MODEL_TOTAL) throw new Error('モデル全体のサイズが想定と一致しません');
      blob = new Blob(buffers, { type: 'application/octet-stream' });
    }
    const dataUrl = URL.createObjectURL(blob);
    await loadSherpaRuntime(dataUrl, () => {
      if (isPk) {
        window.Module.FS_createDataFile('/', 'parakeet-ja.onnx', new Uint8Array(buffers[1]), true, true, false);
        window.Module.FS_createDataFile('/', 'pk-tokens.txt', new Uint8Array(buffers[2]), true, true, false);
        return new OfflineRecognizer({
          modelConfig: { debug: 0, tokens: './pk-tokens.txt', nemoCtc: { model: './parakeet-ja.onnx' } },
        }, window.Module);
      }
      return new OfflineRecognizer({
        modelConfig: {
          debug: 0,
          tokens: './tokens.txt',
          modelType: 'transducer',
          transducer: {
            encoder: './transducer-encoder.onnx',
            decoder: './transducer-decoder.onnx',
            joiner: './transducer-joiner.onnx',
          },
        },
      }, window.Module);
    });
    engineState = 'ready';
    els.badge.textContent = '準備完了（オフライン可）';
    els.engineStatus.textContent = '準備完了。この端末の中だけで処理します。';
    els.dlProgress.classList.add('hidden');
    els.prepareBtn.textContent = '準備完了';
    updateRecordBtn();
    renderText();
  } catch (e) {
    engineState = 'error';
    els.badge.textContent = 'エラー';
    els.badge.classList.add('warn');
    els.engineStatus.textContent = '準備に失敗しました。';
    els.engineError.textContent = String(e && e.message ? e.message : e);
    els.engineError.classList.remove('hidden');
    els.prepareBtn.disabled = false;
    els.prepareBtn.textContent = '再試行する';
  }
}

/* ---------- 認識パイプライン（VAD + オフライン認識） ---------- */
function decodeSamples(samples) {
  const stream = recognizer.createStream();
  stream.acceptWaveform(SR, samples);
  recognizer.decode(stream);
  const text = recognizer.getResult(stream).text || '';
  stream.free();
  return text.trim();
}

function drainVadSegments() {
  let produced = false;
  while (!sherpaVad.isEmpty()) {
    const seg = sherpaVad.front();
    sherpaVad.pop();
    const text = decodeSamples(seg.samples);
    if (text) { addFinal(text); produced = true; }
  }
  if (produced) { pendingChunks = []; pendingLength = 0; }
  return produced;
}

const VAD_WINDOW = () => sherpaVad.config.sileroVad.windowSize;



/* マイク入力は CircularBuffer 経由で窓を切り出す（公式デモ方式） */
function processCircBuffer(isTail) {
  const ws = VAD_WINDOW();
  while (circBuffer.size() > ws) {
    const s = circBuffer.get(circBuffer.head(), ws);
    sherpaVad.acceptWaveform(s);
    circBuffer.pop(ws);
    drainVadSegments();
  }
  if (isTail) {
    const rest = circBuffer.size();
    if (rest > 0) {
      const s = circBuffer.get(circBuffer.head(), rest);
      const padded = new Float32Array(ws);
      padded.set(s.subarray(0, Math.min(rest, ws)));
      sherpaVad.acceptWaveform(padded);
      circBuffer.pop(rest);
    }
    const silence = new Float32Array(ws * 2);
    sherpaVad.acceptWaveform(silence.subarray(0, ws));
    sherpaVad.acceptWaveform(silence.subarray(ws));
    sherpaVad.flush();
    drainVadSegments();
  }
}

function maybeInterimDecode() {
  if (!recording || interimBusy || pendingLength < SR * 0.5) return;
  const now = Date.now();
  if (maybeInterimDecode._last && now - maybeInterimDecode._last < 900) return;
  maybeInterimDecode._last = now;
  interimBusy = true;
  // UIブロックを避けるため次フレームで実行
  setTimeout(() => {
    try {
      let all = concatFloat(pendingChunks, pendingLength);
      if (all.length > SR * 12) all = all.subarray(all.length - SR * 12);
      if (all.length >= SR * 0.4) {
        const text = decodeSamples(all);
        if (recording) { interimText = text; els.interim.textContent = text; }
      }
    } catch (e) { console.warn('interim decode failed', e); }
    interimBusy = false;
  }, 0);
}

function concatFloat(chunks, total) {
  const out = new Float32Array(total);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.length; }
  return out;
}

/* 公式デモ由来のダウンサンプル（awslabs方式） */
function downsampleBuffer(buffer, exportSampleRate) {
  if (exportSampleRate === recordSampleRate) return buffer;
  const ratio = recordSampleRate / exportSampleRate;
  const newLength = Math.round(buffer.length / ratio);
  const result = new Float32Array(newLength);
  let offsetResult = 0, offsetBuffer = 0;
  while (offsetResult < result.length) {
    const nextOffsetBuffer = Math.round((offsetResult + 1) * ratio);
    let accum = 0, count = 0;
    for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) { accum += buffer[i]; count++; }
    result[offsetResult] = accum / count;
    offsetResult++; offsetBuffer = nextOffsetBuffer;
  }
  return result;
}

/* ---------- 録音（オフライン） ---------- */
function onAudioProcess(e) {
  const input = new Float32Array(e.inputBuffer.getChannelData(0));
  const samples = downsampleBuffer(input, SR);
  sessionChunks.push(samples); sessionLength += samples.length;
  pendingChunks.push(samples); pendingLength += samples.length;
  circBuffer.push(samples);
  processCircBuffer(false);
  maybeInterimDecode();
}

async function startMic() {
  if (engineState !== 'ready') { await prepareEngine(); if (engineState !== 'ready') return; }
  try {
    if (!audioCtx) {
      try { audioCtx = new AudioContext({ sampleRate: SR }); }
      catch (_) { audioCtx = new (window.AudioContext || window.webkitAudioContext)(); }
    }
    await audioCtx.resume();
    recordSampleRate = audioCtx.sampleRate;
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    micSource = audioCtx.createMediaStreamSource(micStream);
    procNode = audioCtx.createScriptProcessor(4096, 1, 1);
    procNode.onaudioprocess = onAudioProcess;
    micSource.connect(procNode);
    procNode.connect(audioCtx.destination);
    sherpaVad.reset(); circBuffer.reset();
    pendingChunks = []; pendingLength = 0;
    recording = true;
    updateRecordBtn();
  } catch (e) {
    showMsg('マイクを開始できませんでした: ' + (e && e.message ? e.message : e), true);
  }
}

function stopMic() {
  recording = false;
  if (procNode) { procNode.onaudioprocess = null; try { procNode.disconnect(); } catch (_) {} procNode = null; }
  if (micSource) { try { micSource.disconnect(); } catch (_) {} micSource = null; }
  if (micStream) { micStream.getTracks().forEach((t) => t.stop()); micStream = null; }
  try { processCircBuffer(true); } catch (e) { console.warn(e); }
  interimText = ''; els.interim.textContent = '';
  updateRecordBtn(); renderText();
}

/* ---------- クラウド（Web Speech API） ---------- */
function cloudSupported() { return 'webkitSpeechRecognition' in window || 'SpeechRecognition' in window; }
function startCloud() {
  if (!cloudSupported()) { showMsg('このブラウザは音声認識（Web Speech API）に対応していません。', true); return; }
  const Ctor = window.SpeechRecognition || window.webkitSpeechRecognition;
  cloudRec = new Ctor();
  cloudRec.lang = 'ja-JP';
  cloudRec.continuous = true;
  cloudRec.interimResults = true;
  cloudRec.onresult = (ev) => {
    let interim = '';
    for (let i = ev.resultIndex; i < ev.results.length; i++) {
      const r = ev.results[i];
      if (r.isFinal) addFinal(r[0].transcript);
      else interim += r[0].transcript;
    }
    interimText = interim; els.interim.textContent = interim;
  };
  cloudRec.onerror = (ev) => {
    if (ev.error === 'not-allowed' || ev.error === 'service-not-allowed') {
      cloudWantOn = false; recording = false; updateRecordBtn();
      showMsg('マイクの使用が許可されませんでした。ブラウザの設定を確認してください。', true);
    }
  };
  cloudRec.onend = () => { if (cloudWantOn) { try { cloudRec.start(); } catch (_) {} } };
  cloudWantOn = true;
  recording = true;
  try { cloudRec.start(); } catch (e) { cloudWantOn = false; recording = false; showMsg('クラウド認識を開始できませんでした: ' + e.message, true); }
  updateRecordBtn();
}
function stopCloud() {
  cloudWantOn = false;
  recording = false;
  if (cloudRec) { try { cloudRec.stop(); } catch (_) {} cloudRec = null; }
  interimText = ''; els.interim.textContent = '';
  updateRecordBtn();
}

/* ---------- 録音ボタン ---------- */
function updateRecordBtn() {
  const btn = els.recordBtn;
  if (recording) btn.disabled = false;
  else if (fileBusy) btn.disabled = true;
  else if (mode === 'offline') btn.disabled = engineState !== 'ready';
  else btn.disabled = !cloudSupported();
  btn.textContent = recording ? '■ 録音をとめる' : '● 録音をはじめる';
  btn.classList.toggle('recording', recording);
  renderText();
}
els.recordBtn.addEventListener('click', async () => {
  if (recording) { if (mode === 'cloud') stopCloud(); else stopMic(); return; }
  if (mode === 'cloud') startCloud();
  else await startMic();
});

/* ---------- セッション再認識 ---------- */
els.reDecodeBtn.addEventListener('click', () => {
  if (engineState !== 'ready' || sessionLength === 0 || recording || fileBusy) return;
  const all = concatFloat(sessionChunks, sessionLength);
  if (!confirm('セッション中の音声をもう一度認識し直します（現在のテキストは置き換わります）。よろしいですか？')) return;
  finals = []; persistText(); renderText();
  fileBusy = true;
  els.fileProgress.classList.remove('hidden');
  sherpaVad.reset();
  feedVadProgress(all);
});

/* ---------- ファイル取り込み ---------- */
function feedVadProgress(samples) {
  const ws = VAD_WINDOW();
  const total = samples.length;
  let off = 0;
  // Silero VAD は発話の立ち上がりを区間に含めないことがある（実測: 先頭0.9秒が欠けた）ため、
  // ファイル取り込みでは VAD の start 位置から前0.4秒/後0.2秒の余白を付けて元波形から切り直す。
  const drainFromSource = () => {
    let produced = false;
    while (!sherpaVad.isEmpty()) {
      const seg = sherpaVad.front();
      sherpaVad.pop();
      let segSamples = seg.samples;
      if (typeof seg.start === 'number' && seg.start >= 0) {
        const s0 = Math.max(0, seg.start - Math.floor(SR * 0.4));
        const s1 = Math.min(samples.length, seg.start + seg.samples.length + Math.floor(SR * 0.2));
        if (s1 > s0) segSamples = samples.subarray(s0, s1);
      }
      const text = decodeSamples(segSamples);
      if (text) { addFinal(text); produced = true; }
    }
    if (produced) { pendingChunks = []; pendingLength = 0; }
    return produced;
  };
  const step = () => {
    const t0 = performance.now();
    while (off + ws <= total && performance.now() - t0 < 40) {
      sherpaVad.acceptWaveform(samples.subarray(off, off + ws));
      off += ws;
      drainFromSource();
    }
    els.fileProgress.textContent = `文字起こし中… ${Math.round((off / total) * 100)}%`;
    if (off + ws <= total) { setTimeout(step, 0); return; }
    // 末尾処理
    const rest = total - off;
    const pad = new Float32Array(ws * 2);
    if (rest > 0) pad.set(samples.subarray(off));
    sherpaVad.acceptWaveform(pad.subarray(0, ws));
    sherpaVad.acceptWaveform(pad.subarray(ws));
    sherpaVad.flush();
    drainFromSource();
    fileBusy = false;
    els.fileProgress.classList.add('hidden');
    updateRecordBtn();
    renderText();
    showMsg('文字起こしが完了しました。');
  };
  step();
}

els.fileInput.addEventListener('change', async () => {
  const file = els.fileInput.files && els.fileInput.files[0];
  els.fileInput.value = '';
  if (!file || fileBusy || recording) return;
  if (engineState !== 'ready') { await prepareEngine(); if (engineState !== 'ready') return; }
  fileBusy = true;
  els.fileProgress.classList.remove('hidden');
  els.fileProgress.textContent = '音声を読み込んでいます…';
  try {
    const ab = await file.arrayBuffer();
    const offCtx = new OfflineAudioContext(1, SR, SR);
    const audioBuf = await offCtx.decodeAudioData(ab);
    let samples;
    if (audioBuf.numberOfChannels === 1) {
      samples = audioBuf.getChannelData(0).slice();
    } else {
      const n = audioBuf.length;
      samples = new Float32Array(n);
      for (let ch = 0; ch < audioBuf.numberOfChannels; ch++) {
        const d = audioBuf.getChannelData(ch);
        for (let i = 0; i < n; i++) samples[i] += d[i] / audioBuf.numberOfChannels;
      }
    }
    sessionChunks = [samples]; sessionLength = samples.length;
    sherpaVad.reset();
    feedVadProgress(samples);
  } catch (e) {
    fileBusy = false;
    els.fileProgress.classList.add('hidden');
    showMsg('ファイルを読み込めませんでした: ' + (e && e.message ? e.message : e), true);
  }
});

/* ---------- テキスト操作 ---------- */
els.copyBtn.addEventListener('click', async () => {
  const t = currentText();
  if (!t) { showMsg('コピーするテキストがありません。', true); return; }
  try { await navigator.clipboard.writeText(t); showMsg('コピーしました。'); }
  catch (_) {
    const ta = document.createElement('textarea');
    ta.value = t; document.body.appendChild(ta); ta.select();
    document.execCommand('copy'); ta.remove();
    showMsg('コピーしました。');
  }
});
els.saveBtn.addEventListener('click', () => {
  const t = currentText();
  if (!t) { showMsg('保存するテキストがありません。', true); return; }
  const blob = new Blob([t], { type: 'text/plain;charset=utf-8' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'mojiokoshi-' + new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-') + '.txt';
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 3000);
});
els.clearBtn.addEventListener('click', () => {
  if (!confirm('文字起こしテキストとセッション音声を全て消去します。よろしいですか？')) return;
  finals = []; persistText();
  sessionChunks = []; sessionLength = 0;
  pendingChunks = []; pendingLength = 0;
  interimText = ''; els.interim.textContent = '';
  renderText();
  showMsg('消去しました。');
});

/* ---------- 簡易整形（ローカル） ---------- */
function localFormat(text) {
  let t = text;
  // フィラー除去（文頭・文中の典型的な言い淀み）
  t = t.replace(/(えーと|えっと|あのー|そのー|うーん|んー)/g, '');
  t = t.replace(/(^|\n)(あの|その)、/g, '$1');
  // 空白の整理
  t = t.replace(/[ \t　]+/g, ' ');
  // 行ごとに句点で閉じる
  t = t.split('\n').map((line) => {
    line = line.trim();
    if (!line) return line;
    if (!/[。！？!?]$/.test(line)) line += '。';
    return line;
  }).join('\n');
  return t;
}
function openPolish(title, text) {
  els.polishTitle.textContent = title;
  els.polishText.value = text;
  els.polishCard.classList.remove('hidden');
  els.polishCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}
els.formatBtn.addEventListener('click', () => {
  const t = currentText();
  if (!t) { showMsg('整形するテキストがありません。', true); return; }
  openPolish('簡易整形の結果（フィラー除去・句読点）', localFormat(t));
});
els.polishCloseBtn.addEventListener('click', () => els.polishCard.classList.add('hidden'));
els.polishApplyBtn.addEventListener('click', () => {
  const t = els.polishText.value.trim();
  finals = t ? t.split('\n') : [];
  persistText(); renderText();
  els.polishCard.classList.add('hidden');
  showMsg('置き換えました。');
});

/* ---------- AI整文（ユーザー自身のAPIキー、端末内のみ保存） ---------- */
function loadLlmCfg() {
  try { return JSON.parse(localStorage.getItem(LS_LLM) || '{}'); } catch (_) { return {}; }
}
function saveLlmCfg(cfg) { try { localStorage.setItem(LS_LLM, JSON.stringify(cfg)); } catch (_) {} }
els.llmSaveBtn.addEventListener('click', () => {
  saveLlmCfg({ endpoint: els.llmEndpoint.value.trim(), model: els.llmModel.value.trim(), key: els.llmKey.value.trim() });
  els.llmMsg.textContent = '保存しました（この端末のブラウザ内にのみ保存されます）。';
  els.llmMsg.classList.remove('hidden');
});
els.llmClearBtn.addEventListener('click', () => {
  saveLlmCfg({});
  els.llmEndpoint.value = ''; els.llmModel.value = ''; els.llmKey.value = '';
  els.llmMsg.textContent = 'キーと設定を消去しました。';
  els.llmMsg.classList.remove('hidden');
});
const LLM_PROMPT = 'あなたは日本語の文字起こしを整える編集者です。入力は音声認識の結果です。文脈を読んで、同音異義語の誤り・誤字脱字・助詞の誤りを直し、自然な句読点と段落を付けてください。フィラー（えーと、あの、など）は取り除いてください。意味や事実関係は変えず、要約や加筆はしないでください。整えた本文だけを出力してください。';
els.aiBtn.addEventListener('click', async () => {
  const cfg = loadLlmCfg();
  if (!cfg.key || !cfg.endpoint || !cfg.model) {
    els.settingsDetails.open = true;
    els.settingsDetails.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    els.llmMsg.textContent = 'AI整文を使うには、あなたのAPIキーとエンドポイント、モデル名を設定してください。キーはこの端末内にだけ保存されます。';
    els.llmMsg.classList.remove('hidden');
    return;
  }
  const t = currentText();
  if (!t) { showMsg('整文するテキストがありません。', true); return; }
  els.aiBtn.disabled = true;
  els.aiBtn.textContent = 'AI整文中…';
  try {
    const resp = await fetch(cfg.endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + cfg.key },
      body: JSON.stringify({
        model: cfg.model, temperature: 0.2,
        messages: [{ role: 'system', content: LLM_PROMPT }, { role: 'user', content: t }],
      }),
    });
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const data = await resp.json();
    const out = data && data.choices && data.choices[0] && data.choices[0].message && data.choices[0].message.content;
    if (!out) throw new Error('応答の形式が想定と違います');
    openPolish('AI整文の結果', out.trim());
  } catch (e) {
    showMsg('AI整文に失敗しました: ' + (e && e.message ? e.message : e) + '（キー、モデル名、エンドポイント、CORS設定を確認してください）', true);
  } finally {
    els.aiBtn.disabled = false;
    els.aiBtn.textContent = 'AIで整文';
  }
});

/* ---------- モード切替 ---------- */
function setMode(m) {
  if (recording) { showMsg('録音中は切り替えできません。先に停止してください。', true); return; }
  mode = m;
  els.modeOfflineBtn.classList.toggle('active', m === 'offline');
  els.modeCloudBtn.classList.toggle('active', m === 'cloud');
  els.offlinePanel.classList.toggle('hidden', m !== 'offline');
  els.cloudPanel.classList.toggle('hidden', m !== 'cloud');
  updateRecordBtn();
}
els.modeOfflineBtn.addEventListener('click', () => setMode('offline'));
els.modeCloudBtn.addEventListener('click', () => setMode('cloud'));

/* ---------- 初期化 ---------- */
(function init() {
  const cfg = loadLlmCfg();
  if (cfg.endpoint) els.llmEndpoint.value = cfg.endpoint;
  if (cfg.model) els.llmModel.value = cfg.model;
  if (cfg.key) els.llmKey.value = cfg.key;
  els.cloudSupport.textContent = cloudSupported()
    ? 'このブラウザはクラウドモード（Web Speech API）に対応しています。'
    : 'このブラウザはクラウドモード（Web Speech API）に対応していません。オフラインモードを使ってください。';
  if (!cloudSupported()) els.modeCloudBtn.disabled = true;
  // エンジン種別の選択（既定は高精度 Parakeet）
  const updateSizeNote = () => {
    els.engineSizeNote.textContent = engineKind === 'parakeet'
      ? 'モデル（約656MB）を初回だけダウンロードし、この端末にキャッシュします。2回目以降は機内モードでも動きます。音声は端末の外に出ません。'
      : 'モデル（約170MB）を初回だけダウンロードし、この端末にキャッシュします。2回目以降は機内モードでも動きます。音声は端末の外に出ません。';
  };
  els.engineKind.value = engineKind;
  updateSizeNote();
  els.engineKind.addEventListener('change', () => {
    engineKind = els.engineKind.value === 'zipformer' ? 'zipformer' : 'parakeet';
    try { localStorage.setItem(LS_ENGINE, engineKind); } catch (_) {}
    updateSizeNote();
    if (engineState === 'ready' || engineState === 'busy') location.reload(); // 準備済みエンジンと混ざらないよう再読み込み
  });
  // 選択中エンジンのモデルがキャッシュ済みか軽く確認して表示だけ更新（自動ダウンロードはしない）
  if ('caches' in window) {
    caches.open(MODEL_CACHE).then(async (cache) => {
      const probe = engineKind === 'parakeet' ? '__model__/parakeet-ja.onnx' : '__model__/transducer-encoder.onnx';
      const hit = await cache.match(new Request(new URL(probe, location.href).toString()));
      if (hit) {
        els.engineStatus.textContent = 'モデルはキャッシュ済みです。「エンジンを準備する」を押すとすぐに使えます。';
        els.prepareBtn.textContent = 'エンジンを準備する（キャッシュ済み）';
      }
    }).catch(() => {});
  }
  els.prepareBtn.addEventListener('click', prepareEngine);
  renderText();
  updateRecordBtn();
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('sw.js').catch((e) => console.warn('SW registration failed', e));
    // SW が COOP/COEP を付与できるようになったら一度だけ再読み込みし、
    // crossOriginIsolated（sherpa-onnx WASM のスレッド実行に必要）を有効化する。
    if (!window.crossOriginIsolated && !sessionStorage.getItem('mojiokoshi.coiReload')) {
      sessionStorage.setItem('mojiokoshi.coiReload', '1');
      const reload = () => location.reload();
      if (navigator.serviceWorker.controller) setTimeout(reload, 300);
      else navigator.serviceWorker.ready.then(() => setTimeout(reload, 300)).catch(() => {});
    }
  }
})();
