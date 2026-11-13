'use strict';
/* もじ起こしライブ Service Worker
 * アプリ本体（シェル＋sherpa-onnxランタイム）をキャッシュし、オフライン起動を可能にする。
 * モデルファイル（約170MB）はアプリ側が Cache Storage (mojiokoshi-models-v1) で管理するため、
 * ここでは同一オリジンのリクエストだけを扱い、外部オリジンには一切介入しない。
 */
const CACHE = 'mojiokoshi-shell-v6';
const ASSETS = [
  './',
  './index.html',
  './app.js',
  './manifest.webmanifest',
  './favicon-32.png',
  './icon-192.png',
  './icon-512.png',
  './apple-touch-icon.png',
  './vendor/sherpa/sherpa-onnx-vad.js',
  './vendor/sherpa/sherpa-onnx-asr.js',
  './vendor/sherpa/sherpa-onnx-wasm-main-vad-asr.js',
  './vendor/sherpa/sherpa-onnx-wasm-main-vad-asr.wasm',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE).then((cache) => cache.addAll(ASSETS)).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k.startsWith('mojiokoshi-shell-') && k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

/* GitHub Pages 等は COOP/COEP ヘッダを付けられないため、SW が応答に付与して
 * crossOriginIsolated（SharedArrayBuffer 有効）を成立させる（coi-serviceworker と同方式）。
 * sherpa-onnx WASM のスレッド実行に必須。SW 制御下の2回目以降の読み込みで有効になる。 */
function withIsolationHeaders(resp) {
  if (!resp) return resp;
  const headers = new Headers(resp.headers);
  headers.set('Cross-Origin-Opener-Policy', 'same-origin');
  headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
  return new Response(resp.body, { status: resp.status, statusText: resp.statusText, headers });
}

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);
  if (url.origin !== self.location.origin) return; // モデル取得など外部は素通し（CORSで取得され、COEPとも両立）
  if (event.request.mode === 'navigate') {
    // 画面遷移はネット優先（更新を取りこぼさない）、失敗時はキャッシュ
    event.respondWith(
      fetch(event.request)
        .then((resp) => {
          const copy = resp.clone();
          caches.open(CACHE).then((cache) => cache.put('./index.html', copy));
          return withIsolationHeaders(resp);
        })
        .catch(() => caches.match('./index.html').then((hit) => withIsolationHeaders(hit)))
    );
    return;
  }
  if (event.request.method !== 'GET') return;
  event.respondWith(
    caches.match(event.request, { ignoreSearch: true }).then((hit) => {
      if (hit) return withIsolationHeaders(hit);
      return fetch(event.request).then((resp) => {
        if (resp && resp.ok) {
          const copy = resp.clone();
          caches.open(CACHE).then((cache) => cache.put(event.request, copy));
        }
        return withIsolationHeaders(resp);
      });
    })
  );
});
