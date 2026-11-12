# サードパーティ構成要素と出典

## ランタイム（リポジトリ同梱: vendor/sherpa/）

- **sherpa-onnx WebAssembly (v1.13.2, SIMD)** — Apache License 2.0
  - 出典: https://github.com/k2-fsa/sherpa-onnx/releases/tag/v1.13.2
    （公式パッケージ `sherpa-onnx-wasm-simd-1.13.2-vad-asr-ja-zipformer_reazonspeech.tar.bz2` から、モデルデータ `.data` を除いたランタイムファイルのみを同梱）
  - ファイル: `sherpa-onnx-wasm-main-vad-asr.js` / `.wasm`、`sherpa-onnx-asr.js`、`sherpa-onnx-vad.js`
  - アプリ側コードの一部（ダウンサンプル処理・VAD連携の流れ）は公式デモ `app-vad-asr.js`（同 Apache 2.0）を参考に再構成

## モデル（リポジトリに含めず、初回起動時に取得）

- **ReazonSpeech Zipformer（日本語）** — モデル: Reazon 人工知能研究所 / k2-fsa コミュニティによる ONNX 変換
  - 取得元: https://huggingface.co/csukuangfj/reazonspeech-k2-v2
    （encoder int8 / decoder / joiner int8 / tokens.txt）
  - ReazonSpeech コーパスは CC BY 4.0 で公開されています。
  - 取得ファイルの SHA-256 が公式WASMパッケージ同梱分と一致することをアプリ側で検証します。
    - encoder.int8.onnx: `2c7bd08a8a99f9ddd0d9e458456577b1f6279214e51426f114f9eced44c54e1d`
    - decoder.onnx: `58b18211ae06265466bfa17172dab574df94f76c8bcb61a3640c28ba860e4124`
    - joiner.int8.onnx: `49cc7ea1d3d35a40a27442db5e89996da64bf0e683a903dce76e99e57a12e4de`
    - tokens.txt: `2c3ac659818a48a0c04010e0593bbc4d7c8a24a054340b01131499c05fd52def`
- **Silero VAD** — MIT License（Silero Team）
  - 取得元: 公式デモ用HF Space `k2-fsa/web-assembly-vad-sherpa-onnx` の `sherpa-onnx-wasm-main-vad.data` 内の `/silero_vad.onnx`（先頭1076バイトのREADMEを除いた部分）
    - ※ GitHub Releases の同一ファイルは CORS 非対応でブラウザから取得できないため、この経路を使います。
    - silero_vad.onnx SHA-256: `9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6`

- **Parakeet 日本語専用モデル（高精度エンジン）** — NVIDIA NeMo Parakeet TDT-CTC 0.6B 日本語版の sherpa-onnx 変換（int8）
  - 取得元: https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8
    （model.int8.onnx / tokens.txt）
    - model.int8.onnx SHA-256: `3addd00ef5bd1742078389e540b77394e4a508bdf2f4c9ad1b4a76d93e76598e`
    - tokens.txt SHA-256: `732f64c53909f2620c713f4106b487d92e6f54a6915b3cd3d1dbd32f9f4f392a`
  - 実行方法: 同梱ランタイム（vad-asr パッケージ）の OfflineRecognizer に nemoCtc 設定で読み込む。公式の Parakeet 用 WASM パッケージは存在しないため、この構成が動くことをヘッドレスブラウザで実測して採用した（2026-11-13）。ランタイムの .data は Silero VAD 分のみ実データで、残りはゼロ埋め（このモードでは transducer 用ファイルは読まれない）。
  - 参考: 公式 Parakeet v3（欧州言語版）は日本語に非対応であることを同じサンプルで実測済み。本アプリが使うのは日本語専用モデルである。

## 合計ダウンロード量

- 高精度エンジン（Parakeet）: 約656MB（モデル 655.5MB + VAD 等）
- 軽量エンジン（Zipformer）: 約170MB
- いずれも端末の Cache Storage に保存され、以降はオフライン動作します。
