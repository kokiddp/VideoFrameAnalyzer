#!/usr/bin/env bash
# Downloads ffmpeg + Real-ESRGAN + waifu2x + OCR models into assets/<os>/ so build.rs can embed them. Usage: scripts/fetch-assets.sh [linux|windows]
set -euo pipefail
OS="${1:-linux}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/assets/$OS"; TMP="$(mktemp -d)"
mkdir -p "$OUT/realesrgan"
FF_BASE=https://github.com/BtbN/FFmpeg-Builds/releases/download/latest
RE_BASE=https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0
if [ "$OS" = linux ]; then
  curl -L "$FF_BASE/ffmpeg-master-latest-linux64-gpl.tar.xz" | tar -xJ -C "$TMP"
  cp "$TMP"/ffmpeg-*/bin/ffmpeg "$OUT/"
  curl -L -o "$TMP/re.zip" "$RE_BASE/realesrgan-ncnn-vulkan-20220424-ubuntu.zip"
else
  curl -L -o "$TMP/ff.zip" "$FF_BASE/ffmpeg-master-latest-win64-gpl.zip"; unzip -q "$TMP/ff.zip" -d "$TMP"
  cp "$TMP"/ffmpeg-*/bin/ffmpeg.exe "$OUT/"
  curl -L -o "$TMP/re.zip" "$RE_BASE/realesrgan-ncnn-vulkan-20220424-windows.zip"
fi
unzip -q "$TMP/re.zip" -d "$TMP/re"
RE_DIR="$(dirname "$(find "$TMP/re" -name 'realesrgan-ncnn-vulkan*' -type f ! -name '*.md' | head -1)")"
cp -r "$RE_DIR"/. "$OUT/realesrgan/"
rm -f "$OUT"/realesrgan/input*.jpg "$OUT"/realesrgan/*.mp4 "$OUT"/realesrgan/README*
chmod +x "$OUT/ffmpeg" "$OUT"/realesrgan/realesrgan-ncnn-vulkan 2>/dev/null || true
# waifu2x-ncnn-vulkan
W2X=https://github.com/nihui/waifu2x-ncnn-vulkan/releases/download/20250915/waifu2x-ncnn-vulkan-20250915-$OS.zip
curl -L -o "$TMP/w2x.zip" "$W2X" && unzip -q "$TMP/w2x.zip" -d "$TMP/w2x"
W_DIR="$(dirname "$(find "$TMP/w2x" -name 'waifu2x-ncnn-vulkan*' -type f ! -name '*.md' | head -1)")"
mkdir -p "$OUT/waifu2x" && cp -r "$W_DIR"/. "$OUT/waifu2x/" && rm -f "$OUT"/waifu2x/README*
chmod +x "$OUT"/waifu2x/waifu2x-ncnn-vulkan 2>/dev/null || true
# OCR models (ONNX, run in-process by the pure-Rust `tract` runtime): PaddleOCR text detector + PP-OCRv5 "latin"
# recogniser (en, it, fr, de, nl, pt, es, ...), and the licence-plate detector / European plate recogniser (MIT).
OCR="$OUT/ocr"; mkdir -p "$OCR"
HF=https://huggingface.co/monkt/paddleocr-onnx/resolve/7b02d0a30a07ba2b92ad1ff5a8941ae2c633de65
curl -fL -o "$OCR/text_det.onnx" "$HF/detection/v3/det.onnx"
curl -fL -o "$OCR/text_rec.onnx" "$HF/languages/latin/rec.onnx"
curl -fL -o "$OCR/text_dict.txt" "$HF/languages/latin/dict.txt"
curl -fL -o "$OCR/plate_det.onnx" "https://github.com/ankandrew/open-image-models/releases/download/assets/yolo-v9-t-416-license-plates-end2end.onnx"
curl -fL -o "$OCR/plate_rec.onnx" "https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates/european_mobile_vit_v2_ocr.onnx"
rm -rf "$TMP"; echo "assets ready in $OUT"
