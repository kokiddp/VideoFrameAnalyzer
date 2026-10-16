#!/usr/bin/env bash
# Downloads ffmpeg + Real-ESRGAN into assets/<os>/ so build.rs can embed them. Usage: scripts/fetch-assets.sh [linux|windows]
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
rm -rf "$TMP"; echo "assets ready in $OUT"
