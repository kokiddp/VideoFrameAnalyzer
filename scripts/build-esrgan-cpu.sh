#!/usr/bin/env bash
# Builds native/esrgan_cpu.cpp (ncnn without Vulkan) and copies it next to Real-ESRGAN in assets/<os>/realesrgan/.
# Needs: git, cmake, a C++17 compiler with OpenMP. Usage: scripts/build-esrgan-cpu.sh [linux|windows]
set -euo pipefail
OS="${1:-linux}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${NCNN_WORK:-$(mktemp -d)}"
GEN=(); EXE=""; STATIC=(-static -static-libgcc -static-libstdc++)
if [ "$OS" = windows ]; then GEN=(-G "MinGW Makefiles"); EXE=".exe"; fi
# Linux: keep glibc dynamic but link the GCC runtime and OpenMP (libgomp) statically so the runner has no extra dependencies
if [ "$OS" = linux ]; then STATIC=(-static-libgcc -static-libstdc++ -Wl,-Bstatic -lgomp -Wl,-Bdynamic); fi
[ -d "$WORK/ncnn" ] || git clone --depth 1 https://github.com/Tencent/ncnn.git "$WORK/ncnn"
cmake -S "$WORK/ncnn" -B "$WORK/build" "${GEN[@]}" -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=OFF -DNCNN_BUILD_TOOLS=OFF \
  -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF -DNCNN_BUILD_TESTS=OFF -DNCNN_SHARED_LIB=OFF -DNCNN_OPENMP=ON \
  -DNCNN_SIMPLEOCV=ON -DNCNN_PYTHON=OFF
# an unbounded `-j` starts hundreds of compilers at once and can exhaust RAM (it crashed a 16 GB WSL VM):
# cap the jobs by CPU count and by memory (~1.5 GB per ncnn translation unit)
JOBS="${BUILD_JOBS:-$(nproc 2>/dev/null || echo 4)}"
if [ -z "${BUILD_JOBS:-}" ] && [ -r /proc/meminfo ]; then
  MEM_JOBS=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1536000 ))
  [ "$MEM_JOBS" -lt 1 ] && MEM_JOBS=1
  [ "$MEM_JOBS" -lt "$JOBS" ] && JOBS="$MEM_JOBS"
fi
cmake --build "$WORK/build" -j"$JOBS"
cmake --install "$WORK/build" --prefix "$WORK/inst"
mkdir -p "$ROOT/assets/$OS/realesrgan"
g++ -O3 -std=c++17 -fopenmp -I"$WORK/inst/include/ncnn" "$ROOT/native/esrgan_cpu.cpp" \
  -o "$ROOT/assets/$OS/realesrgan/esrgan-cpu$EXE" "$WORK/inst/lib/libncnn.a" "${STATIC[@]}"
echo "built assets/$OS/realesrgan/esrgan-cpu$EXE"
