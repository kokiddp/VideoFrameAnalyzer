# Video Frame Analyzer

A portable desktop video player built for **inspecting a video frame by frame and exporting frames at the best
possible quality** — with optional live filters (sharpening, denoising, upscaling) and neural-network enhancement
(Real-ESRGAN and waifu2x) running on the GPU or the CPU.

* One self-contained executable for **Windows** and **Linux** — no installer, no external dependencies.
  ffmpeg and the neural upscalers are embedded in the binary and unpacked on first run.
* Opens **any format ffmpeg can read** (MP4, MKV, AVI, MOV, WebM, MPEG-TS, GIF, ...).
* Simple GUI: drag a file onto the window and play.

> Status: developed on Windows 11. The Linux build was validated on Ubuntu 24.04 (WSL2/WSLg): asset download,
> CPU-runner build, release build, test-suite (software Vulkan only) and application start-up all work — see
> [Platform notes](#platform-notes-and-limitations). Not yet tried on a native Linux desktop with a real GPU.

---

## Contents

1. [Features](#features)
2. [Using the program](#using-the-program)
3. [Keyboard shortcuts](#keyboard-shortcuts)
4. [Filters](#filters)
5. [Neural enhancement (Real-ESRGAN / waifu2x)](#neural-enhancement-real-esrgan--waifu2x)
6. [Exporting frames](#exporting-frames)
7. [How the embedded software works](#how-the-embedded-software-works)
8. [Building from source](#building-from-source)
9. [Development and tests](#development-and-tests)
10. [Architecture](#architecture)
11. [Platform notes and limitations](#platform-notes-and-limitations)
12. [Troubleshooting](#troubleshooting)
13. [License](#license)

---

## Features

**Playback**

* Play, pause, step one frame forward/backward, jump to start/end, **play in reverse**.
* **Speed from 0.05× to 16×**, set with a rotary knob, a logarithmic slider, a numeric field, or presets.
* **Frame-by-frame mode**: playback stops and a key press advances exactly one frame.
* Loop playback — seamless in both directions (the next pass is prepared while the current one finishes).
* Seek bar with a **CUE bar** above it: drag to select a sub-range; playback is then confined to that range until you
  clear the selection.
* Zoom with the mouse wheel, pan by dragging, double-click to fit; optional pixel-exact (nearest-neighbour) zoom.

**Frame export**

* PNG, TIFF, WebP (all lossless), BMP, and JPEG at maximum quality (4:4:4, no chroma subsampling).
* 16-bit PNG/TIFF output for sources with more than 8 bits per component.
* Export the current frame, N consecutive frames, or **the whole CUE range** in one click.
* Progress window with bar, elapsed time, a **measured ETA** (speed over a sliding window; for neural exports the
  average time of the frames done so far, shown as a smooth countdown) and a Cancel button.
* The export follows the preview: with **Apply** on, the active filters (and neural stage) are baked into the files;
  with **Apply** off, the original frames are written untouched.

**Image enhancement**

* Live filter chain (ffmpeg): deinterlace, two denoisers, Lanczos and GPU (libplacebo) upscaling, unsharp mask,
  contrast-adaptive sharpening (CAS), brightness/contrast/saturation/gamma.
* Neural enhancement of the paused frame: **Real-ESRGAN** (several models) and **waifu2x** (photo / anime / cunet
  models, denoise levels, 1×/2×/4×), each runnable on **GPU (Vulkan: NVIDIA, AMD, Intel) or CPU**.
* A start-up **benchmark** measures every processor per engine and picks the fastest as default; a **Benchmark window**
  compares models × resolutions × processors on demand.

**Quality of life**

* Log panel (docked bottom-right) with every ffmpeg / neural-engine command line and timing.
* Drag & drop, file dialog, remembers nothing outside its cache folder.
* The right-hand column is user-resizable and never changes width by itself.

---

## Using the program

1. Start the executable and **drag a video onto the window** (or press **Open video…** / `Ctrl+O`). You can also pass
   a file path on the command line.
2. Use the transport buttons under the picture (`|◀◀`, reverse play, step back, play/pause, step forward, `▶▶|`) or the
   keyboard shortcuts below.
3. **Speed**: turn the knob (drag up/right to go faster, mouse wheel to fine-tune, double-click for 1×), drag the
   logarithmic slider, type a value, or click a preset.
4. **Frame-by-frame**: tick *Frame by frame* (or press `F`). `Space`, `Enter` or `→` advance one frame, `←` goes back.
5. **CUE range**: drag on the thin bar above the seek bar to select a range (drag the yellow edges to adjust, click to
   clear; `I`/`O` set the in/out points at the current frame, `Esc` clears). While a range is set, play/loop/reverse
   stay inside it, and an **Export range (N)** button appears.
6. **Export**: choose a format, press **Export current frame** (`E` or `Ctrl+S`). Files go to
   `<video folder>/<video name>_frames/` unless you pick another folder.

While the video **plays**, the filters are automatically suspended (the *Apply* box unticks itself and the filter
controls are disabled) so playback stays smooth; they come back on pause and are always active in frame-by-frame mode.
A pill with a spinner at the top-right of the picture tells you when a frame is still being processed.

---

## Keyboard shortcuts

| Key | Action |
|---|---|
| `Space` | Play / pause (in frame-by-frame mode: next frame) |
| `←` / `→` (`,` / `.`) | Previous / next frame. `Shift` = ±10 frames, `Ctrl` = ±1 second |
| `Home` / `End` | First / last frame |
| `F` | Toggle frame-by-frame mode (`Enter` also advances) |
| `E` or `Ctrl+S` | Export the current frame (or the N consecutive frames set in the panel) |
| `I` / `O` | Set CUE in-point / out-point at the current frame |
| `Esc` | Clear the CUE range |
| `C` (hold) | Show the original frame (A/B comparison, also bypasses the neural stage) |
| `L` | Show / hide the log panel |
| `Ctrl+O` | Open a video |
| Mouse wheel / drag / double-click | Zoom / pan / fit to window |

---

## Filters

Filters are applied in this fixed order, each one can be switched on independently; the **Apply** master switch turns
the whole chain on or off, and **All on / All off** change every filter at once.

| Filter | ffmpeg filter | Notes |
|---|---|---|
| Deinterlace | `bwdif` / `yadif` | `send_frame` mode keeps the frame count |
| Denoise (fast) | `hqdn3d` | one strength slider |
| Denoise (quality, slow) | `nlmeans` | high quality, expensive |
| Lanczos upscale | `scale ... lanczos` | factor 1–4, CPU |
| GPU upscale | `libplacebo` | EWA / spline upscalers via Vulkan |
| Sharpness (unsharp mask) | `unsharp` | amount 0–3, radius 3–13 px |
| Adaptive sharpening | `cas` | AMD-style contrast-adaptive sharpening |
| Colour | `eq` | brightness, contrast, saturation, gamma |

Behaviour worth knowing:

* **Debounce** – a slider change is applied only after you release the mouse (and the value has been stable for a
  moment), never while dragging.
* **Locking** – while a new chain is being applied the filter controls are disabled until the frame on screen reflects
  it (they unlock automatically if ffmpeg rejects the chain, or after 30 s).
* All sliders are clamped to sensible ranges, and every row has a *Reset* button.

---

## Neural enhancement (Real-ESRGAN / waifu2x)

The neural stage runs **on the paused frame** (it is too heavy for real-time playback), after the ffmpeg chain, and
its result is what you see and what gets exported when it is enabled.

| Engine | Models | Options |
|---|---|---|
| **Real-ESRGAN** (ncnn) | `realesr-animevideov3` ×2/×3/×4 (fast), `realesrgan-x4plus` (best quality for real footage), `realesrgan-x4plus-anime` | fixed factor per model |
| **waifu2x** (ncnn) | `upconv_7_photo` (fast, photos/video), `upconv_7_anime_style_art_rgb` (fast, drawings), `cunet` (best, slower) | denoise level none/0–3, scale 1× (denoise only, cunet), 2×, 4× |

**Processor choice.** Each engine can run on any Vulkan GPU or on the CPU. The CPU path is a small runner included in
this project (`native/esrgan_cpu.cpp`, built on ncnn without Vulkan) implementing both engines with the same
algorithms and models as the official Vulkan tools; its output matches the GPU output (mean difference below 0.1 of
255 for waifu2x, ~52 dB PSNR for Real-ESRGAN).

**Default processor.** On first start (and whenever you press *Test*) the program times a 640×360 frame on every
processor for each engine and selects the fastest. The result is cached. Typical findings on a desktop with an
integrated Intel GPU: Real-ESRGAN is faster on the GPU, waifu2x is faster on the CPU (the official Vulkan tool
recompiles its shaders at every launch). With a discrete NVIDIA/AMD GPU the GPU normally wins.

**Benchmark window** (button *Benchmark…*). Choose models, resolutions (360p/720p/1080p), processors and a time limit
per trial; a live table fills in with the time per frame, the fastest processor of each row highlighted. Trials that
exceed the limit are marked `> N s` and larger sizes of the same model/processor are skipped. Results can be copied as
Markdown or CSV. The test image is the current frame (or a synthetic one if no video is open).

> NVIDIA *RTX Video Super Resolution* is **not** supported: it needs NVIDIA's proprietary SDK and cannot be bundled
> in a portable executable. Real-ESRGAN/waifu2x over Vulkan cover NVIDIA, AMD and Intel GPUs.

---

## Exporting frames

* **Formats**: PNG (compression level 9), TIFF (deflate), WebP lossless, BMP, JPEG (`-q:v 1`, 4:4:4).
* **Bit depth**: sources with more than 8 bits per component are written as 16-bit PNG/TIFF; everything goes through
  an accurate colour conversion (`accurate_rnd`, `full_chroma_int`, Lanczos chroma) rather than the faster preview one.
* **What is exported** is decided by the **Apply** switch: on → active filters and (if enabled) the neural stage;
  off → the original frames. The panel states what will be written.
* **Naming**: `<video>_frame_000123.png` where the number is the frame index.
* **Ranges**: *Consecutive frames* (any count) or **Export range** for the CUE selection. A progress window shows the
  bar, elapsed time and estimated remaining time and lets you cancel. The estimate is computed from the measured speed
  (typically within a few percent of the real remaining time once ~1 s of data exists; neural exports show it after
  the first frame is processed).
* Frame positions assume a constant frame rate (see [limitations](#platform-notes-and-limitations)).

---

## How the embedded software works

The executable contains ffmpeg, Real-ESRGAN, waifu2x and the CPU runner **inside itself**, and **does not download
anything at run time**.

1. **Before compiling**, `scripts/fetch-assets.sh` downloads the third-party binaries into `assets/<windows|linux>/`
   and `scripts/build-esrgan-cpu.sh` compiles the CPU runner there. (`assets/` is git-ignored because of its size.)
2. **At compile time**, `build.rs` packs that folder into one zstd-compressed tar (level 19 in release builds) which
   `include_bytes!` embeds in the executable (about 136 MB for the Windows build).
3. **On first run** the program unpacks the bundle into its cache folder —
   `%LOCALAPPDATA%\VideoFrameAnalyzer\<bundle-id>\` on Windows, `~/.cache/VideoFrameAnalyzer/<bundle-id>/` on Linux —
   and reuses it afterwards. A different bundle (new version) is unpacked next to it and stale folders are removed.
   The cache also holds `tmp/` (scratch files) and `bench.txt` (benchmark results). It is safe to delete.

If no bundle was embedded (a build without `assets/`), the program falls back to `ffmpeg` on the `PATH`; see
[Development and tests](#development-and-tests) for the override variables.

---

## Building from source

**Requirements**

* Rust (stable) with a working linker. On Windows either the MSVC build tools or MinGW-w64 (the project is developed
  with the `stable-x86_64-pc-windows-gnu` toolchain + WinLibs).
* To build the CPU runner: `git`, `cmake`, and a C++17 compiler with OpenMP (g++).
* `curl`, `unzip` (and `xz` on Linux) for the asset script.
* Linux GUI libraries for eframe/winit/glow (`libxkbcommon`, `libwayland`, `libx11`, `libxcb`, OpenGL); the
  resulting binary itself links only against glibc (`libc`, `libm`, `libgcc_s`) and loads the GUI libraries at run time.

**Steps**

```sh
# 1. fetch ffmpeg + Real-ESRGAN + waifu2x into assets/<os>/
scripts/fetch-assets.sh windows        # or: linux

# 2. build the CPU runner (ncnn without Vulkan) into assets/<os>/realesrgan/
scripts/build-esrgan-cpu.sh windows    # or: linux

# 3. build the single executable
cargo build --release                  # -> target/release/video-frame-analyzer(.exe)
```

The release profile uses LTO and `panic = "abort"`. The first build compresses ~170 MB of assets and takes a little
longer (about 1–2 minutes on a modern machine). Building ncnn for the CPU runner takes ~2 minutes with 6 jobs; the
script caps the parallel jobs by CPU count and free memory (an unbounded `make -j` can exhaust RAM) — override with
`BUILD_JOBS=n scripts/build-esrgan-cpu.sh linux`. On Linux the runner links libgomp (OpenMP) and the GCC runtime
statically. Two GitHub Actions workflows run the same steps for Windows and Linux: `build.yml` on every branch push, and
`release.yml` when a version tag is pushed (see below).

**Releasing.** Set `version` in `Cargo.toml`, commit, then push a tag with the same number:

```sh
git tag 0.1.0 && git push origin main 0.1.0     # a leading "v" (v0.1.0) works too
```

`release.yml` checks that tag and `Cargo.toml` agree, builds `video-frame-analyzer-<version>-windows-x86_64.zip` and
`-linux-x86_64.tar.gz` (executable + LICENSE + THIRD_PARTY_NOTICES + README, with `.sha256` files) and publishes them as
a GitHub Release with generated notes.

---

## Development and tests

Skip the (slow) bundling while developing and point the program at local tools:

```sh
export VFA_NO_BUNDLE=1                                   # build.rs: do not embed assets
export VFA_FFMPEG=/path/to/ffmpeg
export VFA_REALESRGAN=/path/to/realesrgan-ncnn-vulkan    # optional
export VFA_ESRGAN_CPU=/path/to/esrgan-cpu                # optional
export VFA_WAIFU2X=/path/to/waifu2x-ncnn-vulkan          # optional
cargo run
cargo test --release -- --test-threads=1
```

The test-suite (`src/selftest.rs`) needs `VFA_FFMPEG` and silently skips what it cannot run. It generates its own test
videos and covers, among other things:

* exact seeking, frame count and size detection, export pixel-identity with the preview;
* end-of-video handling, forward/reverse **looping** with a wrong frame-count estimate, reverse playback smoothness;
* a **random-operation stress test** (seeks, speed and direction changes, pauses) that reports any picture freeze;
* filter-chain generations, texture/frame consistency, export with and without filters, export progress/cancel;
* Real-ESRGAN and waifu2x on GPU vs CPU, cancellation, benchmark logic, and the fixed width of the right-hand column.

Useful environment variables for the stress test: `VFA_SEED`, `VFA_SECS`, `VFA_REVERSE_VIDEO` (use your own file).

---

## Architecture

```
src/
  main.rs      window setup
  app.rs       egui application: panels, shortcuts, export / benchmark windows
  video.rs     one opened video: frame cache, decoder streams, block loading, texture, AI stage driver
  player.rs    playback clock (speed, direction, loop, CUE range) – UI independent
  decoder.rs   ffmpeg process → raw RGBA frames over a pipe
  probe.rs     video metadata parsed from `ffmpeg -i` (no ffprobe needed)
  filters.rs   filter definitions → ffmpeg -vf chain; AI settings
  export.rs    full-quality frame export, progress and cancellation
  nn.rs        Real-ESRGAN / waifu2x process control, CPU/GPU selection, start-up benchmark
  bench.rs     interactive benchmark engine
  tools.rs     locating / unpacking the embedded tools
  widgets.rs   knob, transport icons, CUE + seek bars, filter cards
  log.rs       in-memory log shown in the UI
native/        esrgan_cpu.cpp – CPU runner for both neural engines (+ vendored stb headers)
scripts/       fetch-assets.sh, build-esrgan-cpu.sh
```

Key design points:

* Playback decodes with a persistent ffmpeg process; seeks and direction changes use short accurate-seek blocks.
  **Reverse playback** keeps decoding ahead of the playhead and prefetches the wrap-around block for seamless loops.
* A frame cache (≈900 MB budget) keeps recent frames for instant stepping back; eviction never touches the
  neighbourhood of the wanted frame.
* The playback clock holds while a seek is being served, so the player never runs away from the decoder.

---

## Platform notes and limitations

* **Windows** is the primary platform. **Linux** was validated on Ubuntu 24.04 under WSL2: asset script, CPU runner,
  release build (~150 MB binary), the full test-suite and a real start-up with bundle extraction, benchmark and
  decoding. There the only "GPU" is the llvmpipe software Vulkan device, so GPU speed on Linux is untested; real
  Vulkan GPUs need a working driver. File dialogs use the XDG desktop portal. The CI workflow
  (`.github/workflows/build.yml`) has not been run on GitHub yet.
* **No audio** – this is an analysis tool; sound is neither decoded nor played.
* **Constant frame rate is assumed** for positioning; variable-frame-rate files may land a few frames off. Frame
  counts come from the container duration and are corrected when the real end of the file is reached.
* Neural enhancement applies to the **paused** frame only. Heavy models (`x4plus`) take many seconds per 1080p frame.
* Reverse playback at high speed (≥3×) on heavy codecs/resolutions is best-effort.
* Very large neural outputs (e.g. ×4 of 1080p = 8K) can exceed the GPU's maximum texture size; the app then shows an
  error instead of the result (export still works).
* Distribution: the bundled ffmpeg is a **GPL** build, so a distributed executable is subject to the GPL for that
  component — see [License](#license).

---

## Troubleshooting

* **Open the log** (`L` or the *Log* button): it lists every ffmpeg/neural-engine command, their output and timings.
  *Copy* puts it on the clipboard for bug reports.
* **Neural enhancement does nothing / is slow**: check the *Processor* box in the neural card; press *Test* to
  re-measure. `Real-ESRGAN fallito ... GPU Vulkan disponibile?` in the log means no usable Vulkan device — choose the CPU.
* **A filter shows an ffmpeg error**: the chain was rejected (for example `libplacebo` without Vulkan); disable that
  filter. The controls unlock by themselves.
* **Reset everything**: close the program and delete the cache folder (see above).
* **First start is slow**: the ~170 MB bundle is being unpacked once.

---

## License

The source code of this project is released under the **MIT License** — see [`LICENSE`](LICENSE).

The executable bundles third-party software under its own licenses (notably **ffmpeg, GPL**, and MIT/BSD components);
see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). If you distribute a binary that includes the GPL ffmpeg build,
you must comply with the GPL for that component (including providing the corresponding source). Use an LGPL ffmpeg
build in `assets/` if you need a binary that is not GPL-encumbered (some filters, such as `libplacebo`, may then be
unavailable).
