# Third-party notices

Video Frame Analyzer's own source code is MIT-licensed (see `LICENSE`). A release executable additionally contains the
third-party components below. They keep their own licenses; this file lists them for convenience and is not a
substitute for the license texts shipped by each project.

## Components embedded in the executable (`assets/`)

| Component | Use | License | Source |
|---|---|---|---|
| **ffmpeg** (BtbN "master-latest-win64-gpl" / "linux64-gpl" builds) | decoding, filtering, frame export | **GPL v3** (the build enables GPL components such as x264 and libplacebo; ffmpeg itself is LGPL 2.1+/GPL) | https://ffmpeg.org · https://github.com/BtbN/FFmpeg-Builds |
| **Real-ESRGAN-ncnn-vulkan** v0.2.5.0 and its model files | neural upscaling (GPU) | MIT (models: BSD-3-Clause, Real-ESRGAN) | https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan |
| **waifu2x-ncnn-vulkan** 20250915 and its model files | neural upscaling / denoising (GPU) | MIT | https://github.com/nihui/waifu2x-ncnn-vulkan |
| **ncnn** | inference engine of `esrgan-cpu` (CPU runner) | BSD-3-Clause | https://github.com/Tencent/ncnn |
| **stb_image / stb_image_write** (vendored in `native/`) | image I/O in `esrgan-cpu` | MIT / public domain | https://github.com/nothings/stb |
| **PaddleOCR** PP-OCRv3 text detector and PP-OCRv5 "latin" recogniser (ONNX conversions from https://huggingface.co/monkt/paddleocr-onnx) | OCR of text in en, it, fr, de, nl, pt, es and other Latin-script languages | Apache-2.0 | https://github.com/PaddlePaddle/PaddleOCR |
| **open-image-models** `yolo-v9-t-416-license-plates-end2end` | licence-plate detection | MIT | https://github.com/ankandrew/open-image-models |
| **fast-plate-ocr** `european_mobile_vit_v2_ocr` | licence-plate recognition | MIT | https://github.com/ankandrew/fast-plate-ocr |
| Microsoft VC++ OpenMP runtime (`vcomp140.dll`) | shipped next to the Windows Vulkan tools | Microsoft redistributable | https://learn.microsoft.com/cpp/windows/redistributing-visual-cpp-files |

### GPL note

Because the bundled ffmpeg build is GPL-licensed, a binary distribution that includes it must be offered under terms
that satisfy the GPL for that component, including access to the corresponding ffmpeg source (see the BtbN build
page for the exact source revision of the build you bundled). If you need to distribute a binary without GPL
obligations, replace `assets/<os>/ffmpeg` with an LGPL-only build; some features (for example the `libplacebo` GPU
upscaler filter) may then be unavailable.

## Rust crates (statically linked)

The application links, among others: `eframe` / `egui` (MIT or Apache-2.0), `image` (MIT or Apache-2.0),
`rfd` (MIT), `tract-onnx` (ONNX inference for the OCR models; MIT or Apache-2.0), `zstd` and `tar` (MIT or Apache-2.0), `dirs` (MIT or Apache-2.0) and their transitive dependencies,
all permissively licensed. Run `cargo tree` for the complete list or use `cargo-about` / `cargo-license` to generate
the full set of license texts for a release.
