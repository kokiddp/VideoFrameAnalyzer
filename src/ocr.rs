//! OCR engine: scene-text detection + recognition (PaddleOCR PP-OCRv5 "latin": en, it, fr, de, nl, pt, es, ...)
//! and licence-plate detection + recognition, as ONNX models on ONNX Runtime (DirectML GPU on Windows, CPU elsewhere).
use ort::{environment::Environment, session::Session, value::Tensor};
use std::path::Path;
use std::sync::OnceLock;

/// Model files expected in the OCR directory.
pub const FILES: [&str; 5] = ["text_det.onnx", "text_rec.onnx", "text_dict.txt", "plate_det.onnx", "plate_rec.onnx"];
/// ONNX Runtime library shipped next to the models.
pub const RUNTIME: &str = if cfg!(windows) { "onnxruntime.dll" } else { "libonnxruntime.so" };
/// Side of the (square) input of the plate detector.
const PLATE_DET_SIZE: usize = 608;
const PLATE_ALPHABET: &[u8] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_";
const PLATE_SLOTS: usize = 9;
const PLATE_W: usize = 140;
const PLATE_H: usize = 70;
const REC_H: usize = 48;
const REC_WIDTHS: [usize; 4] = [160, 320, 640, 1280];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rect {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

impl Rect {
    pub fn w(&self) -> f32 {
        (self.x1 - self.x0).max(0.0)
    }
    pub fn h(&self) -> f32 {
        (self.y1 - self.y0).max(0.0)
    }
    pub fn center(&self) -> (f32, f32) {
        ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)
    }
    pub fn iou(&self, o: &Rect) -> f32 {
        let iw = (self.x1.min(o.x1) - self.x0.max(o.x0)).max(0.0);
        let ih = (self.y1.min(o.y1) - self.y0.max(o.y0)).max(0.0);
        let inter = iw * ih;
        let union = self.w() * self.h() + o.w() * o.h() - inter;
        if union <= 0.0 {
            0.0
        } else {
            inter / union
        }
    }
}

/// One recognised string with its position (in the coordinates of the analysed image) and confidence 0..1.
#[derive(Clone, Debug)]
pub struct Found {
    pub rect: Rect,
    pub text: String,
    pub conf: f32,
    /// Plates: probability of every character in every slot (`PLATE_SLOTS × alphabet`), empty for text.
    pub slots: Vec<f32>,
    /// Plates: sharpness of the crop, 0..1.
    pub quality: f32,
}

/// Packed RGB image.
#[derive(Clone)]
pub struct Img {
    pub w: usize,
    pub h: usize,
    pub rgb: Vec<u8>,
}

impl Img {
    pub fn from_rgba(w: usize, h: usize, rgba: &[u8]) -> Img {
        let mut rgb = Vec::with_capacity(w * h * 3);
        for px in rgba.chunks_exact(4).take(w * h) {
            rgb.extend_from_slice(&px[..3]);
        }
        Img { w, h, rgb }
    }

    /// Crop (coordinates are clamped to the image; always at least 1×1).
    pub fn crop(&self, r: Rect) -> Img {
        let x0 = (r.x0.floor().max(0.0) as usize).min(self.w.saturating_sub(1));
        let y0 = (r.y0.floor().max(0.0) as usize).min(self.h.saturating_sub(1));
        let x1 = (r.x1.ceil().max(0.0) as usize).clamp(x0 + 1, self.w);
        let y1 = (r.y1.ceil().max(0.0) as usize).clamp(y0 + 1, self.h);
        let mut rgb = Vec::with_capacity((x1 - x0) * (y1 - y0) * 3);
        for y in y0..y1 {
            rgb.extend_from_slice(&self.rgb[(y * self.w + x0) * 3..(y * self.w + x1) * 3]);
        }
        Img { w: x1 - x0, h: y1 - y0, rgb }
    }

    fn rotate_ccw(&self) -> Img {
        let (w, h) = (self.h, self.w);
        let mut rgb = vec![0u8; w * h * 3];
        for y in 0..self.h {
            for x in 0..self.w {
                let (nx, ny) = (y, self.w - 1 - x);
                rgb[(ny * w + nx) * 3..(ny * w + nx) * 3 + 3].copy_from_slice(&self.rgb[(y * self.w + x) * 3..(y * self.w + x) * 3 + 3]);
            }
        }
        Img { w, h, rgb }
    }

    fn halve(&self, hx: bool, hy: bool) -> Img {
        let (w, h) = (if hx { self.w / 2 } else { self.w }, if hy { self.h / 2 } else { self.h });
        let (sx, sy) = (if hx { 2 } else { 1 }, if hy { 2 } else { 1 });
        let mut rgb = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    let mut sum = 0u32;
                    for dy in 0..sy {
                        for dx in 0..sx {
                            sum += self.rgb[((y * sy + dy) * self.w + x * sx + dx) * 3 + c] as u32;
                        }
                    }
                    rgb[(y * w + x) * 3 + c] = (sum / (sx * sy) as u32) as u8;
                }
            }
        }
        Img { w, h, rgb }
    }

    /// Bilinear resize (box-averaged first when shrinking by more than 2×).
    pub fn resize(&self, dw: usize, dh: usize) -> Img {
        let (dw, dh) = (dw.max(1), dh.max(1));
        let mut cur = self.clone();
        loop {
            let (hx, hy) = (cur.w >= 2 * dw && cur.w >= 2, cur.h >= 2 * dh && cur.h >= 2);
            if !hx && !hy {
                break;
            }
            cur = cur.halve(hx, hy);
        }
        if cur.w == dw && cur.h == dh {
            return cur;
        }
        let mut rgb = vec![0u8; dw * dh * 3];
        let (fx, fy) = (cur.w as f32 / dw as f32, cur.h as f32 / dh as f32);
        for y in 0..dh {
            let sy = ((y as f32 + 0.5) * fy - 0.5).clamp(0.0, (cur.h - 1) as f32);
            let (y0, ty) = (sy.floor() as usize, sy.fract());
            let y1 = (y0 + 1).min(cur.h - 1);
            for x in 0..dw {
                let sx = ((x as f32 + 0.5) * fx - 0.5).clamp(0.0, (cur.w - 1) as f32);
                let (x0, tx) = (sx.floor() as usize, sx.fract());
                let x1 = (x0 + 1).min(cur.w - 1);
                for c in 0..3 {
                    let p = |xx: usize, yy: usize| cur.rgb[(yy * cur.w + xx) * 3 + c] as f32;
                    let top = p(x0, y0) * (1.0 - tx) + p(x1, y0) * tx;
                    let bot = p(x0, y1) * (1.0 - tx) + p(x1, y1) * tx;
                    rgb[(y * dw + x) * 3 + c] = (top * (1.0 - ty) + bot * ty + 0.5) as u8;
                }
            }
        }
        Img { w: dw, h: dh, rgb }
    }
}

fn es<E: std::fmt::Display>(e: E) -> String {
    format!("{e}")
}

/// The process-wide ONNX Runtime environment, loaded from the library next to the models.
fn environment(dir: &Path) -> Result<&'static Environment, String> {
    static ENV: OnceLock<Result<Environment, String>> = OnceLock::new();
    ENV.get_or_init(|| {
        // DirectML.dll must already be loaded when the DirectML provider starts: load it by full path
        #[cfg(windows)]
        {
            let dml = dir.join("DirectML.dll");
            if dml.is_file() {
                if let Ok(lib) = unsafe { libloading::Library::new(&dml) } {
                    std::mem::forget(lib);
                }
            }
        }
        ort::init_from(dir.join(RUNTIME)).map_err(es)?.build().map_err(es)
    })
    .as_ref()
    .map_err(|e| e.clone())
}

/// An ONNX session and the name of its (single) input.
struct Model {
    session: Session,
    input: String,
}

impl Model {
    fn open(dir: &Path, file: &str, gpu: bool, threads: usize) -> Result<Model, String> {
        let env = environment(dir)?;
        let mut b = Session::builder(env).map_err(es)?;
        #[cfg(windows)]
        if gpu {
            use ort::ep::directml::PerformancePreference;
            b = b
                .with_execution_providers([ort::ep::DirectML::default().with_performance_preference(PerformancePreference::HighPerformance).build().error_on_failure()])
                .map_err(es)?
                .with_memory_pattern(false)
                .map_err(es)?
                .with_parallel_execution(false)
                .map_err(es)?;
        }
        let _ = gpu;
        let session = b.with_intra_threads(threads.max(1)).map_err(es)?.commit_from_file(dir.join(file)).map_err(|e| format!("{file}: {e}"))?;
        let input = session.inputs()[0].name().to_string();
        Ok(Model { session, input })
    }

    fn run_f32(&mut self, shape: &[usize], data: &[f32]) -> Result<(Vec<usize>, Vec<f32>), String> {
        let t = Tensor::from_array((shape.to_vec(), data.to_vec())).map_err(es)?;
        let outs = self.session.run(ort::inputs![self.input.as_str() => t]).map_err(es)?;
        let (sh, v) = outs[0].try_extract_tensor::<f32>().map_err(es)?;
        Ok((sh.iter().map(|&d| d as usize).collect(), v.to_vec()))
    }

    fn run_u8(&mut self, shape: &[usize], data: &[u8]) -> Result<Vec<f32>, String> {
        let t = Tensor::from_array((shape.to_vec(), data.to_vec())).map_err(es)?;
        let outs = self.session.run(ort::inputs![self.input.as_str() => t]).map_err(es)?;
        let (_, v) = outs[0].try_extract_tensor::<f32>().map_err(es)?;
        Ok(v.to_vec())
    }
}

pub struct Engine {
    dir: std::path::PathBuf,
    dict: Vec<String>,
    gpu: bool,
    threads: usize,
    det: Option<Model>,
    rec: Option<Model>,
    pdet: Option<Model>,
    prec: Option<Model>,
}

/// True when every model file and the ONNX Runtime library are present.
pub fn available(dir: &Path) -> bool {
    FILES.iter().all(|f| dir.join(f).is_file()) && dir.join(RUNTIME).is_file()
}

impl Engine {
    /// `gpu`: run on the GPU (DirectML, Windows) when it works, otherwise on the CPU with `threads` threads.
    pub fn new(dir: &Path, gpu: bool, threads: usize) -> Result<Engine, String> {
        if !available(dir) {
            return Err(format!("modelli OCR mancanti in {}", dir.display()));
        }
        let txt = std::fs::read_to_string(dir.join("text_dict.txt")).map_err(|e| e.to_string())?;
        let mut dict: Vec<String> = txt.split('\n').map(|l| l.trim_end_matches('\r').to_string()).collect();
        if dict.last().is_some_and(|l| l.is_empty()) {
            dict.pop();
        }
        let mut e = Engine { dir: dir.to_path_buf(), dict, gpu: false, threads, det: None, rec: None, pdet: None, prec: None };
        if gpu && cfg!(windows) {
            // the smallest model doubles as the check that the GPU provider really starts
            match Model::open(dir, "text_rec.onnx", true, threads) {
                Ok(m) => {
                    e.gpu = true;
                    e.rec = Some(m);
                }
                Err(err) => crate::log::log(format!("[ocr] GPU non disponibile ({err}): uso la CPU")),
            }
        }
        Ok(e)
    }

    /// Engine configured from the environment (`VFA_OCR_GPU=0` forces the CPU); for tests and tools.
    pub fn auto(dir: &Path) -> Result<Engine, String> {
        let gpu = std::env::var("VFA_OCR_GPU").map_or(true, |v| v != "0");
        Engine::new(dir, gpu, std::thread::available_parallelism().map_or(4, |n| n.get()) / 2)
    }

    pub fn gpu(&self) -> bool {
        self.gpu
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    pub fn device(&self) -> &'static str {
        if self.gpu {
            "GPU (DirectML)"
        } else {
            "CPU"
        }
    }

    fn model<'a>(slot: &'a mut Option<Model>, dir: &Path, file: &str, gpu: bool, threads: usize) -> Result<&'a mut Model, String> {
        if slot.is_none() {
            *slot = Some(Model::open(dir, file, gpu, threads)?);
        }
        Ok(slot.as_mut().unwrap())
    }

    // ---------------------------------------------------------------- text

    fn detect_text(&mut self, img: &Img) -> Result<Vec<Rect>, String> {
        let long = img.w.max(img.h) as f32;
        let ratio = long.clamp(320.0, 960.0) / long;
        let snap = |v: f32| (((v * ratio) / 32.0).round() as usize).max(1) * 32;
        let (dw, dh) = (snap(img.w as f32), snap(img.h as f32));
        let small = img.resize(dw, dh);
        const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
        const STD: [f32; 3] = [0.229, 0.224, 0.225];
        // PaddleOCR normalises a BGR image
        let mut data = vec![0f32; 3 * dw * dh];
        for i in 0..dw * dh {
            for c in 0..3 {
                data[c * dw * dh + i] = (small.rgb[i * 3 + (2 - c)] as f32 / 255.0 - MEAN[c]) / STD[c];
            }
        }
        let (_, prob) = Self::model(&mut self.det, &self.dir, "text_det.onnx", self.gpu, self.threads)?.run_f32(&[1, 3, dh, dw], &data)?;
        if prob.len() != dw * dh {
            return Err(format!("uscita del detector inattesa ({} valori per {}x{})", prob.len(), dw, dh));
        }
        let (sx, sy) = (img.w as f32 / dw as f32, img.h as f32 / dh as f32);
        let mut seen = vec![false; dw * dh];
        let mut boxes = Vec::new();
        let mut stack: Vec<u32> = Vec::new();
        for start in 0..dw * dh {
            if seen[start] || prob[start] <= 0.3 {
                continue;
            }
            seen[start] = true;
            stack.push(start as u32);
            let (mut x0, mut y0, mut x1, mut y1) = (dw, dh, 0usize, 0usize);
            let (mut sum, mut n) = (0f32, 0u32);
            while let Some(p) = stack.pop() {
                let (x, y) = (p as usize % dw, p as usize / dw);
                x0 = x0.min(x);
                x1 = x1.max(x);
                y0 = y0.min(y);
                y1 = y1.max(y);
                sum += prob[p as usize];
                n += 1;
                let mut push = |nx: usize, ny: usize| {
                    let q = ny * dw + nx;
                    if !seen[q] && prob[q] > 0.3 {
                        seen[q] = true;
                        stack.push(q as u32);
                    }
                };
                if x > 0 {
                    push(x - 1, y);
                }
                if x + 1 < dw {
                    push(x + 1, y);
                }
                if y > 0 {
                    push(x, y - 1);
                }
                if y + 1 < dh {
                    push(x, y + 1);
                }
            }
            let (bw, bh) = ((x1 - x0 + 1) as f32, (y1 - y0 + 1) as f32);
            if n < 8 || sum / (n as f32) < 0.55 || bw.min(bh) < 3.0 {
                continue;
            }
            // DB shrinks the text region: grow it back (unclip ratio 1.5)
            let d = bw * bh * 1.5 / (2.0 * (bw + bh));
            boxes.push(Rect {
                x0: ((x0 as f32 - d) * sx).max(0.0),
                y0: ((y0 as f32 - d) * sy).max(0.0),
                x1: (((x1 + 1) as f32 + d) * sx).min(img.w as f32),
                y1: (((y1 + 1) as f32 + d) * sy).min(img.h as f32),
            });
        }
        boxes.sort_by(|a, b| {
            let (ay, by) = (a.center().1, b.center().1);
            if (ay - by).abs() < a.h().min(b.h()) * 0.5 {
                a.x0.partial_cmp(&b.x0).unwrap()
            } else {
                ay.partial_cmp(&by).unwrap()
            }
        });
        Ok(boxes)
    }

    fn recognize_line(&mut self, crop: &Img) -> Result<(String, f32), String> {
        let crop = if crop.h as f32 > crop.w as f32 * 1.5 { crop.rotate_ccw() } else { crop.clone() };
        let want = ((REC_H as f32 * crop.w as f32 / crop.h as f32).ceil() as usize).clamp(8, *REC_WIDTHS.last().unwrap());
        let bucket = *REC_WIDTHS.iter().find(|&&b| b >= want).unwrap();
        let rw = want.min(bucket);
        let small = crop.resize(rw, REC_H);
        let mut data = vec![0f32; 3 * REC_H * bucket];
        for y in 0..REC_H {
            for x in 0..rw {
                for c in 0..3 {
                    data[c * REC_H * bucket + y * bucket + x] = (small.rgb[(y * rw + x) * 3 + (2 - c)] as f32 / 255.0 - 0.5) / 0.5;
                }
            }
        }
        let (sh, v) = Self::model(&mut self.rec, &self.dir, "text_rec.onnx", self.gpu, self.threads)?.run_f32(&[1, 3, REC_H, bucket], &data)?;
        let (t, c) = (sh[sh.len() - 2], sh[sh.len() - 1]);
        // the pad region produces only blanks, so decoding the whole sequence is harmless
        let (mut text, mut sum, mut n, mut last) = (String::new(), 0f32, 0u32, 0usize);
        for step in 0..t {
            let row = &v[step * c..(step + 1) * c];
            let (idx, &p) = row.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
            if idx != 0 && idx != last {
                text.push_str(self.dict.get(idx - 1).map(|s| s.as_str()).unwrap_or(" "));
                sum += p;
                n += 1;
            }
            last = idx;
        }
        Ok((text, if n == 0 { 0.0 } else { sum / n as f32 }))
    }

    /// Detects and reads every text line of the image (Latin script: en, it, fr, de, nl, pt, es, ...).
    pub fn read_text(&mut self, img: &Img) -> Result<Vec<Found>, String> {
        let mut found = Vec::new();
        for r in self.detect_text(img)? {
            let (text, conf) = self.recognize_line(&img.crop(r))?;
            let text = text.trim().to_string();
            if !text.is_empty() {
                found.push(Found { rect: r, text, conf, slots: Vec::new(), quality: 1.0 });
            }
        }
        Ok(found)
    }

    // -------------------------------------------------------------- plates

    /// Reads one plate crop in several variants (framing, rotation / shear for an oblique view, sharpening, contrast) and combines the per-character probabilities: far more tolerant of blur, of a loose or off-centre box and
    /// of an angled view than a single reading. Returns the probabilities (`PLATE_SLOTS × alphabet`) and the sharpness of the
    /// crop (0..1).
    fn read_plate_crop(&mut self, img: &Img, r: Rect) -> Result<(Vec<f32>, f32), String> {
        let vocab = PLATE_ALPHABET.len();
        let variants = &PLATE_VARIANTS;
        // a padded copy of the surroundings is made once; every variant samples from its grey plane
        let pad = Rect { x0: r.x0 - r.w() * 0.35, y0: r.y0 - r.h() * 0.35, x1: r.x1 + r.w() * 0.35, y1: r.y1 + r.h() * 0.35 };
        let region = img.crop(pad);
        let grey = grey_plane(&region);
        let (ox, oy) = (pad.x0.floor().max(0.0), pad.y0.floor().max(0.0));
        let local = Rect { x0: r.x0 - ox, y0: r.y0 - oy, x1: r.x1 - ox, y1: r.y1 - oy };
        let mut acc = vec![0f32; PLATE_SLOTS * vocab];
        let mut wsum = 0.0f32;
        let mut quality = 0.0;
        for (i, v) in variants.iter().enumerate() {
            let mut g = plate_view(&grey, region.w, region.h, local, v);
            if i == 0 {
                quality = sharpness(&g);
            }
            if v.sharpen {
                g = unsharp(&g);
            }
            if v.stretch {
                stretch_contrast(&mut g);
            }
            // the plate recogniser takes 8-bit grey levels
            let g8: Vec<u8> = g.iter().map(|x| (x + 0.5).clamp(0.0, 255.0) as u8).collect();
            let p = Self::model(&mut self.prec, &self.dir, "plate_rec.onnx", self.gpu, self.threads)?.run_u8(&[1, PLATE_H, PLATE_W, 1], &g8)?;
            if p.len() != PLATE_SLOTS * vocab {
                return Err(format!("uscita OCR targhe inattesa ({} valori)", p.len()));
            }
            // a variant that is sure of itself counts more than one that hesitates
            let m: f32 = p.chunks_exact(vocab).map(|s| s.iter().cloned().fold(0.0, f32::max)).sum::<f32>() / PLATE_SLOTS as f32;
            let w = m * m;
            wsum += w;
            for (a, x) in acc.iter_mut().zip(&p) {
                *a += w * x;
            }
        }
        acc.iter_mut().for_each(|a| *a /= wsum.max(1e-6));
        Ok((acc, quality))
    }

    /// Detects licence plates and reads them (upper-case letters and digits only).
    pub fn read_plates(&mut self, img: &Img) -> Result<Vec<Found>, String> {
        // letterbox into the detector's square input
        let s = PLATE_DET_SIZE as f32 / img.w.max(img.h) as f32;
        let (nw, nh) = (((img.w as f32 * s).round() as usize).max(1), ((img.h as f32 * s).round() as usize).max(1));
        let small = img.resize(nw, nh);
        let n = PLATE_DET_SIZE;
        let mut data = vec![114.0 / 255.0; 3 * n * n];
        for y in 0..nh {
            for x in 0..nw {
                for c in 0..3 {
                    data[c * n * n + y * n + x] = small.rgb[(y * nw + x) * 3 + c] as f32 / 255.0;
                }
            }
        }
        let (_, det) = Self::model(&mut self.pdet, &self.dir, "plate_det.onnx", self.gpu, self.threads)?.run_f32(&[1, 3, n, n], &data)?;
        let mut found = Vec::new();
        for row in det.chunks_exact(7) {
            // [batch, x1, y1, x2, y2, class, score] in letterboxed pixels
            if row[6] < 0.4 {
                continue;
            }
            let (bw, bh) = (row[3] - row[1], row[4] - row[2]);
            let (px, py) = (bw * 0.04, bh * 0.08);
            let r = Rect {
                x0: ((row[1] - px) / s).max(0.0),
                y0: ((row[2] - py) / s).max(0.0),
                x1: ((row[3] + px) / s).min(img.w as f32),
                y1: ((row[4] + py) / s).min(img.h as f32),
            };
            if r.w() < 4.0 || r.h() < 4.0 {
                continue;
            }
            let (slots, quality) = self.read_plate_crop(img, r)?;
            let vocab = PLATE_ALPHABET.len();
            let (mut text, mut sum, mut cnt) = (String::new(), 0f32, 0u32);
            for slot in slots.chunks_exact(vocab) {
                let (idx, &p) = slot.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
                if PLATE_ALPHABET[idx] != b'_' {
                    text.push(PLATE_ALPHABET[idx] as char);
                    sum += p;
                    cnt += 1;
                }
            }
            if cnt >= 3 {
                found.push(Found { rect: r, text, conf: (sum / cnt as f32) * row[6].min(1.0), slots, quality });
            }
        }
        Ok(found)
    }
}

/// One way of looking at a plate crop before it is read.
#[derive(Clone, Copy)]
struct PlateVariant {
    /// growth of the box (fraction of its width / height; negative = tighter)
    gx: f32,
    gy: f32,
    /// rotation in degrees and horizontal shear (fraction of the width over the full height)
    angle: f32,
    shear: f32,
    sharpen: bool,
    stretch: bool,
}

const fn pv(gx: f32, gy: f32, angle: f32, shear: f32, sharpen: bool, stretch: bool) -> PlateVariant {
    PlateVariant { gx, gy, angle, shear, sharpen, stretch }
}

/// The variants used for every plate: framing, sharpening and contrast, plus a small rotation and shear each way for plates
/// seen at an angle. (Undoing a horizontal motion blur by deconvolution was tried and did not help.)
const PLATE_VARIANTS: [PlateVariant; 9] = [
    pv(0.0, 0.0, 0.0, 0.0, false, false),
    pv(-0.06, -0.10, 0.0, 0.0, false, false),
    pv(0.12, 0.18, 0.0, 0.0, false, false),
    pv(0.0, 0.0, 0.0, 0.0, true, false),
    pv(0.0, 0.0, 0.0, 0.0, false, true),
    pv(0.0, 0.0, -7.0, 0.0, false, false),
    pv(0.0, 0.0, 7.0, 0.0, false, false),
    pv(0.0, 0.0, 0.0, -0.15, false, false),
    pv(0.0, 0.0, 0.0, 0.15, false, false),
];

fn grey_plane(img: &Img) -> Vec<f32> {
    img.rgb.chunks_exact(3).map(|p| 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32).collect()
}

/// Resamples the plate box `r` of a grey plane to the recogniser's input size, with the variant's framing, rotation and shear.
fn plate_view(grey: &[f32], w: usize, h: usize, r: Rect, v: &PlateVariant) -> Vec<f32> {
    let (cx, cy) = r.center();
    let (bw, bh) = (r.w() * (1.0 + v.gx), r.h() * (1.0 + v.gy));
    let (sin, cos) = v.angle.to_radians().sin_cos();
    // shrinking by more than 2x: average first so thin strokes survive
    let down = ((bw / PLATE_W as f32).max(bh / PLATE_H as f32)).max(1.0);
    let taps = if down >= 1.8 { 2 } else { 1 };
    let mut out = vec![0f32; PLATE_W * PLATE_H];
    for y in 0..PLATE_H {
        for x in 0..PLATE_W {
            let (mut sum, mut n) = (0f32, 0f32);
            for ty in 0..taps {
                for tx in 0..taps {
                    let nx = (x as f32 + (tx as f32 + 0.5) / taps as f32) / PLATE_W as f32 - 0.5;
                    let ny = (y as f32 + (ty as f32 + 0.5) / taps as f32) / PLATE_H as f32 - 0.5;
                    let dx = nx * bw + v.shear * ny * bw;
                    let dy = ny * bh;
                    let sx = (cx + dx * cos - dy * sin - 0.5).clamp(0.0, (w - 1) as f32);
                    let sy = (cy + dx * sin + dy * cos - 0.5).clamp(0.0, (h - 1) as f32);
                    let (x0, y0) = (sx.floor() as usize, sy.floor() as usize);
                    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
                    let (fx, fy) = (sx - x0 as f32, sy - y0 as f32);
                    let top = grey[y0 * w + x0] * (1.0 - fx) + grey[y0 * w + x1] * fx;
                    let bot = grey[y1 * w + x0] * (1.0 - fx) + grey[y1 * w + x1] * fx;
                    sum += top * (1.0 - fy) + bot * fy;
                    n += 1.0;
                }
            }
            out[y * PLATE_W + x] = sum / n;
        }
    }
    out
}

/// Sharpness of a plate crop in 0..1 (variance of the Laplacian, squashed): blurred crops score low.
fn sharpness(g: &[f32]) -> f32 {
    let (w, h) = (PLATE_W, PLATE_H);
    let (mut sum, mut sum2, mut n) = (0f64, 0f64, 0f64);
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let c = g[y * w + x];
            let l = 4.0 * c - g[y * w + x - 1] - g[y * w + x + 1] - g[(y - 1) * w + x] - g[(y + 1) * w + x];
            sum += l as f64;
            sum2 += (l * l) as f64;
            n += 1.0;
        }
    }
    let var = (sum2 / n - (sum / n).powi(2)) as f32;
    var / (var + 150.0)
}

fn unsharp(g: &[f32]) -> Vec<f32> {
    let (w, h) = (PLATE_W, PLATE_H);
    let mut out = g.to_vec();
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let mut blur = 0.0;
            for dy in 0..3 {
                for dx in 0..3 {
                    blur += g[(y + dy - 1) * w + x + dx - 1];
                }
            }
            out[y * w + x] = (g[y * w + x] + 1.2 * (g[y * w + x] - blur / 9.0)).clamp(0.0, 255.0);
        }
    }
    out
}

fn stretch_contrast(g: &mut [f32]) {
    let mut sorted = g.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (lo, hi) = (sorted[sorted.len() * 2 / 100], sorted[sorted.len() * 98 / 100]);
    if hi - lo > 8.0 {
        g.iter_mut().for_each(|v| *v = ((*v - lo) / (hi - lo) * 255.0).clamp(0.0, 255.0));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn norm(s: &str) -> String {
        s.chars().filter(|c| c.is_ascii_alphanumeric()).map(|c| c.to_ascii_uppercase()).collect()
    }

    /// OpenALPR-style benchmark: a folder of images, each with `<name>.txt` holding `file x y w h TEXT` lines.
    /// `VFA_OCR_DIR=<models> VFA_PLATE_IMAGES=<dir> cargo test --release plate_images -- --nocapture`
    #[test]
    fn plate_images() {
        let (Some(dir), Some(imgs)) = (std::env::var_os("VFA_OCR_DIR"), std::env::var_os("VFA_PLATE_IMAGES")) else { return };
        let mut e = Engine::auto(Path::new(&dir)).unwrap();
        let mut files: Vec<_> = std::fs::read_dir(&imgs).unwrap().flatten().map(|d| d.path()).filter(|p| p.extension().is_some_and(|x| x == "jpg")).collect();
        files.sort();
        let (mut truths, mut detected, mut exact, mut preds, mut false_pos, mut sim_sum) = (0, 0, 0, 0, 0, 0.0f32);
        for f in files {
            let truth: Vec<(Rect, String)> = std::fs::read_to_string(f.with_extension("txt"))
                .unwrap_or_default()
                .lines()
                .filter_map(|l| {
                    let c: Vec<&str> = l.split_whitespace().collect();
                    if c.len() < 6 {
                        return None;
                    }
                    let n: Vec<f32> = c[1..5].iter().filter_map(|v| v.parse().ok()).collect();
                    (n.len() == 4).then(|| (Rect { x0: n[0], y0: n[1], x1: n[0] + n[2], y1: n[1] + n[3] }, norm(&c[5..].join(""))))
                })
                .collect();
            let im = image::open(&f).unwrap().to_rgba8();
            let img = Img::from_rgba(im.width() as usize, im.height() as usize, im.as_raw());
            let found = e.read_plates(&img).unwrap();
            preds += found.len();
            let mut line = format!("{:<10}", f.file_stem().unwrap().to_string_lossy());
            for (r, t) in &truth {
                truths += 1;
                let m = found.iter().filter(|p| p.rect.iou(r) >= 0.3).max_by(|a, b| a.conf.partial_cmp(&b.conf).unwrap());
                match m {
                    Some(p) => {
                        detected += 1;
                        exact += (&p.text == t) as i32;
                        sim_sum += crate::ocr_vote::similarity(&p.text, t);
                        line.push_str(&format!("  {} -> {}{}", t, p.text, if &p.text == t { " OK" } else { "" }));
                    }
                    None => line.push_str(&format!("  {} -> (non trovata)", t)),
                }
            }
            false_pos += found.iter().filter(|p| !truth.iter().any(|(r, _)| p.rect.iou(r) >= 0.3)).count();
            println!("{line}");
        }
        println!("TOTAL plates {truths}  detected {detected}  exact {exact} ({:.1}%)  mean char-acc on detected {:.1}%  predictions {preds}  false positives {false_pos}", 100.0 * exact as f32 / truths.max(1) as f32, 100.0 * sim_sum / detected.max(1) as f32);
    }

    /// Manual probe: `VFA_OCR_DIR=<models> VFA_OCR_IMG=a.png;b.png cargo test --release ocr_probe -- --nocapture`
    #[test]
    fn ocr_probe() {
        let (Some(dir), Some(imgs)) = (std::env::var_os("VFA_OCR_DIR"), std::env::var("VFA_OCR_IMG").ok()) else { return };
        let mut e = Engine::auto(Path::new(&dir)).unwrap();
        println!("device: {}", e.device());
        for f in imgs.split(';') {
            let im = image::open(f).unwrap().to_rgba8();
            let img = Img::from_rgba(im.width() as usize, im.height() as usize, im.as_raw());
            let t = std::time::Instant::now();
            let txt = e.read_text(&img).unwrap();
            let t1 = t.elapsed();
            let pl = e.read_plates(&img).unwrap();
            println!("== {f} {}x{}  text {:?}  plates {:?}", img.w, img.h, t1, t.elapsed() - t1);
            for x in txt {
                println!("  T {:.2} [{:.0},{:.0},{:.0},{:.0}] {}", x.conf, x.rect.x0, x.rect.y0, x.rect.x1, x.rect.y1, x.text);
            }
            for x in pl {
                println!("  P {:.2} [{:.0},{:.0},{:.0},{:.0}] {}", x.conf, x.rect.x0, x.rect.y0, x.rect.x1, x.rect.y1, x.text);
            }
        }
    }
}
