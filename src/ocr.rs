//! OCR engine: scene-text detection + recognition (PaddleOCR PP-OCRv5 "latin": en, it, fr, de, nl, pt, es, ...)
//! and licence-plate detection + recognition, all running as ONNX models on the pure-Rust `tract` runtime.
use prost::Message;
use std::collections::HashMap;
use std::path::Path;
use tract_onnx::pb::{tensor_shape_proto::dimension::Value as DimV, type_proto::Value as TypeV, ModelProto};
use tract_onnx::prelude::*;

type Plan = TypedRunnableModel<TypedModel>;

/// Model files expected in the OCR directory.
pub const FILES: [&str; 5] = ["text_det.onnx", "text_rec.onnx", "text_dict.txt", "plate_det.onnx", "plate_rec.onnx"];
/// Side of the (square) input of the plate detector.
const PLATE_DET_SIZE: usize = 416;
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

fn fix_dims(vi: &mut tract_onnx::pb::ValueInfoProto) {
    if let Some(TypeV::TensorType(tt)) = vi.r#type.as_mut().and_then(|t| t.value.as_mut()) {
        if let Some(sh) = tt.shape.as_mut() {
            for (i, d) in sh.dim.iter_mut().enumerate() {
                if let Some(DimV::DimParam(p)) = d.value.as_mut() {
                    // the converted models name dynamic dimensions "DynamicDimension.0" or leave them empty, which tract cannot parse
                    let clean: String = p.chars().filter(|c| c.is_ascii_alphanumeric()).collect();
                    *p = format!("d{i}{clean}");
                }
            }
        }
    }
}

/// Loads an ONNX model with a fixed input shape. `patch` repairs the dynamic-dimension names of the PaddleOCR / plate-OCR exports.
fn load(path: &Path, shape: &[usize], patch: bool) -> Result<Plan, String> {
    let run = || -> TractResult<Plan> {
        let bytes = std::fs::read(path)?;
        let mut mp = ModelProto::decode(&bytes[..])?;
        if patch {
            if let Some(g) = mp.graph.as_mut() {
                g.value_info.clear();
                for v in g.input.iter_mut().chain(g.output.iter_mut()) {
                    fix_dims(v);
                }
            }
        }
        let m = tract_onnx::onnx().model_for_proto_model(&mp)?;
        m.with_input_fact(0, f32::fact(shape).into())?.into_optimized()?.into_runnable()
    };
    run().map_err(|e| format!("{}: {e:#}", path.display()))
}

fn run(plan: &Plan, shape: &[usize], data: &[f32]) -> Result<Tensor, String> {
    let t = Tensor::from_shape(shape, data).map_err(|e| format!("{e:#}"))?;
    let mut out = plan.run(tvec!(t.into())).map_err(|e| format!("{e:#}"))?;
    Ok(out.remove(0).into_tensor())
}

pub struct Engine {
    dir: std::path::PathBuf,
    dict: Vec<String>,
    det: HashMap<(usize, usize), Plan>,
    rec: HashMap<usize, Plan>,
    pdet: Option<Plan>,
    prec: Option<Plan>,
}

/// True when every model file is present.
pub fn available(dir: &Path) -> bool {
    FILES.iter().all(|f| dir.join(f).is_file())
}

impl Engine {
    pub fn new(dir: &Path) -> Result<Engine, String> {
        if !available(dir) {
            return Err(format!("modelli OCR mancanti in {}", dir.display()));
        }
        let txt = std::fs::read_to_string(dir.join("text_dict.txt")).map_err(|e| e.to_string())?;
        let mut dict: Vec<String> = txt.split('\n').map(|l| l.trim_end_matches('\r').to_string()).collect();
        if dict.last().is_some_and(|l| l.is_empty()) {
            dict.pop();
        }
        Ok(Engine { dir: dir.to_path_buf(), dict, det: HashMap::new(), rec: HashMap::new(), pdet: None, prec: None })
    }

    // ---------------------------------------------------------------- text

    fn detect_text(&mut self, img: &Img) -> Result<Vec<Rect>, String> {
        let long = img.w.max(img.h) as f32;
        let ratio = long.clamp(320.0, 960.0) / long;
        let snap = |v: f32| (((v * ratio) / 32.0).round() as usize).max(1) * 32;
        let (dw, dh) = (snap(img.w as f32), snap(img.h as f32));
        if !self.det.contains_key(&(dw, dh)) {
            if self.det.len() >= 6 {
                self.det.clear();
            }
            let plan = load(&self.dir.join("text_det.onnx"), &[1, 3, dh, dw], true)?;
            self.det.insert((dw, dh), plan);
        }
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
        let out = run(&self.det[&(dw, dh)], &[1, 3, dh, dw], &data)?;
        let prob = out.as_slice::<f32>().map_err(|e| format!("{e:#}"))?;
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
        if !self.rec.contains_key(&bucket) {
            let plan = load(&self.dir.join("text_rec.onnx"), &[1, 3, REC_H, bucket], true)?;
            self.rec.insert(bucket, plan);
        }
        let small = crop.resize(rw, REC_H);
        let mut data = vec![0f32; 3 * REC_H * bucket];
        for y in 0..REC_H {
            for x in 0..rw {
                for c in 0..3 {
                    data[c * REC_H * bucket + y * bucket + x] = (small.rgb[(y * rw + x) * 3 + (2 - c)] as f32 / 255.0 - 0.5) / 0.5;
                }
            }
        }
        let out = run(&self.rec[&bucket], &[1, 3, REC_H, bucket], &data)?;
        let sh = out.shape().to_vec();
        let (t, c) = (sh[sh.len() - 2], sh[sh.len() - 1]);
        let v = out.as_slice::<f32>().map_err(|e| format!("{e:#}"))?;
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
                found.push(Found { rect: r, text, conf });
            }
        }
        Ok(found)
    }

    // -------------------------------------------------------------- plates

    fn plate_plans(&mut self) -> Result<(), String> {
        if self.pdet.is_none() {
            self.pdet = Some(load(&self.dir.join("plate_det.onnx"), &[1, 3, PLATE_DET_SIZE, PLATE_DET_SIZE], false)?);
        }
        if self.prec.is_none() {
            self.prec = Some(load(&self.dir.join("plate_rec.onnx"), &[1, PLATE_H, PLATE_W, 1], true)?);
        }
        Ok(())
    }

    /// Detects licence plates and reads them (upper-case letters and digits only).
    pub fn read_plates(&mut self, img: &Img) -> Result<Vec<Found>, String> {
        self.plate_plans()?;
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
        let out = run(self.pdet.as_ref().unwrap(), &[1, 3, n, n], &data)?;
        let det = out.as_slice::<f32>().map_err(|e| format!("{e:#}"))?;
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
            let plate = img.crop(r).resize(PLATE_W, PLATE_H);
            let gray: Vec<f32> = plate.rgb.chunks_exact(3).map(|p| 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32).collect();
            let o = run(self.prec.as_ref().unwrap(), &[1, PLATE_H, PLATE_W, 1], &gray)?;
            let v = o.as_slice::<f32>().map_err(|e| format!("{e:#}"))?;
            let vocab = PLATE_ALPHABET.len();
            if v.len() != PLATE_SLOTS * vocab {
                return Err(format!("uscita OCR targhe inattesa ({} valori)", v.len()));
            }
            let (mut text, mut sum, mut cnt) = (String::new(), 0f32, 0u32);
            for slot in v.chunks_exact(vocab) {
                let (idx, &p) = slot.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
                if PLATE_ALPHABET[idx] != b'_' {
                    text.push(PLATE_ALPHABET[idx] as char);
                    sum += p;
                    cnt += 1;
                }
            }
            if cnt >= 3 {
                found.push(Found { rect: r, text, conf: (sum / cnt as f32) * row[6].min(1.0) });
            }
        }
        Ok(found)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Manual probe: `VFA_OCR_DIR=<models> VFA_OCR_IMG=a.png;b.png cargo test --release ocr_probe -- --nocapture`
    #[test]
    fn ocr_probe() {
        let (Some(dir), Some(imgs)) = (std::env::var_os("VFA_OCR_DIR"), std::env::var("VFA_OCR_IMG").ok()) else { return };
        let mut e = Engine::new(Path::new(&dir)).unwrap();
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
