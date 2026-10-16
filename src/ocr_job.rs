//! Background OCR job: decodes frames (with the preview's filters, optionally neural-enhanced), reads text and/or plates,
//! then groups the readings of the whole series into one result per thing seen.
use crate::decoder::{Stream, Waker};
use crate::filters::AiSettings;
use crate::nn;
use crate::ocr::{Engine, Found, Img, Rect};
use crate::ocr_vote::{aggregate, Hit, Kind, Reading};
use crate::tools::Tools;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

pub type SharedEngine = Arc<Mutex<Option<Engine>>>;

#[derive(Clone)]
pub struct Spec {
    pub video: PathBuf,
    pub fps: f64,
    pub vf: String,
    pub ai: Option<AiSettings>,
    pub start: u64,
    pub count: u64,
    /// analyse one frame every `step`
    pub step: u64,
    /// region of interest as fractions of the frame: x0, y0, x1, y1
    pub roi: Option<[f32; 4]>,
    pub text: bool,
    pub plates: bool,
}

/// Fraction of `inner` covered by `outer`.
fn covered(inner: &Rect, outer: &Rect) -> f32 {
    let iw = (inner.x1.min(outer.x1) - inner.x0.max(outer.x0)).max(0.0);
    let ih = (inner.y1.min(outer.y1) - inner.y0.max(outer.y0)).max(0.0);
    let a = inner.w() * inner.h();
    if a <= 0.0 {
        0.0
    } else {
        iw * ih / a
    }
}

/// Reads one frame; rectangles come back normalised to the whole frame (0..1).
fn read_frame(engine: &mut Engine, img: &Img, roi: Option<[f32; 4]>, text: bool, plates: bool) -> Result<Vec<(Found, Kind)>, String> {
    let (w, h) = (img.w as f32, img.h as f32);
    let (sub, ox, oy) = match roi {
        Some(r) => {
            let (x0, y0) = ((r[0] * w).floor(), (r[1] * h).floor());
            (img.crop(Rect { x0, y0, x1: (r[2] * w).ceil(), y1: (r[3] * h).ceil() }), x0, y0)
        }
        None => (img.clone(), 0.0, 0.0),
    };
    let norm = |f: Found| Found { rect: Rect { x0: (f.rect.x0 + ox) / w, y0: (f.rect.y0 + oy) / h, x1: (f.rect.x1 + ox) / w, y1: (f.rect.y1 + oy) / h }, ..f };
    let mut out: Vec<(Found, Kind)> = Vec::new();
    if plates {
        for f in engine.read_plates(&sub)? {
            if f.conf >= 0.3 {
                out.push((norm(f), Kind::Plate));
            }
        }
    }
    if text {
        let plate_rects: Vec<Rect> = out.iter().map(|(f, _)| f.rect).collect();
        for f in engine.read_text(&sub)? {
            let f = norm(f);
            // a plate is also seen by the text reader: keep only the plate reading
            if f.conf >= 0.3 && !plate_rects.iter().any(|p| covered(&f.rect, p) > 0.6) {
                out.push((f, Kind::Text));
            }
        }
    }
    Ok(out)
}

/// Runs the job. `progress(done, total)` counts frames analysed; setting `cancel` ends it early with the results so far.
pub fn run(tools: &Tools, engine: &SharedEngine, spec: &Spec, cancel: &AtomicBool, progress: &dyn Fn(u64, u64)) -> Result<Vec<Hit>, String> {
    let mut guard = engine.lock().unwrap();
    if guard.is_none() {
        let dir = tools.ocr_dir.as_deref().ok_or("modelli OCR non inclusi in questa build")?;
        *guard = Some(Engine::new(dir)?);
    }
    let eng = guard.as_mut().unwrap();
    let wake: Waker = Arc::new(|| {});
    let mut stream = Stream::open(tools, &spec.video, spec.start, spec.fps, &spec.vf, Some(spec.count), 4, wake);
    let step = spec.step.max(1);
    let total = spec.count.div_ceil(step);
    let (mut readings, mut done) = (Vec::new(), 0u64);
    progress(0, total);
    loop {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let Some(frame) = stream.try_next() else {
            if let Some(end) = &stream.ended {
                if done == 0 {
                    return Err(end.clone().unwrap_or_else(|| "nessun fotogramma decodificato".into()));
                }
                break;
            }
            std::thread::sleep(Duration::from_millis(3));
            continue;
        };
        if (frame.index - spec.start) % step != 0 {
            continue;
        }
        let enhanced;
        let frame = match &spec.ai {
            Some(ai) => {
                enhanced = nn::enhance_frame(tools, &frame, ai, cancel)?;
                &enhanced
            }
            None => &*frame,
        };
        let img = Img::from_rgba(frame.w, frame.h, &frame.rgba);
        let t = std::time::Instant::now();
        for (f, kind) in read_frame(eng, &img, spec.roi, spec.text, spec.plates)? {
            readings.push(Reading { frame: frame.index, rect: f.rect, text: f.text, conf: f.conf, kind });
        }
        crate::log::log(format!("[ocr] fotogramma {} in {:.2}s", frame.index, t.elapsed().as_secs_f32()));
        done += 1;
        progress(done, total);
    }
    Ok(aggregate(readings, step * 3 + 2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ocr_vote::Kind;

    /// Manual probe: `VFA_FFMPEG=ffmpeg VFA_OCR_DIR=<models> VFA_OCR_VIDEO=clip.mp4 cargo test --release job_probe -- --nocapture`
    #[test]
    fn job_probe() {
        let Some(video) = std::env::var_os("VFA_OCR_VIDEO") else { return };
        if std::env::var_os("VFA_OCR_DIR").is_none() {
            return;
        }
        let tools = crate::tools::prepare().unwrap();
        let spec = Spec { video: video.into(), fps: 15.0, vf: String::new(), ai: None, start: 0, count: 12, step: 2, roi: None, text: true, plates: true };
        let engine: SharedEngine = Arc::new(Mutex::new(None));
        let hits = run(&tools, &engine, &spec, &AtomicBool::new(false), &|d, t| println!("progress {d}/{t}")).unwrap();
        for h in &hits {
            println!(
                "{:?} {:?} conf {:.2} frames {}-{} best {} n={} fmt={:?} alts={:?} rect=[{:.2},{:.2},{:.2},{:.2}]",
                h.kind, h.text, h.conf, h.first, h.last, h.best, h.readings, h.format, h.alts, h.rect.x0, h.rect.y0, h.rect.x1, h.rect.y1
            );
        }
        assert!(hits.iter().any(|h| h.kind == Kind::Plate));
    }
}
