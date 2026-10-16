//! Background OCR job: decodes frames (with the preview's filters, optionally neural-enhanced), reads text and/or plates,
//! then groups the readings of the whole series into one result per thing seen.
use crate::decoder::{Frame, Stream, Waker};
use crate::filters::AiSettings;
use crate::nn;
use crate::ocr::{Engine, Found, Img, Rect};
use crate::ocr_vote::{aggregate, Hit, Kind, Reading};
use crate::tools::Tools;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

pub type SharedEngine = Arc<Mutex<Vec<Engine>>>;

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
    /// run the models on the GPU when possible (DirectML, Windows)
    pub gpu: bool,
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

/// Frames analysed at the same time; each worker owns a copy of the models.
fn worker_count(gpu: bool) -> usize {
    std::env::var("VFA_OCR_WORKERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| if gpu { 2 } else { (std::thread::available_parallelism().map_or(2, |n| n.get()) / 4).clamp(1, 4) })
        .max(1)
}

/// Neural enhancement (optional) and reading of one frame.
fn process_frame(tools: &Tools, eng: &mut Engine, frame: &Frame, spec: &Spec, cancel: &AtomicBool) -> Result<Vec<(Found, Kind)>, String> {
    let enhanced;
    let frame = match &spec.ai {
        Some(ai) => {
            enhanced = nn::enhance_frame(tools, frame, ai, cancel)?;
            &enhanced
        }
        None => frame,
    };
    let img = Img::from_rgba(frame.w, frame.h, &frame.rgba);
    let t = std::time::Instant::now();
    let found = read_frame(eng, &img, spec.roi, spec.text, spec.plates)?;
    crate::log::log(format!("[ocr] fotogramma {} in {:.2}s", frame.index, t.elapsed().as_secs_f32()));
    Ok(found)
}

/// Runs the job. `progress(done, total)` counts frames analysed; setting `cancel` ends it early with the results so far.
pub fn run(tools: &Tools, engine: &SharedEngine, spec: &Spec, cancel: &AtomicBool, progress: &dyn Fn(u64, u64)) -> Result<Vec<Hit>, String> {
    let dir = tools.ocr_dir.as_deref().ok_or("modelli OCR non inclusi in questa build")?;
    let workers = worker_count(spec.gpu);
    let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
    let threads = if spec.gpu { 2 } else { (cores / workers).max(2) };
    let mut engines = std::mem::take(&mut *engine.lock().unwrap());
    // engines made for another device / thread count are rebuilt
    engines.retain(|e| e.threads() == threads && (e.gpu() || !spec.gpu || !cfg!(windows)));
    let mut made = Ok(());
    while engines.len() < workers {
        match Engine::new(dir, spec.gpu, threads) {
            Ok(e) => engines.push(e),
            Err(e) => {
                made = Err(e);
                break;
            }
        }
    }
    if let Some(e) = engines.first() {
        crate::log::log(format!("[ocr] dispositivo: {}, {} lavoratori", e.device(), workers));
    }
    let result = match made {
        Ok(()) => run_with(tools, &mut engines[..workers], spec, cancel, progress),
        Err(e) => Err(e),
    };
    *engine.lock().unwrap() = engines;
    result
}

fn run_with(tools: &Tools, engines: &mut [Engine], spec: &Spec, cancel: &AtomicBool, progress: &dyn Fn(u64, u64)) -> Result<Vec<Hit>, String> {
    let wake: Waker = Arc::new(|| {});
    let mut stream = Stream::open(tools, &spec.video, spec.start, spec.fps, &spec.vf, Some(spec.count), 4, wake);
    let step = spec.step.max(1);
    let total = spec.count.div_ceil(step);
    let (ftx, frx) = std::sync::mpsc::sync_channel::<Arc<Frame>>(engines.len() * 2);
    let frx = Mutex::new(frx);
    let (rtx, rrx) = std::sync::mpsc::channel::<Result<(u64, Vec<(Found, Kind)>), String>>();
    let stop = AtomicBool::new(false);
    let (mut readings, mut done, mut decoded, mut error, mut decode_err) = (Vec::new(), 0u64, 0u64, None::<String>, None::<String>);
    progress(0, total);
    std::thread::scope(|sc| {
        for eng in engines.iter_mut() {
            let (frx, rtx, stop) = (&frx, rtx.clone(), &stop);
            sc.spawn(move || loop {
                let Ok(fr) = frx.lock().unwrap().recv() else { break };
                if stop.load(Ordering::Relaxed) || cancel.load(Ordering::Relaxed) {
                    continue; // keep draining so the producer never blocks
                }
                let res = process_frame(tools, eng, &fr, spec, cancel).map(|v| (fr.index, v));
                if res.is_err() {
                    stop.store(true, Ordering::Relaxed);
                }
                let _ = rtx.send(res);
            });
        }
        drop(rtx);
        let mut handle = |r: Result<(u64, Vec<(Found, Kind)>), String>| match r {
            Ok((frame, found)) => {
                for (f, kind) in found {
                    readings.push(Reading { frame, rect: f.rect, text: f.text, conf: f.conf, kind, slots: f.slots, quality: f.quality });
                }
                done += 1;
                progress(done, total);
            }
            Err(e) => {
                error.get_or_insert(e);
            }
        };
        loop {
            while let Ok(r) = rrx.try_recv() {
                handle(r);
            }
            if cancel.load(Ordering::Relaxed) || stop.load(Ordering::Relaxed) {
                break;
            }
            match stream.try_next() {
                Some(frame) => {
                    decoded += 1;
                    if (frame.index - spec.start) % step == 0 && ftx.send(frame).is_err() {
                        break;
                    }
                }
                None => {
                    if let Some(end) = &stream.ended {
                        if decoded == 0 {
                            decode_err = Some(end.clone().unwrap_or_else(|| "nessun fotogramma decodificato".into()));
                        }
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(3));
                }
            }
        }
        drop(ftx);
        for r in rrx.iter() {
            handle(r);
        }
    });
    match error.or(decode_err) {
        Some(e) => Err(e),
        None => Ok(aggregate(readings, step * 3 + 2)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ocr_vote::Kind;

    /// Real-video benchmark: clips named `<id>_<TRUTH>.mp4` (the FANVID clips) in `VFA_FANVID_DIR`; a clip succeeds when a
    /// plate track voted to exactly the truth text. `VFA_FANVID_LIMIT`, `VFA_FANVID_STEP` (default 2) tune the run.
    #[test]
    fn fanvid_bench() {
        let Some(dir) = std::env::var_os("VFA_FANVID_DIR") else { return };
        if std::env::var_os("VFA_OCR_DIR").is_none() {
            return;
        }
        let tools = crate::tools::prepare().unwrap();
        let limit: usize = std::env::var("VFA_FANVID_LIMIT").ok().and_then(|v| v.parse().ok()).unwrap_or(usize::MAX);
        let step: u64 = std::env::var("VFA_FANVID_STEP").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
        let mut files: Vec<_> = std::fs::read_dir(dir).unwrap().flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "mp4")).collect();
        files.sort_by_key(|p| p.file_stem().unwrap().to_string_lossy().split('_').next().and_then(|n| n.parse::<u32>().ok()).unwrap_or(0));
        let engine: SharedEngine = Arc::new(Mutex::new(Vec::new()));
        let (mut ok, mut n, mut sim_sum, mut found_any) = (0, 0, 0.0f32, 0);
        let t0 = std::time::Instant::now();
        for f in files.into_iter().take(limit) {
            let stem = f.file_stem().unwrap().to_string_lossy().to_string();
            let truth: String = stem.split_once('_').unwrap().1.chars().filter(|c| c.is_ascii_alphanumeric()).collect::<String>().to_uppercase();
            let spec = Spec { video: f.clone(), fps: 30.0, vf: String::new(), ai: None, start: 0, count: 100_000, step, roi: None, text: false, plates: true, gpu: std::env::var("VFA_OCR_GPU").map_or(true, |v| v != "0") };
            let t = std::time::Instant::now();
            let hits = run(&tools, &engine, &spec, &AtomicBool::new(false), &|_, _| {}).unwrap();
            let plates: Vec<&Hit> = hits.iter().filter(|h| h.kind == Kind::Plate && h.readings >= 2).collect();
            let best = plates.iter().max_by(|a, b| crate::ocr_vote::similarity(&a.text, &truth).partial_cmp(&crate::ocr_vote::similarity(&b.text, &truth)).unwrap());
            let sim = best.map_or(0.0, |h| crate::ocr_vote::similarity(&h.text, &truth));
            let hit = plates.iter().any(|h| h.text == truth);
            n += 1;
            ok += hit as u32;
            sim_sum += sim;
            found_any += !plates.is_empty() as u32;
            println!(
                "{:<26} {}  best {:<10} sim {:>3.0}%  tracks {:<2} {:>5.1}s",
                stem,
                if hit { "OK " } else { "-- " },
                best.map_or("", |h| h.text.as_str()),
                100.0 * sim,
                plates.len(),
                t.elapsed().as_secs_f32()
            );
        }
        println!("TOTAL clips {n}  exact {ok} ({:.1}%)  any plate {found_any}  mean best similarity {:.1}%  total time {:.0}s", 100.0 * ok as f32 / n.max(1) as f32, 100.0 * sim_sum / n.max(1) as f32, t0.elapsed().as_secs_f32());
    }

    /// Plate benchmark on synthetic "fast vehicle" clips named `<TRUTH>_<level>.mp4` in `VFA_BENCH_DIR`:
    /// `VFA_FFMPEG=<ffmpeg> VFA_OCR_DIR=<models> VFA_BENCH_DIR=<dir> cargo test --release plate_bench -- --nocapture`
    #[test]
    fn plate_bench() {
        let Some(dir) = std::env::var_os("VFA_BENCH_DIR") else { return };
        if std::env::var_os("VFA_OCR_DIR").is_none() {
            return;
        }
        let tools = crate::tools::prepare().unwrap();
        let mut eng = Engine::auto(tools.ocr_dir.as_deref().unwrap()).unwrap();
        let mut files: Vec<_> = std::fs::read_dir(dir).unwrap().flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "mp4")).collect();
        files.sort();
        let wake: Waker = Arc::new(|| {});
        let (mut tot_exact, mut tot_frames, mut tot_voted, mut tot_clips) = (0, 0, 0, 0);
        for f in files {
            let stem = f.file_stem().unwrap().to_string_lossy().to_string();
            let (truth, _level) = stem.split_once('_').unwrap();
            let (frames, _) = Stream::open(&tools, &f, 0, 25.0, "", Some(50), 4, wake.clone()).collect_all();
            let t = std::time::Instant::now();
            let (mut readings, mut exact, mut detected, mut chars) = (Vec::new(), 0, 0, 0.0f32);
            for fr in &frames {
                let img = Img::from_rgba(fr.w, fr.h, &fr.rgba);
                let found = eng.read_plates(&img).unwrap();
                if let Some(best) = found.iter().max_by(|a, b| a.conf.partial_cmp(&b.conf).unwrap()) {
                    detected += 1;
                    exact += (best.text == truth) as u32;
                    chars += crate::ocr_vote::similarity(&best.text, truth);
                }
                for x in found {
                    readings.push(Reading { frame: fr.index, rect: Rect { x0: x.rect.x0 / fr.w as f32, y0: x.rect.y0 / fr.h as f32, x1: x.rect.x1 / fr.w as f32, y1: x.rect.y1 / fr.h as f32 }, text: x.text, conf: x.conf, kind: Kind::Plate, slots: x.slots, quality: x.quality });
                }
            }
            let hits = aggregate(readings, 5);
            let main = hits.iter().max_by_key(|h| h.readings);
            let voted = main.map(|h| h.text.clone()).unwrap_or_default();
            println!(
                "{:<22} frames {:>2}  detected {:>2}  exact {:>2}  char-acc {:>5.1}%  tracks {}  VOTED {:<10} {}   {:.2}s/frame",
                stem, frames.len(), detected, exact, 100.0 * chars / detected.max(1) as f32, hits.len(), voted, if voted == truth { "OK" } else { "--" }, t.elapsed().as_secs_f32() / frames.len().max(1) as f32
            );
            tot_exact += exact;
            tot_frames += frames.len();
            tot_voted += (voted == truth) as u32;
            tot_clips += 1;
        }
        println!("TOTAL per-frame exact {tot_exact}/{tot_frames}   voted correct {tot_voted}/{tot_clips}");
    }

    /// Manual probe: `VFA_FFMPEG=ffmpeg VFA_OCR_DIR=<models> VFA_OCR_VIDEO=clip.mp4 cargo test --release job_probe -- --nocapture`
    #[test]
    fn job_probe() {
        let Some(video) = std::env::var_os("VFA_OCR_VIDEO") else { return };
        if std::env::var_os("VFA_OCR_DIR").is_none() {
            return;
        }
        let tools = crate::tools::prepare().unwrap();
        let spec = Spec { video: video.into(), fps: 15.0, vf: String::new(), ai: None, start: 0, count: 12, step: 2, roi: None, text: true, plates: true, gpu: true };
        let engine: SharedEngine = Arc::new(Mutex::new(Vec::new()));
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
