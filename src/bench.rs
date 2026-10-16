//! Interactive benchmark of the neural engines: models x resolutions x processors.
use crate::filters::AiSettings;
use crate::log::log;
use crate::nn;
use crate::tools::Tools;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Clone)]
pub struct Spec {
    pub label: String,
    pub ai: AiSettings,
    /// processors to test for this spec (-1 = CPU, otherwise GPU index)
    pub devices: Vec<i32>,
}

#[derive(Clone)]
pub struct Plan {
    pub specs: Vec<Spec>,
    /// frame heights in pixels (360, 720, 1080 ...)
    pub heights: Vec<u32>,
    pub timeout_secs: u64,
}

impl Plan {
    pub fn total(&self) -> usize {
        self.specs.iter().map(|s| s.devices.len()).sum::<usize>() * self.heights.len()
    }
}

#[derive(Clone, Debug)]
pub struct Cell {
    pub secs: Option<f64>,
    pub note: String,
}

pub enum Msg {
    Cell { spec: usize, res: usize, dev: i32, cell: Cell },
    Finished,
}

fn synthetic(w: u32, h: u32) -> image::RgbaImage {
    image::RgbaImage::from_fn(w, h, |x, y| {
        let n = (x.wrapping_mul(2654435761) ^ y.wrapping_mul(40503)) >> 7 & 15;
        image::Rgba([((x * 255 / w) as u8).wrapping_add(n as u8), ((y * 255 / h) as u8).wrapping_add(n as u8), (((x + y) * 255 / (w + h)) as u8) ^ n as u8, 255])
    })
}

/// One timed execution; `Err(note)` on failure / timeout.
fn run_once(tools: &Tools, input: &std::path::Path, ai: &AiSettings, timeout: u64, global_cancel: &Arc<AtomicBool>) -> Result<f64, String> {
    let out = nn::temp_path(tools, "png");
    let run_cancel = Arc::new(AtomicBool::new(false));
    let timed_out = Arc::new(AtomicBool::new(false));
    let done = Arc::new(AtomicBool::new(false));
    let (rc, to, dn) = (run_cancel.clone(), timed_out.clone(), done.clone());
    let started = Instant::now();
    let g = global_cancel.clone();
    let watcher = std::thread::spawn(move || {
        while !dn.load(Ordering::Relaxed) {
            if g.load(Ordering::Relaxed) {
                rc.store(true, Ordering::Relaxed);
                break;
            }
            if started.elapsed() > Duration::from_secs(timeout) {
                to.store(true, Ordering::Relaxed);
                rc.store(true, Ordering::Relaxed);
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    });
    let r = nn::run_file(tools, input, &out, ai, &run_cancel);
    let secs = started.elapsed().as_secs_f64();
    done.store(true, Ordering::Relaxed);
    let _ = watcher.join();
    let _ = std::fs::remove_file(&out);
    match r {
        Ok(_) => Ok(secs),
        Err(_) if timed_out.load(Ordering::Relaxed) => Err(format!("> {timeout} s")),
        Err(e) if e == "annullato" => Err("annullato".into()),
        Err(e) => Err(format!("errore: {}", e.lines().next().unwrap_or(""))),
    }
}

/// Runs the whole plan (blocking; call from a thread).
pub fn run(tools: &Tools, plan: &Plan, source: Option<(u32, u32, Vec<u8>)>, cancel: &Arc<AtomicBool>, tx: &Sender<Msg>, wake: &dyn Fn()) {
    let base = match source.and_then(|(w, h, px)| image::RgbaImage::from_raw(w, h, px)) {
        Some(img) => img,
        None => synthetic(1920, 1080),
    };
    let aspect = base.width() as f64 / base.height() as f64;
    // one input file per resolution
    let mut inputs = Vec::new();
    for &h in &plan.heights {
        let w = ((h as f64 * aspect / 2.0).round() as u32 * 2).max(2);
        let img = image::imageops::resize(&base, w, h, image::imageops::FilterType::Lanczos3);
        let path = nn::temp_path(tools, "png");
        if img.save(&path).is_err() {
            log("[bench] impossibile creare l'immagine di prova");
            let _ = tx.send(Msg::Finished);
            return;
        }
        inputs.push(path);
    }
    'outer: for (si, spec) in plan.specs.iter().enumerate() {
        for &dev in &spec.devices {
            let mut ai = spec.ai.clone();
            ai.set_device(dev);
            let mut warmed = dev < 0; // CPU needs no warm-up
            let mut gave_up = false;
            for (ri, input) in inputs.iter().enumerate() {
                if cancel.load(Ordering::Relaxed) {
                    break 'outer;
                }
                let cell = if gave_up {
                    Cell { secs: None, note: "saltato".into() }
                } else {
                    let mut r = run_once(tools, input, &ai, plan.timeout_secs, cancel);
                    // GPU tools fill shader / driver caches on first use: repeat once and report the warm run
                    if !warmed {
                        if let Ok(t) = r {
                            if t < 40.0 && !cancel.load(Ordering::Relaxed) {
                                r = run_once(tools, input, &ai, plan.timeout_secs, cancel);
                            }
                        }
                        warmed = true;
                    }
                    match r {
                        Ok(t) => Cell { secs: Some(t), note: String::new() },
                        Err(n) if n == "annullato" => break 'outer,
                        Err(n) => {
                            gave_up = n.starts_with('>');
                            Cell { secs: None, note: n }
                        }
                    }
                };
                log(format!("[bench] {} {}p dev {}: {:?}", spec.label, plan.heights[ri], dev, cell));
                let _ = tx.send(Msg::Cell { spec: si, res: ri, dev, cell });
                wake();
            }
        }
    }
    for p in inputs {
        let _ = std::fs::remove_file(p);
    }
    let _ = tx.send(Msg::Finished);
    wake();
}
