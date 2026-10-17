//! Full-quality frame export (single frame or a consecutive range).
use crate::i18n::{tf, tr};
use crate::decoder::seek_args;
use crate::filters::AiSettings;
use crate::nn;
use crate::tools::{command, Tools};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, PartialEq)]
pub enum Format {
    Png,
    Tiff,
    Jpeg,
    Webp,
    Bmp,
}

impl Format {
    pub const ALL: [Format; 5] = [Format::Png, Format::Tiff, Format::Jpeg, Format::Webp, Format::Bmp];
    pub fn label(self) -> &'static str {
        match self {
            Format::Png => tr("PNG (lossless)"),
            Format::Tiff => tr("TIFF (lossless)"),
            Format::Jpeg => tr("JPEG (qualità massima)"),
            Format::Webp => tr("WebP (lossless)"),
            Format::Bmp => tr("BMP (non compresso)"),
        }
    }
    pub fn ext(self) -> &'static str {
        match self {
            Format::Png => "png",
            Format::Tiff => "tif",
            Format::Jpeg => "jpg",
            Format::Webp => "webp",
            Format::Bmp => "bmp",
        }
    }
    /// (pixel format, extra encoder args)
    fn encoder(self, high_bit: bool) -> (&'static str, Vec<&'static str>) {
        match self {
            Format::Png => (if high_bit { "rgb48be" } else { "rgb24" }, vec!["-compression_level", "9"]),
            Format::Tiff => (if high_bit { "rgb48le" } else { "rgb24" }, vec!["-compression_algo", "deflate"]),
            Format::Jpeg => ("yuvj444p", vec!["-q:v", "1", "-qmin", "1"]),
            Format::Webp => ("bgra", vec!["-c:v", "libwebp", "-lossless", "1", "-quality", "100", "-compression_level", "6"]),
            Format::Bmp => ("bgr24", vec![]),
        }
    }
}

#[derive(Clone)]
pub struct Job {
    pub video: PathBuf,
    pub out_dir: PathBuf,
    pub start: u64,
    pub count: u64,
    pub fps: f64,
    pub vf: String,
    pub format: Format,
    pub high_bit: bool,
    pub ai: Option<AiSettings>,
}

fn name(video: &Path, idx: u64, ext: &str) -> String {
    let stem = video.file_stem().and_then(|s| s.to_str()).unwrap_or("video");
    format!("{stem}_frame_{idx:06}.{ext}")
}

fn ffmpeg_err(out: &std::process::Output) -> String {
    let e = String::from_utf8_lossy(&out.stderr);
    let mut v: Vec<&str> = e.lines().rev().filter(|l| !l.trim().is_empty()).take(5).collect();
    v.reverse();
    v.join("\n")
}

/// Extracts `count` frames starting at `start`, written as `pattern` (printf style) with `-start_number`.
/// `on_frame` receives the number of frames written so far; setting `cancel` kills ffmpeg.
fn extract(tools: &Tools, job: &Job, fmt: Format, pattern: &Path, cancel: &AtomicBool, on_frame: &dyn Fn(u64)) -> Result<(), String> {
    let (pix, extra) = fmt.encoder(job.high_bit);
    let mut vf = job.vf.clone();
    if !vf.is_empty() {
        vf.push(',');
    }
    vf.push_str(&format!("scale=flags=accurate_rnd+full_chroma_int+lanczos,format={pix}"));
    let mut cmd = command(&tools.ffmpeg);
    cmd.args(["-y", "-hide_banner", "-loglevel", "error", "-progress", "pipe:1", "-nostats", "-stats_period", "0.25"]);
    cmd.args(seek_args(job.start, job.fps));
    cmd.arg("-i").arg(&job.video);
    cmd.args(["-an", "-sn", "-dn", "-fps_mode", "passthrough", "-vf", &vf]);
    cmd.args(["-frames:v", &job.count.to_string(), "-start_number", &job.start.to_string()]);
    cmd.args(extra);
    cmd.args(["-f", "image2"]);
    cmd.arg(pattern);
    cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    crate::log::log(format!("[export] {:?}", cmd));
    let mut child = cmd.spawn().map_err(|e| format!("ffmpeg: {e}"))?;
    let out = child.stdout.take().unwrap();
    let err = child.stderr.take().unwrap();
    let child = Mutex::new(Some(child));
    let done = AtomicBool::new(false);
    let errtext = Mutex::new(String::new());
    std::thread::scope(|sc| {
        sc.spawn(|| {
            while !done.load(Ordering::Relaxed) {
                if cancel.load(Ordering::Relaxed) {
                    if let Some(c) = child.lock().unwrap().as_mut() {
                        let _ = c.kill();
                    }
                    break;
                }
                std::thread::sleep(Duration::from_millis(40));
            }
        });
        sc.spawn(|| {
            for l in BufReader::new(err).lines().map_while(Result::ok) {
                let mut t = errtext.lock().unwrap();
                if t.len() < 6000 {
                    t.push_str(&l);
                    t.push('\n');
                }
            }
        });
        for l in BufReader::new(out).lines().map_while(Result::ok) {
            if let Some(n) = l.strip_prefix("frame=") {
                if let Ok(n) = n.trim().parse::<u64>() {
                    on_frame(n);
                }
            }
        }
        done.store(true, Ordering::Relaxed);
    });
    let status = child.lock().unwrap().take().unwrap().wait().map_err(|e| e.to_string())?;
    if cancel.load(Ordering::Relaxed) {
        return Err(tr("Annullato").into());
    }
    if !status.success() {
        let t = errtext.lock().unwrap().clone();
        crate::log::log(format!("[export] ffmpeg fallito:\n{t}"));
        let mut v: Vec<&str> = t.lines().filter(|l| !l.trim().is_empty()).collect();
        let keep = v.len().saturating_sub(5);
        return Err(tf("Export fallito:\n{0}", &[&v.split_off(keep).join("\n")]));
    }
    Ok(())
}

/// Throughput of a counter measured over a sliding window (ignores start-up latency and bursts).
struct Rate {
    samples: RefCell<VecDeque<(Instant, u64)>>,
}

impl Rate {
    fn new() -> Rate {
        Rate { samples: RefCell::new(VecDeque::new()) }
    }

    /// Records `n` completed units and returns units/second once at least ~1 s of history exists.
    fn push(&self, n: u64) -> Option<f64> {
        let now = Instant::now();
        let mut s = self.samples.borrow_mut();
        s.push_back((now, n));
        while s.len() > 2 && now.duration_since(s.front().unwrap().0).as_secs_f64() > 6.0 {
            s.pop_front();
        }
        let (t0, n0) = *s.front().unwrap();
        let dt = now.duration_since(t0).as_secs_f64();
        (dt >= 0.8 && n > n0).then(|| (n - n0) as f64 / dt)
    }
}

/// Runs an export. `progress(fraction 0..1, text, estimated seconds remaining)` is called as work advances;
/// `cancel` aborts it. The estimate comes from the measured speed of the current phase, never from a guess.
pub fn run(tools: &Tools, job: &Job, progress: &dyn Fn(f32, String, Option<f64>), cancel: &AtomicBool) -> Result<Vec<PathBuf>, String> {
    std::fs::create_dir_all(&job.out_dir).map_err(|e| tf("cartella: {0}", &[&e]))?;
    let stem = job.video.file_stem().and_then(|s| s.to_str()).unwrap_or("video").to_string();
    let ext = job.format.ext();
    let total = job.count.max(1);
    let started = Instant::now();
    let mut written = Vec::new();
    progress(0.0, tr("Avvio…").into(), None);

    match &job.ai {
        None => {
            let pattern = job.out_dir.join(format!("{stem}_frame_%06d.{ext}"));
            let rate = Rate::new();
            extract(tools, job, job.format, &pattern, cancel, &|n| {
                let remaining = rate.push(n).map(|r| total.saturating_sub(n) as f64 / r);
                progress((n as f32 / total as f32).min(1.0), tf("Fotogramma {0}/{1}", &[&n.min(total), &total]), remaining)
            })?;
            for i in 0..job.count {
                let p = job.out_dir.join(name(&job.video, job.start + i, ext));
                if p.exists() {
                    written.push(p);
                }
            }
        }
        Some(ai) => {
            // 1) lossless PNG intermediates (with the ffmpeg filter chain), 2) neural engine, 3) final format
            let tmp = tools.temp.join(format!("export_{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&tmp);
            std::fs::create_dir_all(&tmp).map_err(|e| e.to_string())?;
            let r = (|| -> Result<(), String> {
                // extraction is quick; the neural stage dominates and its per-frame time is unknown until the
                // first frame is done, so no estimate is shown (and the bar stays near zero) during extraction
                let rate = Rate::new();
                extract(tools, job, Format::Png, &tmp.join("in_%06d.png"), cancel, &|n| {
                    let _ = rate.push(n);
                    progress(0.02 * (n as f32 / total as f32).min(1.0), tf("Estrazione {0}/{1}", &[&n.min(total), &total]), None)
                })?;
                // moving average of the time one frame takes in the neural stage (including conversion / writing)
                let mut per_frame: Option<f64> = None;
                for i in 0..job.count {
                    if cancel.load(Ordering::Relaxed) {
                        return Err(tr("Annullato").into());
                    }
                    let idx = job.start + i;
                    let input = tmp.join(format!("in_{idx:06}.png"));
                    if !input.exists() {
                        break;
                    }
                    let remaining = per_frame.map(|e| total.saturating_sub(i) as f64 * e);
                    let frac = match remaining {
                        Some(r) => {
                            let el = started.elapsed().as_secs_f64();
                            ((el / (el + r)) as f32).clamp(0.02, 0.999)
                        }
                        None => 0.02,
                    };
                    progress(frac, tf("Rete neurale: fotogramma {0}/{1}", &[&(i + 1), &total]), remaining);
                    crate::log::log(format!("[export] AI {}/{}", i + 1, total));
                    let frame_start = Instant::now();
                    let upscaled = tmp.join(format!("up_{idx:06}.png"));
                    nn::run_file(tools, &input, &upscaled, ai, cancel)
                        .map_err(|e| if e == "annullato" { tr("Annullato").to_string() } else { e })?;
                    let dest = job.out_dir.join(name(&job.video, idx, ext));
                    if job.format == Format::Png {
                        std::fs::copy(&upscaled, &dest).map_err(|e| e.to_string())?;
                    } else {
                        let (pix, extra) = job.format.encoder(false);
                        let out = command(&tools.ffmpeg)
                            .args(["-y", "-hide_banner", "-loglevel", "error", "-i"])
                            .arg(&upscaled)
                            .args(["-vf", &format!("format={pix}")])
                            .args(extra)
                            .arg(&dest)
                            .output()
                            .map_err(|e| e.to_string())?;
                        if !out.status.success() {
                            return Err(ffmpeg_err(&out));
                        }
                    }
                    written.push(dest);
                    let dt = frame_start.elapsed().as_secs_f64();
                    per_frame = Some(match per_frame {
                        None => dt,
                        Some(old) => 0.6 * old + 0.4 * dt,
                    });
                }
                Ok(())
            })();
            let _ = std::fs::remove_dir_all(&tmp);
            r?;
        }
    }
    if written.is_empty() {
        return Err(tr("Nessun fotogramma esportato (posizione oltre la fine del filmato?)").into());
    }
    progress(1.0, tr("Completato").into(), Some(0.0));
    Ok(written)
}
