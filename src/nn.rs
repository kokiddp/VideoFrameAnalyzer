//! Neural upscaling: Real-ESRGAN and waifu2x (ncnn). GPU through the Vulkan tools, CPU through our own ncnn runner.
use crate::decoder::Frame;
use crate::filters::{AiSettings, Engine};
use crate::log::log;
use crate::tools::{command, Tools};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

static COUNTER: AtomicU64 = AtomicU64::new(0);

pub fn temp_path(tools: &Tools, ext: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    tools.temp.join(format!("vfa_{}_{}.{}", std::process::id(), n, ext))
}

/// Scale implied by a model name suffix such as `-x2`, `-x4plus`, `-x4plus-anime`.
fn model_scale(model: &str, default: u32) -> u32 {
    model
        .rsplit_once("-x")
        .map(|(_, r)| r.chars().take_while(|c| c.is_ascii_digit()).collect::<String>())
        .and_then(|d| d.parse().ok())
        .unwrap_or(default)
}

/// Output scale factor of the configured engine.
pub fn output_scale(ai: &AiSettings) -> u32 {
    match ai.engine {
        Engine::Esrgan => model_scale(&ai.model, ai.scale),
        Engine::Waifu2x => w2x_scale(ai),
    }
}

/// upconv models only exist as 2x; scale 1 (denoise only) needs cunet and a noise level.
pub fn w2x_scale(ai: &AiSettings) -> u32 {
    match ai.w2x_scale {
        1 if ai.w2x_model == "cunet" && ai.w2x_noise >= 0 => 1,
        4 => 4,
        _ => 2,
    }
}

pub fn cpu_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).saturating_sub(2).max(1)
}

pub fn engine_available(tools: &Tools, e: Engine) -> bool {
    match e {
        Engine::Esrgan => (tools.realesrgan.is_some() || tools.realesrgan_cpu.is_some()) && !tools.models.is_empty(),
        Engine::Waifu2x => tools.waifu2x.is_some() && !tools.w2x_models.is_empty(),
    }
}

fn build_command(tools: &Tools, ai: &AiSettings, input: &Path, output: &Path) -> Result<Command, String> {
    let device = ai.device();
    let missing = || format!("{} non incluso in questo eseguibile", ai.engine.label());
    let mut cmd;
    match ai.engine {
        Engine::Esrgan => {
            let exe = if device < 0 { tools.realesrgan_cpu.as_ref() } else { tools.realesrgan.as_ref() }.ok_or_else(missing)?;
            let dir = exe.parent().unwrap();
            cmd = command(exe);
            cmd.current_dir(dir)
                .arg("-i")
                .arg(input)
                .arg("-o")
                .arg(output)
                .arg("-m")
                .arg(dir.join("models"))
                .args(["-n", &ai.model, "-s", &output_scale(ai).to_string(), "-f", "png"]);
        }
        Engine::Waifu2x => {
            let gpu_exe = tools.waifu2x.as_ref().ok_or_else(missing)?;
            let mdir = gpu_exe.parent().unwrap().join(format!("models-{}", ai.w2x_model));
            let scale = w2x_scale(ai).to_string();
            if device < 0 {
                let exe = tools.realesrgan_cpu.as_ref().ok_or_else(missing)?;
                cmd = command(exe);
                cmd.current_dir(exe.parent().unwrap())
                    .args(["-e", "waifu2x"])
                    .arg("-i")
                    .arg(input)
                    .arg("-o")
                    .arg(output)
                    .arg("-m")
                    .arg(&mdir)
                    .args(["-N", &ai.w2x_noise.to_string(), "-s", &scale]);
            } else {
                cmd = command(gpu_exe);
                cmd.current_dir(gpu_exe.parent().unwrap())
                    .arg("-i")
                    .arg(input)
                    .arg("-o")
                    .arg(output)
                    .arg("-m")
                    .arg(&mdir)
                    .args(["-n", &ai.w2x_noise.to_string(), "-s", &scale, "-f", "png"]);
            }
        }
    }
    if device < 0 {
        cmd.args(["-j", &cpu_threads().to_string()]);
    } else {
        cmd.args(["-g", &device.to_string()]);
    }
    cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    Ok(cmd)
}

/// Never-set flag for callers that do not support cancellation.
pub static NEVER: AtomicBool = AtomicBool::new(false);

/// Runs the configured engine on a PNG file and writes a PNG. Setting `cancel` kills the process.
/// Returns the tool's stderr lines.
pub fn run_file(tools: &Tools, input: &Path, output: &Path, ai: &AiSettings, cancel: &AtomicBool) -> Result<Vec<String>, String> {
    let device = ai.device();
    let mut cmd = build_command(tools, ai, input, output)?;
    log(format!("[AI] {:?}", cmd));
    let t0 = Instant::now();
    let mut child = cmd.spawn().map_err(|e| {
        let m = format!("{} non avviabile: {e}", ai.engine.label());
        log(format!("[AI] {m}"));
        m
    })?;
    let out = child.stdout.take().unwrap();
    let err = child.stderr.take().unwrap();
    let child = Mutex::new(Some(child));
    let done = AtomicBool::new(false);
    let mut tail: Vec<String> = Vec::new();
    std::thread::scope(|sc| {
        // watcher: kills the process when cancelled
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
        // stdout is drained on its own thread so the child can never block on a full pipe
        sc.spawn(|| {
            for l in BufReader::new(out).lines().map_while(Result::ok) {
                if !l.trim().is_empty() {
                    log(format!("[AI out] {l}"));
                }
            }
        });
        for l in BufReader::new(err).lines().map_while(Result::ok) {
            if l.trim().is_empty() {
                continue;
            }
            log(format!("[AI] {l}"));
            tail.push(l);
        }
        done.store(true, Ordering::Relaxed);
    });
    let status = child.lock().unwrap().take().unwrap().wait().map_err(|e| e.to_string())?;
    if cancel.load(Ordering::Relaxed) {
        let _ = std::fs::remove_file(output);
        log(format!("[AI] annullato dopo {:.1}s", t0.elapsed().as_secs_f64()));
        return Err("annullato".into());
    }
    log(format!("[AI] terminato: {status} in {:.1}s", t0.elapsed().as_secs_f64()));
    if !status.success() || !output.exists() {
        let t: Vec<String> = tail.iter().rev().take(4).rev().cloned().collect();
        return Err(format!(
            "{} fallito ({status}); {}\n{}",
            ai.engine.label(),
            if device < 0 { "CPU" } else { "GPU Vulkan disponibile?" },
            t.join("\n")
        ));
    }
    Ok(tail)
}

pub fn enhance_frame(tools: &Tools, frame: &Frame, ai: &AiSettings, cancel: &AtomicBool) -> Result<Frame, String> {
    log(format!("[AI] fotogramma {} {}x{} -> {}", frame.index, frame.w, frame.h, ai.key()));
    let (inp, outp) = (temp_path(tools, "png"), temp_path(tools, "png"));
    let img = image::RgbaImage::from_raw(frame.w as u32, frame.h as u32, frame.rgba.clone()).ok_or("frame non valido")?;
    img.save(&inp).map_err(|e| e.to_string())?;
    let r = run_file(tools, &inp, &outp, ai, cancel);
    let _ = std::fs::remove_file(&inp);
    r?;
    let res = image::open(&outp).map_err(|e| e.to_string());
    let _ = std::fs::remove_file(&outp);
    let res = res?.to_rgba8();
    log(format!("[AI] risultato {}x{}", res.width(), res.height()));
    Ok(Frame { index: frame.index, w: res.width() as usize, h: res.height() as usize, rgba: res.into_raw() })
}

// ---------------------------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub struct Device {
    /// -1 = CPU, otherwise Vulkan GPU index
    pub id: i32,
    pub name: String,
    /// benchmark time for a 640x360 frame (lower is better)
    pub secs: f64,
}

pub type Benchmark = Vec<(Engine, Vec<Device>)>;

fn cache_file(tools: &Tools) -> PathBuf {
    tools.temp.parent().unwrap_or(&tools.temp).join("bench.txt")
}

pub fn default_settings(tools: &Tools) -> AiSettings {
    AiSettings {
        enabled: false,
        engine: if engine_available(tools, Engine::Esrgan) || !engine_available(tools, Engine::Waifu2x) { Engine::Esrgan } else { Engine::Waifu2x },
        model: tools.models.iter().find(|m| m.as_str() == "realesr-animevideov3-x2").or(tools.models.first()).cloned().unwrap_or_else(|| "realesr-animevideov3-x2".into()),
        scale: 2,
        w2x_model: tools.w2x_models.first().cloned().unwrap_or_else(|| "upconv_7_photo".into()),
        w2x_scale: 2,
        w2x_noise: 1,
        dev_esrgan: 0,
        dev_w2x: 0,
    }
}

/// Times every available processor on a small test frame, per engine; the fastest becomes the default.
pub fn benchmark(tools: &Tools, force: bool) -> Benchmark {
    let engines: Vec<Engine> = Engine::ALL.into_iter().filter(|e| engine_available(tools, *e)).collect();
    let header = format!(
        "v2 threads={} cpu={} esrgan={} w2x={}",
        cpu_threads(),
        tools.realesrgan_cpu.is_some(),
        engines.contains(&Engine::Esrgan),
        engines.contains(&Engine::Waifu2x)
    );
    let cache = cache_file(tools);
    if !force {
        if let Ok(txt) = std::fs::read_to_string(&cache) {
            let mut lines = txt.lines();
            if lines.next() == Some(header.as_str()) {
                let mut res: Benchmark = Vec::new();
                for l in lines {
                    let mut p = l.splitn(4, '\t');
                    let (Some(e), Some(id), Some(secs), Some(name)) = (p.next(), p.next(), p.next(), p.next()) else { continue };
                    let Some(engine) = Engine::ALL.into_iter().find(|x| x.id() == e) else { continue };
                    let (Ok(id), Ok(secs)) = (id.parse(), secs.parse()) else { continue };
                    let d = Device { id, secs, name: name.to_string() };
                    match res.iter_mut().find(|(x, _)| *x == engine) {
                        Some((_, v)) => v.push(d),
                        None => res.push((engine, vec![d])),
                    }
                }
                if !res.is_empty() && engines.iter().all(|e| res.iter().any(|(x, v)| x == e && !v.is_empty())) {
                    log(format!("[bench] risultati in cache: {res:?}"));
                    return res;
                }
            }
        }
    }

    let (w, h) = (640u32, 360u32);
    let img = image::RgbImage::from_fn(w, h, |x, y| {
        let n = (x.wrapping_mul(2654435761) ^ y.wrapping_mul(40503)) >> 7 & 15;
        image::Rgb([((x * 255 / w) as u8).wrapping_add(n as u8), ((y * 255 / h) as u8).wrapping_add(n as u8), (((x + y) * 255 / (w + h)) as u8) ^ n as u8])
    });
    let (inp, outp) = (temp_path(tools, "png"), temp_path(tools, "png"));
    if img.save(&inp).is_err() {
        return vec![];
    }
    let mut result: Benchmark = Vec::new();
    for engine in engines {
        let mut devs: Vec<Device> = Vec::new();
        let mut try_device = |id: i32| -> bool {
            let mut ai = default_settings(tools);
            ai.engine = engine;
            ai.set_device(id);
            // GPU tools compile shaders / fill driver caches on their first run: warm up, then time
            let runs = if id >= 0 { 2 } else { 1 };
            let mut last: Result<Vec<String>, String> = Err("nessuna esecuzione".into());
            let mut secs = 0.0;
            for _ in 0..runs {
                let _ = std::fs::remove_file(&outp);
                let cancel = std::sync::Arc::new(AtomicBool::new(false));
                let done = std::sync::Arc::new(AtomicBool::new(false));
                {
                    let (c, d) = (cancel.clone(), done.clone());
                    std::thread::spawn(move || {
                        for _ in 0..900 {
                            if d.load(Ordering::Relaxed) {
                                return;
                            }
                            std::thread::sleep(Duration::from_millis(100));
                        }
                        c.store(true, Ordering::Relaxed);
                    });
                }
                let t0 = Instant::now();
                last = run_file(tools, &inp, &outp, &ai, &cancel);
                done.store(true, Ordering::Relaxed);
                secs = t0.elapsed().as_secs_f64();
                if last.is_err() {
                    break;
                }
            }
            match last {
                Ok(lines) => {
                    let name = if id < 0 {
                        format!("CPU ({} thread)", cpu_threads())
                    } else {
                        lines
                            .iter()
                            .find_map(|l| {
                                let rest = l.strip_prefix('[')?;
                                let (idx, rest) = rest.split_once(' ')?;
                                (idx.parse::<i32>().ok()? == id).then(|| rest.split(']').next().unwrap_or("").to_string())
                            })
                            .map(|n| format!("GPU {id}: {n}"))
                            .unwrap_or_else(|| format!("GPU {id}"))
                    };
                    log(format!("[bench] {} {name}: {secs:.2}s", engine.label()));
                    devs.push(Device { id, name, secs });
                    true
                }
                Err(e) => {
                    log(format!("[bench] {} dispositivo {id} non utilizzabile: {}", engine.label(), e.lines().next().unwrap_or("")));
                    false
                }
            }
        };
        let has_gpu_tool = match engine {
            Engine::Esrgan => tools.realesrgan.is_some(),
            Engine::Waifu2x => tools.waifu2x.is_some(),
        };
        if has_gpu_tool {
            for id in 0..8 {
                if !try_device(id) {
                    break;
                }
            }
        }
        if tools.realesrgan_cpu.is_some() {
            try_device(-1);
        }
        devs.sort_by(|a, b| a.secs.partial_cmp(&b.secs).unwrap());
        if !devs.is_empty() {
            result.push((engine, devs));
        }
    }
    let _ = std::fs::remove_file(&inp);
    let _ = std::fs::remove_file(&outp);
    if !result.is_empty() {
        let mut txt = header + "\n";
        for (e, devs) in &result {
            for d in devs {
                txt.push_str(&format!("{}\t{}\t{}\t{}\n", e.id(), d.id, d.secs, d.name));
            }
        }
        let _ = std::fs::write(&cache, txt);
    }
    result
}
