//! Locates (and on first run unpacks) the embedded ffmpeg / Real-ESRGAN binaries.
use std::path::{Path, PathBuf};
use std::process::Command;

static BUNDLE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/bundle.tar.zst"));
const BUNDLE_ID: &str = include_str!(concat!(env!("OUT_DIR"), "/bundle.id"));

#[derive(Clone, Debug)]
pub struct Tools {
    pub ffmpeg: PathBuf,
    /// Real-ESRGAN (ncnn/Vulkan) executable, if bundled.
    pub realesrgan: Option<PathBuf>,
    /// CPU-only runner (ncnn without Vulkan), next to the Vulkan one.
    pub realesrgan_cpu: Option<PathBuf>,
    pub models: Vec<String>,
    /// waifu2x-ncnn-vulkan executable and its model directories (names without the `models-` prefix)
    pub waifu2x: Option<PathBuf>,
    pub w2x_models: Vec<String>,
    pub temp: PathBuf,
}

const EXE: &str = if cfg!(windows) { ".exe" } else { "" };

/// Prepares the tools; the first run extracts the bundle into the user cache directory.
pub fn prepare() -> Result<Tools, String> {
    let base = dirs::cache_dir().unwrap_or_else(std::env::temp_dir).join("VideoFrameAnalyzer");
    let temp = base.join("tmp");
    let _ = std::fs::create_dir_all(&temp);

    if BUNDLE.is_empty() {
        // Development fallback: use whatever is on PATH.
        let ffmpeg = std::env::var_os("VFA_FFMPEG").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("ffmpeg"));
        let realesrgan = std::env::var_os("VFA_REALESRGAN").map(PathBuf::from);
        let models = realesrgan
            .as_ref()
            .and_then(|r| std::fs::read_dir(r.parent()?.join("models")).ok())
            .map(|rd| rd.flatten().filter_map(|e| e.file_name().to_str().and_then(|n| n.strip_suffix(".param")).map(String::from)).collect())
            .unwrap_or_default();
        let realesrgan_cpu = std::env::var_os("VFA_ESRGAN_CPU").map(PathBuf::from);
        let waifu2x = std::env::var_os("VFA_WAIFU2X").map(PathBuf::from);
        let w2x_models = waifu2x.as_deref().map(w2x_model_dirs).unwrap_or_default();
        return Ok(Tools { ffmpeg, realesrgan, realesrgan_cpu, models, waifu2x, w2x_models, temp });
    }

    let dir = base.join(BUNDLE_ID.trim());
    let marker = dir.join(".complete");
    if !marker.exists() {
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
        let dec = zstd::Decoder::new(BUNDLE).map_err(|e| e.to_string())?;
        tar::Archive::new(dec).unpack(&dir).map_err(|e| format!("unpack failed: {e}"))?;
        std::fs::write(&marker, b"ok").map_err(|e| e.to_string())?;
        // drop stale versions
        if let Ok(rd) = std::fs::read_dir(&base) {
            for e in rd.flatten() {
                if e.path() != dir && e.file_name() != "tmp" && e.path().is_dir() {
                    let _ = std::fs::remove_dir_all(e.path());
                }
            }
        }
    }

    let ffmpeg = dir.join(format!("ffmpeg{EXE}"));
    let re = dir.join("realesrgan").join(format!("realesrgan-ncnn-vulkan{EXE}"));
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        for p in [&ffmpeg, &re, &dir.join("realesrgan").join("esrgan-cpu"), &dir.join("waifu2x").join("waifu2x-ncnn-vulkan")] {
            if let Ok(m) = std::fs::metadata(p) {
                let mut perm = m.permissions();
                perm.set_mode(0o755);
                let _ = std::fs::set_permissions(p, perm);
            }
        }
    }
    if !ffmpeg.exists() {
        return Err("ffmpeg missing from the embedded bundle".into());
    }
    let (realesrgan, models) = if re.exists() {
        let mut m: Vec<String> = std::fs::read_dir(dir.join("realesrgan").join("models"))
            .map(|rd| {
                rd.flatten()
                    .filter_map(|e| e.file_name().to_str().and_then(|n| n.strip_suffix(".param")).map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        m.sort();
        (Some(re), m)
    } else {
        (None, vec![])
    };
    let cpu = dir.join("realesrgan").join(format!("esrgan-cpu{EXE}"));
    let realesrgan_cpu = cpu.exists().then_some(cpu);
    crate::log::log(format!("[init] ffmpeg={} realesrgan={:?} cpu={:?} modelli={:?}", ffmpeg.display(), realesrgan, realesrgan_cpu, models));
    let w2x = dir.join("waifu2x").join(format!("waifu2x-ncnn-vulkan{EXE}"));
    let waifu2x = w2x.exists().then_some(w2x);
    let w2x_models = waifu2x.as_deref().map(w2x_model_dirs).unwrap_or_default();
    crate::log::log(format!("[init] waifu2x={:?} modelli={:?}", waifu2x, w2x_models));
    Ok(Tools { ffmpeg, realesrgan, realesrgan_cpu, models, waifu2x, w2x_models, temp })
}

fn w2x_model_dirs(exe: &Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(exe.parent().unwrap())
        .map(|rd| rd.flatten().filter_map(|e| e.file_name().to_str().and_then(|n| n.strip_prefix("models-")).map(String::from)).collect())
        .unwrap_or_default();
    // best general-purpose / fastest first
    v.sort_by_key(|m| match m.as_str() {
        "upconv_7_photo" => 0,
        "upconv_7_anime_style_art_rgb" => 1,
        "cunet" => 2,
        _ => 3,
    });
    v
}

/// A `Command` that does not flash a console window on Windows.
pub fn command(exe: &Path) -> Command {
    let mut c = Command::new(exe);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        c.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    c
}
