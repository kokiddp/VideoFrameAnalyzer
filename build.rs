//! Packs everything under `assets/<target_os>/` (ffmpeg, Real-ESRGAN, ...) into a single
//! zstd-compressed tar that is embedded in the executable and unpacked on first run.
use std::{env, fs, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=VFA_NO_BUNDLE");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("assets").join(&os);
    println!("cargo:rerun-if-changed={}", dir.display());

    let mut bytes = Vec::new();
    if dir.is_dir() && env::var_os("VFA_NO_BUNDLE").is_none() {
        let level = if env::var("PROFILE").map(|p| p == "release").unwrap_or(false) { 19 } else { 3 };
        let mut enc = zstd::Encoder::new(Vec::new(), level).unwrap();
        let _ = enc.multithread(std::thread::available_parallelism().map(|n| n.get() as u32).unwrap_or(1));
        let mut tar = tar::Builder::new(enc);
        tar.follow_symlinks(false);
        tar.append_dir_all(".", &dir).unwrap();
        bytes = tar.into_inner().unwrap().finish().unwrap();
    } else {
        println!("cargo:warning=no assets/{os} bundle found: the app will use ffmpeg from PATH");
    }
    // cheap content id (FNV-1a over the compressed bundle) used as cache directory name
    let mut h: u64 = 0xcbf29ce484222325;
    for b in &bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    fs::write(out.join("bundle.tar.zst"), &bytes).unwrap();
    fs::write(out.join("bundle.id"), format!("{:016x}", h)).unwrap();
}
