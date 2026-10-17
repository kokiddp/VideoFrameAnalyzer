//! Self-update: looks up the latest GitHub release, downloads the portable build for this platform, checks its
//! SHA-256 and swaps the running executable. Network access goes through the system `curl` (shipped with
//! Windows 10+ and virtually every Linux distribution), so no TLS stack is linked into the program.
use crate::decoder::Waker;
use crate::i18n::{tf, tr};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc::{channel, Receiver};

const API_URL: &str = "https://api.github.com/repos/kokiddp/VideoFrameAnalyzer/releases/latest";
const PLATFORM: &str = if cfg!(windows) { "windows-x86_64.zip" } else { "linux-x86_64.tar.gz" };
const EXE: &str = if cfg!(windows) { "video-frame-analyzer.exe" } else { "video-frame-analyzer" };

#[derive(Clone, Debug, PartialEq)]
pub struct Release {
    pub version: String,
    pub notes: String,
    pub asset_name: String,
    pub asset_url: String,
    pub asset_size: u64,
    pub sha_url: Option<String>,
}

fn curl() -> Command {
    let mut c = Command::new("curl");
    c.args(["-fsSL", "--retry", "2", "--connect-timeout", "15", "-A", "VideoFrameAnalyzer-updater"]);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        c.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    c
}

fn run(mut c: Command) -> Result<Vec<u8>, String> {
    let out = c.output().map_err(|e| tf("curl is not available: {0}", &[&e]))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(tf("download failed: {0}", &[&err.trim()]));
    }
    Ok(out.stdout)
}

/// Numeric components of "v1.2.3" (a pre-release suffix is ignored).
fn parse_version(v: &str) -> Vec<u64> {
    v.trim().trim_start_matches('v').split(['.', '-', '+']).map_while(|p| p.parse().ok()).collect()
}

pub fn is_newer(candidate: &str, current: &str) -> bool {
    let (mut a, mut b) = (parse_version(candidate), parse_version(current));
    let n = a.len().max(b.len());
    a.resize(n, 0);
    b.resize(n, 0);
    !a.is_empty() && a > b
}

fn parse_release(body: &[u8]) -> Result<Option<Release>, String> {
    let v: serde_json::Value = serde_json::from_slice(body).map_err(|e| tf("unexpected reply from GitHub: {0}", &[&e]))?;
    let tag = v["tag_name"].as_str().ok_or_else(|| tr("unexpected reply from GitHub: no version").to_string())?;
    let version = tag.trim_start_matches('v').to_string();
    if !is_newer(&version, env!("CARGO_PKG_VERSION")) {
        return Ok(None);
    }
    let assets = v["assets"].as_array().cloned().unwrap_or_default();
    let find = |suffix: &str| assets.iter().find(|a| a["name"].as_str().is_some_and(|n| n.ends_with(suffix)));
    let asset = find(PLATFORM).ok_or_else(|| tf("release {0} has no build for this platform", &[&version]))?;
    let name = asset["name"].as_str().unwrap_or_default().to_string();
    Ok(Some(Release {
        version,
        notes: v["body"].as_str().unwrap_or_default().trim().to_string(),
        asset_url: asset["browser_download_url"].as_str().unwrap_or_default().to_string(),
        asset_size: asset["size"].as_u64().unwrap_or(0),
        sha_url: find(&format!("{PLATFORM}.sha256")).and_then(|a| a["browser_download_url"].as_str()).map(String::from),
        asset_name: name,
    }))
}

/// Latest release if it is newer than the running version.
pub fn check() -> Result<Option<Release>, String> {
    let mut c = curl();
    c.args(["-H", "Accept: application/vnd.github+json", API_URL]);
    parse_release(&run(c)?)
}

fn sha256_file(p: &Path) -> Result<String, String> {
    let mut f = std::fs::File::open(p).map_err(|e| e.to_string())?;
    let mut h = Sha256::new();
    std::io::copy(&mut f, &mut h).map_err(|e| e.to_string())?;
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

fn work_dir() -> PathBuf {
    dirs::cache_dir().unwrap_or_else(std::env::temp_dir).join("VideoFrameAnalyzer").join("update")
}

/// Path of the partial download (its size drives the progress bar).
fn download_path(rel: &Release) -> PathBuf {
    work_dir().join(&rel.asset_name)
}

fn extract(archive: &Path, dest: &Path) -> Result<(), String> {
    #[cfg(windows)]
    let mut c = {
        // the Windows tar (bsdtar) reads zip files; avoid any other tar that may be first on the PATH
        let root = std::env::var_os("SystemRoot").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(r"C:\Windows"));
        let mut c = Command::new(root.join("System32").join("tar.exe"));
        use std::os::windows::process::CommandExt;
        c.creation_flags(0x0800_0000);
        c
    };
    #[cfg(not(windows))]
    let mut c = Command::new("tar");
    let out = c.arg("-xf").arg(archive).arg("-C").arg(dest).output().map_err(|e| tf("cannot extract the update: {0}", &[&e]))?;
    if !out.status.success() {
        return Err(tf("cannot extract the update: {0}", &[&String::from_utf8_lossy(&out.stderr).trim()]));
    }
    Ok(())
}

/// Downloads, verifies and unpacks the release; returns the path of the new executable.
fn download(rel: &Release) -> Result<PathBuf, String> {
    let dir = work_dir();
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let archive = download_path(rel);
    let mut c = curl();
    c.arg("-o").arg(&archive).arg(&rel.asset_url);
    run(c)?;

    // the checksum is mandatory when the release publishes one
    if let Some(u) = &rel.sha_url {
        let mut c = curl();
        c.arg(u);
        let text = String::from_utf8_lossy(&run(c)?).to_string();
        let expected = text.split_whitespace().next().unwrap_or("").to_ascii_lowercase();
        if sha256_file(&archive)? != expected {
            let _ = std::fs::remove_file(&archive);
            return Err(tr("the downloaded file is corrupt (checksum mismatch)").to_string());
        }
    }
    let out = dir.join("unpacked");
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    extract(&archive, &out)?;
    let _ = std::fs::remove_file(&archive);
    // the archive holds one folder with the executable inside
    for e in std::fs::read_dir(&out).map_err(|e| e.to_string())?.flatten() {
        let p = e.path().join(EXE);
        if p.is_file() {
            return Ok(p);
        }
    }
    Err(tr("the update archive does not contain the program").to_string())
}

fn old_path(exe: &Path) -> PathBuf {
    let mut s = exe.as_os_str().to_owned();
    s.push(".old");
    PathBuf::from(s)
}

/// Replaces the running executable with `new`. The old file is renamed (allowed even while it runs on
/// Windows) and removed by `cleanup` on the next start.
fn install(new: &Path) -> Result<PathBuf, String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let old = old_path(&exe);
    let _ = std::fs::remove_file(&old);
    std::fs::rename(&exe, &old).map_err(|e| tf("cannot replace the program ({0}): is the folder writable?", &[&e]))?;
    if let Err(e) = std::fs::copy(new, &exe) {
        let _ = std::fs::remove_file(&exe);
        let _ = std::fs::rename(&old, &exe);
        return Err(tf("cannot replace the program ({0}): is the folder writable?", &[&e]));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&exe, std::fs::Permissions::from_mode(0o755));
    }
    let _ = std::fs::remove_dir_all(work_dir());
    Ok(exe)
}

/// Startup housekeeping: remove the executable left behind by the previous update.
fn cleanup() {
    if let Ok(exe) = std::env::current_exe() {
        let _ = std::fs::remove_file(old_path(&exe));
    }
}

fn relaunch(exe: &Path) {
    let _ = Command::new(exe).spawn();
    std::process::exit(0);
}

// ---- preferences -------------------------------------------------------------------------------------------

fn prefs_path() -> Option<PathBuf> {
    Some(dirs::config_dir()?.join("VideoFrameAnalyzer").join("update"))
}

#[derive(Clone, Debug)]
struct Prefs {
    auto_check: bool,
    /// version the user chose not to be reminded about
    skip: String,
}

impl Prefs {
    fn load() -> Prefs {
        let mut p = Prefs { auto_check: true, skip: String::new() };
        if let Some(t) = prefs_path().and_then(|f| std::fs::read_to_string(f).ok()) {
            for l in t.lines() {
                match l.split_once('=') {
                    Some(("auto", v)) => p.auto_check = v.trim() != "0",
                    Some(("skip", v)) => p.skip = v.trim().to_string(),
                    _ => {}
                }
            }
        }
        p
    }
    fn save(&self) {
        if let Some(f) = prefs_path() {
            if let Some(d) = f.parent() {
                let _ = std::fs::create_dir_all(d);
            }
            let _ = std::fs::write(f, format!("auto={}\nskip={}\n", self.auto_check as u8, self.skip));
        }
    }
}

// ---- UI state ----------------------------------------------------------------------------------------------

enum Msg {
    Checked { manual: bool, result: Result<Option<Release>, String> },
    Installed(Result<PathBuf, String>),
}

#[derive(Clone)]
enum Phase {
    Idle,
    Checking,
    /// a newer release is on offer (window open)
    Offer(Release),
    Installing(Release),
    /// result of a manual check / failure, shown in a small window: (text, is_error)
    Info(String, bool),
}

pub struct Updater {
    prefs: Prefs,
    phase: Phase,
    rx: Option<Receiver<Msg>>,
    wake: Waker,
}

impl Updater {
    pub fn new(wake: Waker) -> Updater {
        cleanup();
        let mut u = Updater { prefs: Prefs::load(), phase: Phase::Idle, rx: None, wake };
        if u.prefs.auto_check {
            u.start_check(false);
        }
        u
    }

    fn start_check(&mut self, manual: bool) {
        if matches!(self.phase, Phase::Checking | Phase::Installing(_)) {
            return;
        }
        self.phase = if manual { Phase::Checking } else { Phase::Idle };
        let (tx, rx) = channel();
        self.rx = Some(rx);
        let wake = self.wake.clone();
        std::thread::spawn(move || {
            let _ = tx.send(Msg::Checked { manual, result: check() });
            wake();
        });
    }

    fn start_install(&mut self, rel: Release) {
        self.phase = Phase::Installing(rel.clone());
        let (tx, rx) = channel();
        self.rx = Some(rx);
        let wake = self.wake.clone();
        std::thread::spawn(move || {
            let _ = tx.send(Msg::Installed(download(&rel).and_then(|p| install(&p))));
            wake();
        });
    }

    pub fn poll(&mut self) {
        let Some(rx) = &self.rx else { return };
        let Ok(msg) = rx.try_recv() else { return };
        self.rx = None;
        match msg {
            Msg::Checked { manual, result } => {
                self.phase = match result {
                    Ok(Some(r)) if manual || r.version != self.prefs.skip => Phase::Offer(r),
                    Ok(Some(_)) => Phase::Idle,
                    Ok(None) if manual => Phase::Info(tf("Stai usando l'ultima versione ({0}).", &[&env!("CARGO_PKG_VERSION")]), false),
                    Ok(None) => Phase::Idle,
                    Err(e) if manual => Phase::Info(tf("Impossibile cercare aggiornamenti: {0}", &[&e]), true),
                    Err(_) => Phase::Idle, // silent on the automatic check (offline, rate limit...)
                };
            }
            Msg::Installed(Ok(exe)) => relaunch(&exe),
            Msg::Installed(Err(e)) => self.phase = Phase::Info(tf("Aggiornamento non riuscito: {0}", &[&e]), true),
        }
    }

    /// Entries of the Options menu.
    pub fn menu(&mut self, ui: &mut eframe::egui::Ui) {
        let busy = matches!(self.phase, Phase::Checking | Phase::Installing(_));
        if ui.add_enabled(!busy, eframe::egui::Button::new(tr("Cerca aggiornamenti"))).clicked() {
            self.start_check(true);
            ui.close();
        }
        if ui.checkbox(&mut self.prefs.auto_check, tr("Cerca aggiornamenti all'avvio")).changed() {
            self.prefs.save();
        }
    }

    pub fn window(&mut self, ctx: &eframe::egui::Context) {
        use eframe::egui::{self, Align2, RichText, Vec2};
        let mut next = None;
        let win = |title: &str| egui::Window::new(title.to_string()).collapsible(false).resizable(false).anchor(Align2::CENTER_CENTER, Vec2::ZERO);
        match self.phase.clone() {
            Phase::Idle => {}
            Phase::Checking => {
                win(tr("Aggiornamenti")).show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label(tr("Controllo degli aggiornamenti…"));
                    });
                });
            }
            Phase::Offer(rel) => {
                win(tr("Aggiornamento disponibile")).fixed_size([440.0, 0.0]).show(ctx, |ui| {
                    ui.label(RichText::new(tf("La versione {0} è disponibile (in uso: {1}).", &[&rel.version, &env!("CARGO_PKG_VERSION")])).strong());
                    if !rel.notes.is_empty() {
                        ui.add_space(4.0);
                        egui::ScrollArea::vertical().max_height(200.0).show(ui, |ui| {
                            ui.label(&rel.notes);
                        });
                    }
                    ui.add_space(6.0);
                    ui.horizontal(|ui| {
                        if ui.button(RichText::new(tr("Aggiorna ora")).strong()).clicked() {
                            next = Some(Phase::Installing(rel.clone()));
                        }
                        if ui.button(tr("Più tardi")).clicked() {
                            next = Some(Phase::Idle);
                        }
                        if ui.button(tr("Salta questa versione")).clicked() {
                            self.prefs.skip = rel.version.clone();
                            self.prefs.save();
                            next = Some(Phase::Idle);
                        }
                    });
                });
            }
            Phase::Installing(rel) => {
                let done = std::fs::metadata(download_path(&rel)).map(|m| m.len()).unwrap_or(0);
                let frac = if rel.asset_size > 0 { (done as f32 / rel.asset_size as f32).clamp(0.0, 1.0) } else { 0.0 };
                win(tr("Aggiornamento in corso")).fixed_size([380.0, 0.0]).show(ctx, |ui| {
                    ui.label(tf("Download della versione {0}…", &[&rel.version]));
                    ui.add(egui::ProgressBar::new(frac).show_percentage().animate(true));
                    ui.label(tr("Il programma si riavvierà al termine."));
                });
                ctx.request_repaint_after(std::time::Duration::from_millis(200));
            }
            Phase::Info(text, err) => {
                win(tr("Aggiornamenti")).fixed_size([380.0, 0.0]).show(ctx, |ui| {
                    ui.colored_label(if err { egui::Color32::LIGHT_RED } else { egui::Color32::LIGHT_GREEN }, text);
                    if ui.button(tr("Chiudi")).clicked() {
                        next = Some(Phase::Idle);
                    }
                });
            }
        }
        match next {
            // the install thread takes over; an Offer is only left through this transition
            Some(Phase::Installing(rel)) => self.start_install(rel),
            Some(p) => self.phase = p,
            None => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_order() {
        assert!(is_newer("0.5.0", "0.4.0"));
        assert!(is_newer("v0.10.0", "0.9.9"));
        assert!(is_newer("1.0", "0.9.9"));
        assert!(!is_newer("0.4.0", "0.4.0"));
        assert!(!is_newer("0.4", "0.4.0"));
        assert!(!is_newer("0.3.9", "0.4.0"));
        assert!(!is_newer("garbage", "0.4.0"));
    }

    #[test]
    fn release_parsing() {
        let json = format!(
            r#"{{"tag_name":"v99.0.0","body":"notes","assets":[
            {{"name":"video-frame-analyzer-99.0.0-{p}","browser_download_url":"https://x/a","size":10}},
            {{"name":"video-frame-analyzer-99.0.0-{p}.sha256","browser_download_url":"https://x/s","size":1}}]}}"#,
            p = PLATFORM
        );
        let r = parse_release(json.as_bytes()).unwrap().unwrap();
        assert_eq!(r.version, "99.0.0");
        assert_eq!(r.asset_url, "https://x/a");
        assert_eq!(r.sha_url.as_deref(), Some("https://x/s"));
        assert!(parse_release(br#"{"tag_name":"0.0.1","assets":[]}"#).unwrap().is_none());
    }
}
