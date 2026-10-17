//! Video metadata, obtained by parsing `ffmpeg -i` output (so ffprobe is not needed).
use crate::i18n::{tf, tr};
use crate::tools::{command, Tools};
use std::path::Path;

#[derive(Clone, Debug)]
pub struct VideoInfo {
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub duration: Option<f64>,
    pub frames: Option<u64>,
    pub codec: String,
    pub pix_fmt: String,
    pub high_bit_depth: bool,
}

/// Splits on commas that are not inside parentheses or brackets.
fn split_top(s: &str) -> Vec<&str> {
    let (mut depth, mut start, mut out) = (0i32, 0, Vec::new());
    for (i, c) in s.char_indices() {
        match c {
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            ',' if depth == 0 => {
                out.push(s[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push(s[start..].trim());
    out
}

fn snap_fps(f: f64) -> f64 {
    for (n, d) in [(24000.0, 1001.0), (30000.0, 1001.0), (60000.0, 1001.0), (120000.0, 1001.0)] {
        if (f - n / d).abs() < 0.02 {
            return n / d;
        }
    }
    f
}

fn parse_size(tok: &str) -> Option<(u32, u32)> {
    let t = tok.split_whitespace().next()?;
    let (w, h) = t.split_once('x')?;
    Some((w.parse().ok()?, h.parse().ok()?))
}

pub fn probe(tools: &Tools, path: &Path) -> Result<VideoInfo, String> {
    let out = command(&tools.ffmpeg)
        .args(["-hide_banner", "-i"])
        .arg(path)
        .output()
        .map_err(|e| tf("cannot run ffmpeg: {0}", &[&e]))?;
    let text = String::from_utf8_lossy(&out.stderr).to_string();

    let duration = text.lines().find_map(|l| {
        let l = l.trim();
        let rest = l.strip_prefix("Duration:")?.trim();
        let t = rest.split(',').next()?.trim();
        let mut p = t.split(':');
        let (h, m, s): (f64, f64, f64) = (p.next()?.parse().ok()?, p.next()?.parse().ok()?, p.next()?.parse().ok()?);
        Some(h * 3600.0 + m * 60.0 + s)
    });

    let line = text
        .lines()
        .find(|l| l.contains("Video:") && l.contains("Stream #") && !l.contains("attached pic"))
        .ok_or_else(|| {
            let last = text.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("");
            tf("no video stream found ({0})", &[&last.trim()])
        })?;
    let desc = &line[line.find("Video:").unwrap() + 6..];
    let parts = split_top(desc);
    let codec = parts.first().map(|s| s.split_whitespace().next().unwrap_or("").to_string()).unwrap_or_default();
    let pix_fmt = parts
        .get(1)
        .map(|s| s.split('(').next().unwrap_or("").trim().to_string())
        .unwrap_or_default();
    let (width, height) = parts.iter().skip(1).find_map(|p| parse_size(p)).ok_or(tr("cannot determine video size"))?;
    let fps_tok = |suffix: &str| {
        parts.iter().find_map(|p| p.strip_suffix(suffix).and_then(|n| n.trim().replace('k', "000").parse::<f64>().ok()))
    };
    let fps = snap_fps(fps_tok(" fps").or_else(|| fps_tok(" tbr")).filter(|f| *f > 0.0 && *f < 1000.0).unwrap_or(25.0));
    let high = ["10", "12", "14", "16"].iter().any(|d| pix_fmt.contains(d)) && !pix_fmt.starts_with("pal");
    Ok(VideoInfo {
        width,
        height,
        fps,
        duration,
        frames: duration.map(|d| (d * fps).round() as u64).filter(|n| *n > 0),
        codec,
        pix_fmt,
        high_bit_depth: high,
    })
}
