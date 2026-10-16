//! Tiny in-memory log shown in the debug window.
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

static LINES: Mutex<Vec<String>> = Mutex::new(Vec::new());
static START: OnceLock<Instant> = OnceLock::new();
const MAX: usize = 3000;

pub fn log(msg: impl AsRef<str>) {
    let t = START.get_or_init(Instant::now).elapsed().as_secs_f64();
    let mut l = LINES.lock().unwrap();
    for line in msg.as_ref().lines() {
        l.push(format!("[{t:8.2}] {line}"));
    }
    if l.len() > MAX {
        let n = l.len() - MAX;
        l.drain(..n);
    }
}

pub fn snapshot() -> Vec<String> {
    LINES.lock().unwrap().clone()
}

pub fn clear() {
    LINES.lock().unwrap().clear();
}
