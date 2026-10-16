//! Streaming decoder: runs ffmpeg and reads raw RGBA frames from its stdout.
use crate::tools::{command, Tools};
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::process::{Child, Stdio};
use std::sync::mpsc::{sync_channel, Receiver, TryRecvError};
use std::sync::{Arc, Mutex};

pub struct Frame {
    pub index: u64,
    pub w: usize,
    pub h: usize,
    pub rgba: Vec<u8>,
}

pub enum Msg {
    Frame(Arc<Frame>),
    /// The stream is over; carries an error text when ffmpeg failed.
    End(Option<String>),
}

pub type Waker = Arc<dyn Fn() + Send + Sync>;
pub type Killer = Arc<Mutex<Option<Child>>>;

pub struct Stream {
    rx: Receiver<Msg>,
    child: Arc<Mutex<Option<Child>>>,
    peeked: Option<Msg>,
    /// Index of the next frame that will be delivered.
    pub pos: u64,
    pub vf: String,
    pub ended: Option<Option<String>>,
}

/// Seek argument so that the first decoded frame is exactly `frame` (constant frame rate assumed).
pub fn seek_args(frame: u64, fps: f64) -> Vec<String> {
    if frame == 0 {
        vec![]
    } else {
        vec!["-ss".into(), format!("{:.6}", (frame as f64 - 0.5) / fps)]
    }
}

impl Stream {
    pub fn open(
        tools: &Tools,
        path: &Path,
        start: u64,
        fps: f64,
        vf: &str,
        max_frames: Option<u64>,
        queue: usize,
        wake: Waker,
    ) -> Stream {
        let (tx, rx) = sync_channel::<Msg>(queue.max(1));
        let child_slot: Arc<Mutex<Option<Child>>> = Arc::new(Mutex::new(None));
        let mut cmd = command(&tools.ffmpeg);
        cmd.args(["-hide_banner", "-nostats", "-loglevel", "info"]);
        cmd.args(seek_args(start, fps));
        cmd.arg("-i").arg(path);
        cmd.args(["-an", "-sn", "-dn"]);
        if !vf.is_empty() {
            cmd.args(["-vf", vf]);
        }
        cmd.args(["-fps_mode", "passthrough", "-pix_fmt", "rgba", "-f", "rawvideo"]);
        if let Some(n) = max_frames {
            cmd.args(["-frames:v", &n.to_string()]);
        }
        cmd.arg("-").stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());

        crate::log::log(format!("[dec] {:?}", cmd));
        match cmd.spawn() {
            Err(e) => {
                let _ = tx.send(Msg::End(Some(format!("cannot start ffmpeg: {e}"))));
            }
            Ok(mut child) => {
                let stdout = child.stdout.take().unwrap();
                let stderr = child.stderr.take().unwrap();
                *child_slot.lock().unwrap() = Some(child);

                let (size_tx, size_rx) = std::sync::mpsc::channel::<(usize, usize)>();
                let log = Arc::new(Mutex::new(String::new()));
                let log2 = log.clone();
                std::thread::spawn(move || {
                    // ffmpeg separates lines with \n; split manually to be robust to \r
                    let mut rd = BufReader::new(stderr);
                    let mut buf = Vec::new();
                    let (mut in_output, mut sent) = (false, false);
                    while rd.read_until(b'\n', &mut buf).unwrap_or(0) > 0 {
                        let line = String::from_utf8_lossy(&buf).to_string();
                        buf.clear();
                        if line.starts_with("Output #0") {
                            in_output = true;
                        }
                        if in_output && !sent && line.contains("Video: rawvideo") {
                            if let Some(sz) = find_size(&line[line.find("Video: rawvideo").unwrap()..]) {
                                let _ = size_tx.send(sz);
                                sent = true;
                            }
                        }
                        let mut l = log2.lock().unwrap();
                        if l.len() < 8000 {
                            l.push_str(&line);
                        }
                    }
                });

                let child_ref = child_slot.clone();
                std::thread::spawn(move || {
                    let mut out = BufReader::with_capacity(1 << 20, stdout);
                    let mut index = start;
                    let size = size_rx.recv_timeout(std::time::Duration::from_secs(60)).ok();
                    let mut n = 0u64;
                    if let Some((w, h)) = size {
                        loop {
                            let mut data = vec![0u8; w * h * 4];
                            if out.read_exact(&mut data).is_err() {
                                break;
                            }
                            let f = Arc::new(Frame { index, w, h, rgba: data });
                            index += 1;
                            n += 1;
                            if tx.send(Msg::Frame(f)).is_err() {
                                kill(&child_ref);
                                return;
                            }
                            wake();
                        }
                    }
                    kill(&child_ref);
                    let err = if n == 0 {
                        std::thread::sleep(std::time::Duration::from_millis(100));
                        let l = log.lock().unwrap();
                        let tail: Vec<&str> = l.lines().rev().take(6).collect();
                        let msg = tail.into_iter().rev().collect::<Vec<_>>().join("\n");
                        crate::log::log(format!("[dec] nessun fotogramma, ffmpeg:\n{msg}"));
                        Some(msg)
                    } else {
                        None
                    };
                    let _ = tx.send(Msg::End(err));
                    wake();
                });
            }
        }
        Stream { rx, child: child_slot, peeked: None, pos: start, vf: vf.to_string(), ended: None }
    }

    /// Handle that lets another thread kill the ffmpeg process (used to abort stale block loads).
    pub fn killer(&self) -> Killer {
        self.child.clone()
    }

    fn fill(&mut self) {
        if self.peeked.is_none() {
            match self.rx.try_recv() {
                Ok(m) => self.peeked = Some(m),
                Err(TryRecvError::Empty) => {}
                Err(TryRecvError::Disconnected) => self.peeked = Some(Msg::End(None)),
            }
        }
    }

    /// Non-blocking: next frame if available.
    pub fn try_next(&mut self) -> Option<Arc<Frame>> {
        self.fill();
        match self.peeked.take() {
            Some(Msg::Frame(f)) => {
                self.pos = f.index + 1;
                Some(f)
            }
            Some(Msg::End(e)) => {
                self.ended = Some(e);
                None
            }
            None => None,
        }
    }

    /// Blocking: collects every remaining frame (for chunk loads).
    pub fn collect_all(mut self) -> (Vec<Arc<Frame>>, Option<String>) {
        let mut v = Vec::new();
        loop {
            let m = match self.peeked.take() {
                Some(m) => m,
                None => self.rx.recv().unwrap_or(Msg::End(None)),
            };
            match m {
                Msg::Frame(f) => v.push(f),
                Msg::End(e) => return (v, e),
            }
        }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        kill(&self.child);
    }
}

pub fn kill(c: &Killer) {
    if let Some(mut ch) = c.lock().unwrap().take() {
        let _ = ch.kill();
        let _ = ch.wait();
    }
}

/// Finds the first `WxH` token in a stream description line.
fn find_size(s: &str) -> Option<(usize, usize)> {
    for tok in s.split(|c: char| c == ' ' || c == ',' || c == '[') {
        if let Some((w, h)) = tok.split_once('x') {
            if let (Ok(w), Ok(h)) = (w.parse::<usize>(), h.parse::<usize>()) {
                if w > 0 && h > 0 {
                    return Some((w, h));
                }
            }
        }
    }
    None
}
