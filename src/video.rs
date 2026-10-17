//! Playback state of one opened video: frame cache, decoder stream and on-demand AI stage.
use crate::i18n::{tf};
use crate::decoder::{Frame, Killer, Stream, Waker};
use crate::filters::AiSettings;
use crate::nn;
use crate::probe::VideoInfo;
use crate::tools::Tools;
use eframe::egui::{ColorImage, Context, TextureHandle, TextureOptions};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, TryRecvError};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const CACHE_BYTES: usize = 896 << 20;
const RESTART_THROTTLE: Duration = Duration::from_millis(80);

/// Where playback is confined (CUE range) and whether it loops.
#[derive(Clone, Copy)]
pub struct PlayRange {
    pub lo: u64,
    pub hi: Option<u64>,
    pub looping: bool,
    /// playback speed multiplier (scales how far ahead reverse playback decodes)
    pub speed: f32,
}

type ChunkResult = (Vec<Arc<Frame>>, Option<String>);

pub struct AiResult {
    pub tex: TextureHandle,
    pub idx: u64,
    pub gen: u64,
    pub key: String,
}

pub struct Video {
    pub path: PathBuf,
    pub info: VideoInfo,
    pub vf: String,
    /// Bumped whenever the filter chain changes (invalidates cache and AI results).
    pub gen: u64,
    stream: Option<Stream>,
    cache: BTreeMap<u64, Arc<Frame>>,
    cache_bytes: usize,
    /// Index of the frame currently shown.
    pub cur: u64,
    /// Index the user/clock wants to see.
    pub want: u64,
    pub shown: Option<Arc<Frame>>,
    pub shown_since: Instant,
    pub tex: Option<TextureHandle>,
    /// (identity of the uploaded frame, nearest): the texture is re-uploaded whenever the frame object changes,
    /// also when the same index is re-decoded with another filter chain
    tex_for: Option<(usize, u64, u64, bool)>,
    /// (receiver, first frame, last frame) of the block being loaded
    chunk: Option<(Receiver<ChunkResult>, u64, u64, Killer)>,
    /// True while the wanted frame cannot be produced by the live stream (seek / direction change in progress):
    /// the playback clock holds instead of running away from the decoder.
    pub hard_wait: bool,
    /// Filter generation of the frame currently on screen (equals `gen` once the new chain is visible).
    pub shown_gen: u64,
    /// tail block already requested for this end (avoids re-requesting frames that do not exist)
    tail_tried: Option<u64>,
    /// Stream already positioned at frame 0, prepared while the end is approaching so looping is seamless.
    loop_next: Option<Stream>,
    loop_next_at: u64,
    /// -1 reverse play, 1 forward play, 0 paused: decides which cached frames are evicted first.
    evict_dir: i8,
    last_restart: Instant,
    pub eof: Option<u64>,
    pub error: Option<String>,
    pub loading: bool,
    // AI stage
    pub ai: Option<AiResult>,
    ai_pending: Option<(u64, u64, String, Arc<AtomicBool>, Receiver<Result<Frame, String>>)>,
    pub ai_busy: bool,
    pub ai_started: Option<Instant>,
    pub ai_err: Option<String>,
    ai_err_at: Option<(u64, String)>,
}

impl Video {
    pub fn new(path: PathBuf, info: VideoInfo, vf: String) -> Video {
        Video {
            path,
            info,
            vf,
            gen: 0,
            stream: None,
            cache: BTreeMap::new(),
            cache_bytes: 0,
            cur: 0,
            want: 0,
            shown: None,
            shown_since: Instant::now(),
            tex: None,
            tex_for: None,
            chunk: None,
            hard_wait: false,
            shown_gen: 0,
            tail_tried: None,
            loop_next: None,
            loop_next_at: 0,
            evict_dir: 0,
            last_restart: Instant::now() - Duration::from_secs(1),
            eof: None,
            error: None,
            loading: true,
            ai: None,
            ai_pending: None,
            ai_busy: false,
            ai_started: None,
            ai_err: None,
            ai_err_at: None,
        }
    }

    /// Last frame index (exact once the end was reached, estimated before).
    pub fn last_frame(&self) -> Option<u64> {
        self.eof.or(self.info.frames.map(|n| n.saturating_sub(1)))
    }

    pub fn clamp_want(&self, idx: i64) -> u64 {
        let mut i = idx.max(0) as u64;
        if let Some(e) = self.eof {
            i = i.min(e);
        }
        i
    }

    pub fn set_filters(&mut self, vf: String) {
        self.vf = vf;
        self.gen += 1;
        self.stream = None;
        self.loop_next = None;
        self.chunk = None;
        self.cache.clear();
        self.cache_bytes = 0;
        self.ai = None;
        self.ai_err = None;
        self.eof = None;
        self.error = None;
        self.want = self.cur;
        self.last_restart = Instant::now() - Duration::from_secs(1);
    }

    fn insert(&mut self, f: Arc<Frame>) {
        // a decoded frame beyond the believed end proves the end was wrong
        if matches!(self.eof, Some(e) if f.index > e) {
            self.eof = None;
        }
        self.cache_bytes += f.rgba.len();
        if let Some(old) = self.cache.insert(f.index, f) {
            self.cache_bytes -= old.rgba.len();
        }
        while self.cache_bytes > CACHE_BYTES && self.cache.len() > 2 {
            let k = self.eviction_victim();
            if let Some(old) = self.cache.remove(&k) {
                self.cache_bytes -= old.rgba.len();
            }
        }
    }

    /// Picks the cached frame to drop. Never touches the neighbourhood of the wanted frame (otherwise a block
    /// loaded for a far seek would be evicted before it can be shown and loaded again forever); keeps about a
    /// second of already-played frames so that reversing the direction is instant.
    fn eviction_victim(&self) -> u64 {
        let frame_bytes = (self.info.width as usize * self.info.height as usize * 4).max(1);
        let keep = (self.info.fps.ceil() as u64).min((CACHE_BYTES / frame_bytes / 3) as u64).max(2);
        let (cur, want) = (self.cur, self.want);
        let protected = |k: u64| k.abs_diff(want) <= keep + 16 || k.abs_diff(cur) <= keep;
        let keys: Vec<u64> = self.cache.keys().copied().collect();
        let mut order: Vec<u64> = match self.evict_dir {
            // reverse: drop what is far above the playhead first (the nearest-above frames serve a quick reversal)
            d if d < 0 => {
                let mut v: Vec<u64> = keys.iter().copied().filter(|k| *k > cur + keep).collect();
                v.extend(keys.iter().copied().filter(|k| *k <= cur + keep));
                v
            }
            // forward: drop what is far behind first, then the farthest ahead
            d if d > 0 => {
                let mut v: Vec<u64> = keys.iter().copied().filter(|k| *k + keep < cur).collect();
                v.extend(keys.iter().rev().copied().filter(|k| *k + keep >= cur));
                v
            }
            _ => {
                let mut v = keys.clone();
                v.sort_by_key(|k| std::cmp::Reverse(k.abs_diff(cur)));
                v
            }
        };
        order.retain(|k| !protected(*k));
        order.first().copied().unwrap_or_else(|| keys.iter().copied().max_by_key(|k| k.abs_diff(want)).unwrap())
    }

    fn show(&mut self, f: Arc<Frame>) {
        if self.shown.as_ref().map(|s| s.index) != Some(f.index) {
            self.shown_since = Instant::now();
        }
        self.cur = f.index;
        self.shown = Some(f);
        self.shown_gen = self.gen;
        self.error = None;
    }

    /// Jump to `lo` for looping / "play" pressed at the end (continues seamlessly if prepared).
    pub fn wrap_to_start(&mut self, lo: u64) {
        self.chunk = None;
        self.stream = match self.loop_next.take() {
            Some(s) if s.vf == self.vf && self.loop_next_at == lo => Some(s),
            _ => None,
        };
        self.cur = lo;
        self.want = lo;
        self.loading = true;
    }

    /// Jump to `hi` (reverse looping / reverse play started at the beginning).
    pub fn wrap_to_end(&mut self, hi: u64) {
        // a tail chunk prefetched for exactly this jump is kept and will land in the cache
        if !matches!(&self.chunk, Some((_, _, end, _)) if *end == hi) {
            self.chunk = None;
        }
        self.stream = None;
        self.cur = hi;
        self.want = hi;
        self.loading = true;
    }

    fn start_chunk(&mut self, tools: &Tools, wake: &Waker, end: u64, k: u64) {
        let vf = self.vf.clone();
        let start = (end + 1).saturating_sub(k);
        let count = end - start + 1;
        let (tx, rx) = channel();
        let s = Stream::open(tools, &self.path, start, self.info.fps, &vf, Some(count), count as usize + 1, wake.clone());
        let killer = s.killer();
        std::thread::spawn(move || {
            let t0 = std::time::Instant::now();
            let r = s.collect_all();
            crate::log::log(format!("[rev] blocco {start}-{end}: {} fotogrammi in {:.2}s", r.0.len(), t0.elapsed().as_secs_f64()));
            let _ = tx.send(r);
        });
        self.chunk = Some((rx, start, end, killer));
    }

    /// Makes progress towards displaying frame `want`; never blocks.
    /// `dir`: -1 = reverse playback, 1 = forward playback, 0 = paused/stepping.
    pub fn fetch(&mut self, tools: &Tools, wake: &Waker, ctx: &Context, dir: i8, range: &PlayRange) {
        self.evict_dir = dir;
        let fps = self.info.fps;
        // prepare the restart stream while the end of the range is near (seamless forward loop)
        if !range.looping || dir <= 0 {
            self.loop_next = None;
        } else if self.loop_next.is_none() {
            if let Some(hi) = range.hi.or(self.last_frame()) {
                if self.cur + fps.ceil() as u64 + 2 >= hi && hi > range.lo {
                    let vf = self.vf.clone();
                    self.loop_next = Some(Stream::open(tools, &self.path, range.lo, fps, &vf, None, 8, wake.clone()));
                    self.loop_next_at = range.lo;
                }
            }
        }
        // reverse playback: keep decoding ahead of the playhead so there is no stall at chunk borders
        let frame_bytes = (self.info.width as usize * self.info.height as usize * 4).max(1);
        let max_by_mem = (CACHE_BYTES / 2 / frame_bytes).max(4) as u64;
        if dir < 0 && self.chunk.is_none() && self.cache.contains_key(&self.want) {
            let mut edge = self.want;
            while edge > range.lo && self.cache.contains_key(&(edge - 1)) {
                edge -= 1;
            }
            let rate = (fps * range.speed.max(0.1) as f64).ceil() as u64;
            let lead = rate.max(8).min(max_by_mem / 2);
            let k = (rate * 2).clamp(16, 120).min(max_by_mem);
            if edge > range.lo && self.want - edge < lead {
                self.start_chunk(tools, wake, edge - 1, k);
            } else if edge <= range.lo && range.looping {
                // the start of the range is already cached: prepare the tail so the wrap is seamless
                if let Some(hi) = range.hi.or(self.last_frame()) {
                    if hi > range.lo && !self.cache.contains_key(&hi) && self.tail_tried != Some(hi) {
                        self.tail_tried = Some(hi);
                        self.start_chunk(tools, wake, hi, k);
                    }
                }
            }
        }
        // finished chunk load?
        if let Some((rx, c_start, c_end, _)) = &self.chunk {
            let (c_start, c_end) = (*c_start, *c_end);
            match rx.try_recv() {
                Ok((frames, err)) => {
                    if frames.is_empty() && err.as_deref().map_or(false, is_hard_error) {
                        self.error = err;
                    }
                    let got = frames.len() as u64;
                    for f in frames {
                        self.insert(f);
                    }
                    self.chunk = None;
                    // fewer frames than requested: the file ends earlier than believed
                    if got < c_end - c_start + 1 && err_is_benign(&self.error) {
                        let real_end = if got > 0 { c_start + got - 1 } else { c_start.saturating_sub(1) };
                        crate::log::log(format!("[play] fine reale del filmato: fotogramma {real_end}"));
                        self.eof = Some(real_end);
                        self.want = self.want.min(real_end);
                        if self.cur > real_end {
                            self.cur = real_end;
                        }
                    }
                }
                Err(TryRecvError::Empty) => {}
                Err(TryRecvError::Disconnected) => self.chunk = None,
            }
        }
        let want = self.want;
        if let Some(f) = self.cache.get(&want).cloned() {
            self.show(f);
            self.loading = false;
            self.hard_wait = false;
            return;
        }
        let forward = want >= self.cur;
        let fps = self.info.fps;

        // 1) pull from the running forward stream
        let vf = self.vf.clone();
        let usable = matches!(&self.stream, Some(st) if st.vf == vf && st.pos <= want && want - st.pos < 90 && st.ended.is_none());
        if usable {
            self.hard_wait = false;
            let mut pulled = Vec::new();
            let mut ended = None;
            {
                let st = self.stream.as_mut().unwrap();
                while st.pos <= want {
                    match st.try_next() {
                        Some(f) => pulled.push(f),
                        None => break,
                    }
                }
                if let Some(e) = &st.ended {
                    ended = Some((e.clone(), st.pos));
                }
            }
            let latest = pulled.last().cloned();
            for f in pulled {
                self.insert(f);
            }
            if let Some(f) = self.cache.get(&want).cloned() {
                self.show(f);
                self.loading = false;
            } else if let Some((err, pos)) = ended {
                self.stream = None;
                if err.as_deref().map_or(false, is_hard_error) {
                    self.error = err;
                }
                if pos > 0 {
                    // `pos` frames exist before the end; a zero-length tail is just the end of the file
                    let last = pos - 1;
                    self.eof = Some(last);
                    self.want = self.want.min(last);
                    if let Some(f) = self.cache.get(&self.want).cloned() {
                        self.show(f);
                    }
                }
                self.loading = false;
            } else {
                if let Some(f) = latest.filter(|_| forward) {
                    self.show(f);
                }
                self.loading = true;
            }
            return;
        }

        // from here on `want` needs a seek: hold the playback clock until data arrives
        self.hard_wait = true;
        if let Some((_, c_start, c_end, killer)) = &self.chunk {
            if *c_start <= want && want <= *c_end {
                // the block being loaded will contain it
                self.loading = true;
                return;
            }
            // stale block (the user moved elsewhere): stop wasting CPU on it
            crate::decoder::kill(killer);
            self.chunk = None;
        }

        // 2) going backwards: load a short block that ends at `want` (small = quick to start; prefetch makes it bigger)
        if !forward {
            self.start_chunk(tools, wake, want, 16.min(max_by_mem));
            self.loading = true;
            return;
        }

        // 3) (re)start the forward stream at `want`
        if self.last_restart.elapsed() < RESTART_THROTTLE {
            ctx.request_repaint_after(RESTART_THROTTLE);
            self.loading = true;
            return;
        }
        self.stream = Some(Stream::open(tools, &self.path, want, fps, &vf, None, 8, wake.clone()));
        self.last_restart = Instant::now();
        self.loading = true;
    }

    /// Uploads the shown frame to the GPU if it changed.
    pub fn update_texture(&mut self, ctx: &Context, nearest: bool) {
        let Some(f) = self.shown.clone() else { return };
        let key = (Arc::as_ptr(&f) as usize, f.index, self.shown_gen, nearest);
        if self.tex_for == Some(key) && self.tex.is_some() {
            return;
        }
        let max = ctx.input(|i| i.max_texture_side);
        if f.w > max || f.h > max {
            self.error = Some(tf("Immagine {0}×{1} oltre il limite della GPU ({2})", &[&f.w, &f.h, &max]));
            return;
        }
        let opts = if nearest { TextureOptions::NEAREST } else { TextureOptions::LINEAR };
        let img = ColorImage::from_rgba_unmultiplied([f.w, f.h], &f.rgba);
        match &mut self.tex {
            Some(t) if t.size() == [f.w, f.h] => t.set(img, opts),
            _ => self.tex = Some(ctx.load_texture("frame", img, opts)),
        }
        self.tex_for = Some(key);
    }

    // ---- AI stage -------------------------------------------------------------------------

    pub fn ai_key_matches(&self, key: &str) -> bool {
        self.shown_gen == self.gen && matches!(&self.ai, Some(a) if a.idx == self.cur && a.gen == self.gen && a.key == key)
    }

    /// Polls the running AI job, cancels it as soon as it is no longer wanted, and starts
    /// a new one for the shown frame once it is settled.
    pub fn drive_ai(&mut self, tools: &Tools, wake: &Waker, ctx: &Context, wanted: Option<(String, AiSettings)>, settled: bool) {
        let wanted_key = wanted.as_ref().map(|w| w.0.clone());
        if let Some((idx, gen, key, cancel, rx)) = &self.ai_pending {
            let stale = wanted_key.as_deref() != Some(key.as_str()) || *idx != self.cur || *gen != self.gen;
            if stale && !cancel.load(Ordering::Relaxed) {
                crate::log::log("[AI] richiesta non più valida: annullo");
                cancel.store(true, Ordering::Relaxed);
            }
            match rx.try_recv() {
                Ok(res) => {
                    let (idx, gen, key) = (*idx, *gen, key.clone());
                    let cancelled = cancel.load(Ordering::Relaxed);
                    self.ai_pending = None;
                    self.ai_busy = false;
                    if !cancelled {
                        match res {
                            Ok(f) => {
                                let max = ctx.input(|i| i.max_texture_side);
                                if f.w > max || f.h > max {
                                    self.ai_err = Some(tf("Risultato {0}x{1} oltre il limite della GPU ({2}); usa un fattore minore", &[&f.w, &f.h, &max]));
                                    self.ai_err_at = Some((idx, key));
                                } else if gen == self.gen {
                                    let img = ColorImage::from_rgba_unmultiplied([f.w, f.h], &f.rgba);
                                    let tex = ctx.load_texture("ai", img, TextureOptions::LINEAR);
                                    self.ai = Some(AiResult { tex, idx, gen, key });
                                    self.ai_err = None;
                                }
                            }
                            Err(e) => {
                                crate::log::log(format!("[AI] errore: {e}"));
                                self.ai_err = Some(e);
                                self.ai_err_at = Some((idx, key));
                            }
                        }
                    }
                }
                Err(TryRecvError::Empty) => {
                    ctx.request_repaint_after(Duration::from_millis(200));
                    return;
                }
                Err(TryRecvError::Disconnected) => {
                    self.ai_pending = None;
                    self.ai_busy = false;
                }
            }
        }
        let Some((key, ai_cfg)) = wanted else {
            self.ai_err = None;
            return;
        };
        if self.ai_err.is_some() && self.ai_err_at.as_ref() != Some(&(self.cur, key.clone())) {
            self.ai_err = None;
        }
        // the neural stage must start from the frame produced by the current filter chain, not a stale one
        if !settled || self.shown_gen != self.gen || self.ai_err.is_some() || self.ai_key_matches(&key) || self.ai_pending.is_some() {
            return;
        }
        if self.shown_since.elapsed() < Duration::from_millis(250) {
            ctx.request_repaint_after(Duration::from_millis(260));
            return;
        }
        let Some(frame) = self.shown.clone() else { return };
        let (tx, rx) = channel();
        let tools = tools.clone();
        let wake = wake.clone();
        let cancel = Arc::new(AtomicBool::new(false));
        self.ai_pending = Some((frame.index, self.gen, key, cancel.clone(), rx));
        self.ai_busy = true;
        self.ai_started = Some(Instant::now());
        std::thread::spawn(move || {
            let _ = tx.send(nn::enhance_frame(&tools, &frame, &ai_cfg, &cancel));
            wake();
        });
    }
}

/// ffmpeg prints benign notes when a seek lands past the end ("Output file is empty ..."); only real failures are errors.
fn is_hard_error(msg: &str) -> bool {
    ["rror", "nvalid", "No such", "not found", "nrecognized", "nknown", "failed", "Cannot"].iter().any(|k| msg.contains(k))
}

fn err_is_benign(e: &Option<String>) -> bool {
    e.is_none()
}
