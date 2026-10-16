//! Playback clock: decides which frame should be on screen and when, independent of the UI.
use crate::video::{PlayRange, Video};
use std::time::Instant;

pub struct Playback {
    pub playing: bool,
    pub reverse: bool,
    pub looping: bool,
    pub speed: f32,
    pub play_pos: f64,
    pub last_tick: Instant,
    wait_since: Option<Instant>,
}

impl Playback {
    pub fn new() -> Playback {
        Playback { playing: false, reverse: false, looping: false, speed: 1.0, play_pos: 0.0, last_tick: Instant::now(), wait_since: None }
    }

    /// -1 reverse playback, 1 forward playback, 0 paused (what `Video::fetch` needs to know).
    pub fn dir(&self) -> i8 {
        if !self.playing {
            0
        } else if self.reverse {
            -1
        } else {
            1
        }
    }

    pub fn range(&self, cue: Option<(u64, u64)>) -> PlayRange {
        match cue {
            Some((a, b)) => PlayRange { lo: a, hi: Some(b), looping: self.looping, speed: self.speed },
            None => PlayRange { lo: 0, hi: None, looping: self.looping, speed: self.speed },
        }
    }

    /// Play / pause in a direction. Starting outside the range or at its end begins from the proper edge.
    pub fn toggle(&mut self, v: &mut Video, reverse: bool, range: &PlayRange) {
        if self.playing && self.reverse == reverse {
            self.playing = false;
            return;
        }
        let hi = range.hi.or(v.eof);
        let outside = v.cur < range.lo || range.hi.map_or(false, |h| v.cur > h);
        if !reverse && (outside || hi.map_or(false, |h| v.cur >= h)) {
            v.wrap_to_start(range.lo);
        } else if reverse && (outside || v.cur <= range.lo) {
            if let Some(h) = hi.or(v.last_frame()) {
                v.wrap_to_end(h);
            }
        }
        self.play_pos = v.cur as f64;
        self.playing = true;
        self.reverse = reverse;
        self.last_tick = Instant::now();
        self.wait_since = None;
    }

    /// Moves the playhead; call once per UI frame.
    pub fn advance(&mut self, v: &mut Video, now: Instant, range: &PlayRange) {
        let dt = now.duration_since(self.last_tick).as_secs_f64().min(0.1);
        self.last_tick = now;
        if !self.playing {
            return;
        }
        // a seek / direction change is still being served: do not let the clock run away from the decoder
        if v.hard_wait {
            let since = *self.wait_since.get_or_insert(now);
            if now.duration_since(since).as_secs() >= 6 {
                crate::log::log(format!("[play] stallo (richiesto {}, mostrato {}): fermo", v.want, v.cur));
                self.playing = false;
                self.wait_since = None;
            }
            return;
        }
        let lo = range.lo as f64;
        if self.reverse {
            // reverse playback waits for the requested frame instead of skipping ahead of the decoder
            if v.cur != v.want {
                let since = *self.wait_since.get_or_insert(now);
                if now.duration_since(since).as_secs() >= 6 {
                    crate::log::log(format!("[play] stallo in riproduzione inversa (richiesto {}, mostrato {}): fermo", v.want, v.cur));
                    self.playing = false;
                    self.wait_since = None;
                }
                return;
            }
            self.wait_since = None;
            self.play_pos -= dt * v.info.fps * self.speed as f64;
            if self.play_pos < lo {
                match (self.looping, range.hi.or(v.last_frame())) {
                    (true, Some(h)) => {
                        self.play_pos = h as f64;
                        v.wrap_to_end(h);
                    }
                    _ => {
                        self.play_pos = lo;
                        self.playing = false;
                    }
                }
            }
        } else {
            self.play_pos += dt * v.info.fps * self.speed as f64;
            if let Some(h) = range.hi.or(v.eof) {
                if self.play_pos > h as f64 {
                    if self.looping {
                        self.play_pos = lo;
                        v.wrap_to_start(range.lo);
                    } else {
                        self.play_pos = h as f64;
                        self.playing = false;
                    }
                }
            }
        }
        v.want = v.clamp_want(self.play_pos.floor() as i64);
    }
}
