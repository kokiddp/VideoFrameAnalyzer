//! Pipeline tests. Require `VFA_FFMPEG` (path to ffmpeg); skipped otherwise.
use crate::decoder::Stream;
use crate::export::{self, Format};
use crate::{probe, tools};
use std::sync::Arc;

fn setup() -> Option<(tools::Tools, std::path::PathBuf)> {
    std::env::var_os("VFA_FFMPEG")?;
    let t = tools::prepare().ok()?;
    let p = t.temp.join("selftest.mp4");
    if !p.exists() {
        let st = tools::command(&t.ffmpeg)
            .args(["-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30000/1001:duration=4", "-c:v", "libx264", "-g", "25", "-pix_fmt", "yuv420p"])
            .arg(&p)
            .status()
            .ok()?;
        assert!(st.success());
    }
    Some((t, p))
}

#[test]
fn probe_and_seek_accuracy() {
    let Some((t, p)) = setup() else { return };
    let info = probe::probe(&t, &p).unwrap();
    println!("{info:?}");
    assert_eq!((info.width, info.height), (320, 240));
    assert!((info.fps - 30000.0 / 1001.0).abs() < 1e-6);
    assert_eq!(info.frames, Some(120));

    let wake: crate::decoder::Waker = Arc::new(|| {});
    // reference: sequential decode of all frames
    let all = Stream::open(&t, &p, 0, info.fps, "", None, 4, wake.clone()).collect_all().0;
    assert_eq!(all.len(), 120, "frame count");
    for start in [1u64, 7, 24, 25, 26, 77, 119] {
        let (fr, err) = Stream::open(&t, &p, start, info.fps, "", Some(3), 8, wake.clone()).collect_all();
        assert!(err.is_none(), "{err:?}");
        assert_eq!(fr[0].index, start);
        assert_eq!(fr[0].rgba, all[start as usize].rgba, "seek to frame {start} returned a different picture");
    }
    // filters change the size: output size must be detected from ffmpeg's output
    let (fr, _) = Stream::open(&t, &p, 5, info.fps, "scale=640:480:flags=lanczos,unsharp=lx=5:ly=5:la=1.0:cx=5:cy=5:ca=0.5,cas=strength=0.5", Some(1), 4, wake).collect_all();
    assert_eq!((fr[0].w, fr[0].h), (640, 480));
}

#[test]
fn export_matches_stream() {
    let Some((t, p)) = setup() else { return };
    let info = probe::probe(&t, &p).unwrap();
    let wake: crate::decoder::Waker = Arc::new(|| {});
    let dir = t.temp.join("selftest_out");
    let _ = std::fs::remove_dir_all(&dir);
    for fmt in Format::ALL {
        let job = export::Job { video: p.clone(), out_dir: dir.clone(), start: 49, count: 2, fps: info.fps, vf: String::new(), format: fmt, high_bit: false, ai: None };
        let files = export::run(&t, &job, &|_, _, _| {}, &std::sync::atomic::AtomicBool::new(false)).unwrap();
        assert_eq!(files.len(), 2, "{}", fmt.label());
    }
    // PNG pixel check against the decoder's frame 50 (second exported)
    let png = image::open(dir.join("selftest_frame_000050.png")).unwrap().to_rgba8();
    // (chroma upsampling differs slightly between preview and export, so compare against neighbours)
    let (fr, _) = Stream::open(&t, &p, 49, info.fps, "", Some(3), 4, wake).collect_all();
    let d = |f: &crate::decoder::Frame| -> f64 {
        let diff: u64 = png.as_raw().iter().zip(f.rgba.iter()).map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs() as u64).sum();
        diff as f64 / png.as_raw().len() as f64
    };
    let (d49, d50, d51) = (d(&fr[0]), d(&fr[1]), d(&fr[2]));
    println!("export(50) vs preview 49/50/51: {d49:.2} {d50:.2} {d51:.2}");
    assert!(d50 < d49 && d50 < d51 && d50 < 4.0, "exported frame is not frame 50");
}

#[test]
fn esrgan_runs() {
    let Some((t, p)) = setup() else { return };
    if t.realesrgan.is_none() {
        return;
    }
    let wake: crate::decoder::Waker = Arc::new(|| {});
    for (vf, model) in [("", "realesr-animevideov3-x2"), ("scale=1920:1080", "realesr-animevideov3-x2"), ("", "realesrgan-x4plus")] {
        let (fr, _) = Stream::open(&t, &p, 10, 30.0, vf, Some(1), 2, wake.clone()).collect_all();
        let t0 = std::time::Instant::now();
        let r = { let mut ai = crate::nn::default_settings(&t); ai.model = model.to_string(); ai.dev_esrgan = 0; crate::nn::enhance_frame(&t, &fr[0], &ai, &crate::nn::NEVER) };
        println!("{model} {}x{} -> {:?} in {:.1}s", fr[0].w, fr[0].h, r.as_ref().map(|f| (f.w, f.h)), t0.elapsed().as_secs_f64());
        r.unwrap();
    }
    for l in crate::log::snapshot().iter().rev().take(8) {
        println!("{l}");
    }
}

#[test]
fn esrgan_cancel() {
    let Some((t, p)) = setup() else { return };
    if t.realesrgan.is_none() {
        return;
    }
    let wake: crate::decoder::Waker = Arc::new(|| {});
    let (fr, _) = Stream::open(&t, &p, 10, 30.0, "scale=1920:1080", Some(1), 2, wake).collect_all();
    let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let c2 = cancel.clone();
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(2500));
        c2.store(true, std::sync::atomic::Ordering::Relaxed);
    });
    let t0 = std::time::Instant::now();
    let r = { let mut ai = crate::nn::default_settings(&t); ai.model = "realesrgan-x4plus".into(); crate::nn::enhance_frame(&t, &fr[0], &ai, &cancel) };
    println!("cancelled after {:.1}s: {:?}", t0.elapsed().as_secs_f64(), r.as_ref().err());
    assert!(r.is_err() && t0.elapsed().as_secs_f64() < 6.0);
}

#[test]
fn benchmark_picks_fastest() {
    let Some((t, _)) = setup() else { return };
    let bench = crate::nn::benchmark(&t, true);
    println!("{bench:#?}");
    for (_, devs) in &bench {
        assert!(!devs.is_empty());
        assert!(devs.windows(2).all(|w| w[0].secs <= w[1].secs));
    }
}

#[test]
fn waifu2x_gpu_and_cpu_agree() {
    let Some((t, p)) = setup() else { return };
    if t.waifu2x.is_none() || t.realesrgan_cpu.is_none() {
        return;
    }
    let wake: crate::decoder::Waker = Arc::new(|| {});
    let (fr, _) = Stream::open(&t, &p, 10, 30.0, "", Some(1), 2, wake).collect_all();
    let mut ai = crate::nn::default_settings(&t);
    ai.engine = crate::filters::Engine::Waifu2x;
    ai.w2x_model = "upconv_7_photo".into();
    let mut results = Vec::new();
    for dev in [0, -1] {
        ai.dev_w2x = dev;
        let t0 = std::time::Instant::now();
        let r = crate::nn::enhance_frame(&t, &fr[0], &ai, &crate::nn::NEVER).unwrap();
        println!("waifu2x device {dev}: {}x{} in {:.2}s", r.w, r.h, t0.elapsed().as_secs_f64());
        assert_eq!((r.w, r.h), (fr[0].w * 2, fr[0].h * 2));
        results.push(r);
    }
    let diff: u64 = results[0].rgba.iter().zip(&results[1].rgba).map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs() as u64).sum();
    let mean = diff as f64 / results[0].rgba.len() as f64;
    println!("mean abs diff gpu vs cpu: {mean:.3}");
    assert!(mean < 1.5);
}

/// Drives `Video::fetch` headlessly until the wanted frame is shown.
fn settle(v: &mut crate::video::Video, t: &tools::Tools, ctx: &eframe::egui::Context, wake: &crate::decoder::Waker, looping: bool) {
    let t0 = std::time::Instant::now();
    loop {
        v.fetch(t, wake, ctx, 0, &crate::video::PlayRange { lo: 0, hi: None, looping, speed: 1.0 });
        if v.cur == v.want && v.shown.as_ref().map(|s| s.index) == Some(v.want) {
            return;
        }
        assert!(t0.elapsed().as_secs() < 8, "timeout: want {} cur {} eof {:?} err {:?}", v.want, v.cur, v.eof, v.error);
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
}

#[test]
fn end_of_video_behaviour() {
    let Some((t, p)) = setup() else { return };
    let info = probe::probe(&t, &p).unwrap();
    let ctx = eframe::egui::Context::default();
    let wake: crate::decoder::Waker = Arc::new(|| {});
    let mut v = crate::video::Video::new(p.clone(), info.clone(), String::new());

    // forward through the whole file, then past the end
    for w in 0..120u64 {
        v.want = w;
        settle(&mut v, &t, &ctx, &wake, false);
    }
    v.want = 125;
    let t0 = std::time::Instant::now();
    while v.eof.is_none() && t0.elapsed().as_secs() < 8 {
        v.fetch(&t, &wake, &ctx, 0, &crate::video::PlayRange { lo: 0, hi: None, looping: false, speed: 1.0 });
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert_eq!(v.eof, Some(119), "end of file detected");
    assert!(v.error.is_none(), "no error at EOF: {:?}", v.error);
    settle(&mut v, &t, &ctx, &wake, false);
    assert_eq!(v.cur, 119);

    // manual stepping back and forth at the end
    for w in [118u64, 117, 118, 119, 100, 101, 119, 0, 1, 2, 119] {
        v.want = v.clamp_want(w as i64);
        settle(&mut v, &t, &ctx, &wake, false);
        assert_eq!(v.cur, w);
    }
    assert!(v.error.is_none(), "{:?}", v.error);

    // seamless loop: end -> start -> a few frames, with the prefetch stream
    v.want = 118;
    settle(&mut v, &t, &ctx, &wake, true);
    v.want = 119;
    settle(&mut v, &t, &ctx, &wake, true);
    v.wrap_to_start(0);
    let t1 = std::time::Instant::now();
    for w in 0..10u64 {
        v.want = w;
        settle(&mut v, &t, &ctx, &wake, true);
    }
    println!("loop restart + 10 frames took {:?}", t1.elapsed());

    // fresh video: jump straight to the (estimated) end, past it, and back
    let mut v = crate::video::Video::new(p, info, String::new());
    for w in [119u64, 110, 109, 108, 130] {
        v.want = v.clamp_want(w as i64);
        let t0 = std::time::Instant::now();
        while t0.elapsed().as_secs() < 8 {
            v.fetch(&t, &wake, &ctx, 0, &crate::video::PlayRange { lo: 0, hi: None, looping: false, speed: 1.0 });
            if v.cur == v.want && v.shown.as_ref().map(|s| s.index) == Some(v.want) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(v.cur, v.want, "jump to {w}");
    }
    assert_eq!(v.cur, 119);
}

#[test]
fn reverse_playback_is_smooth() {
    let Some((t, _)) = setup() else { return };
    let p = std::env::var_os("VFA_REVERSE_VIDEO").map(std::path::PathBuf::from).unwrap_or_else(|| t.temp.join("selftest_long.mp4"));
    if !p.exists() {
        let st = tools::command(&t.ffmpeg)
            .args(["-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=30:duration=20", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-g", "250", "-pix_fmt", "yuv420p"])
            .arg(&p)
            .status()
            .unwrap();
        assert!(st.success());
    }
    let info = probe::probe(&t, &p).unwrap();
    let ctx = eframe::egui::Context::default();
    let wake: crate::decoder::Waker = Arc::new(|| {});
    let mut v = crate::video::Video::new(p, info.clone(), String::new());
    let last = info.frames.unwrap() - 1;
    v.want = last;
    let range = crate::video::PlayRange { lo: 0, hi: None, looping: false, speed: 1.0 };
    // settle on the last frame first
    let t0 = std::time::Instant::now();
    while v.cur != v.want || v.shown.is_none() {
        v.fetch(&t, &wake, &ctx, 0, &range);
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert!(t0.elapsed().as_secs() < 10);
    }
    // reverse play at 1x with the same "wait for data" clock as the app
    let mut pos = last as f64;
    let mut last_tick = std::time::Instant::now();
    let (mut waited, mut max_stall, mut stall_start) = (0.0f64, 0.0f64, None::<std::time::Instant>);
    let start = std::time::Instant::now();
    let mut shown_frames = 0u32;
    while pos > 30.0 && start.elapsed().as_secs() < 40 {
        let now = std::time::Instant::now();
        let dt = now.duration_since(last_tick).as_secs_f64().min(0.1);
        last_tick = now;
        if v.cur == v.want {
            if let Some(s) = stall_start.take() {
                let d = s.elapsed().as_secs_f64();
                waited += d;
                max_stall = max_stall.max(d);
            }
            pos -= dt * info.fps;
            let w = pos.floor().max(0.0) as u64;
            if w != v.want {
                shown_frames += 1;
            }
            v.want = w;
        } else if stall_start.is_none() {
            stall_start = Some(now);
        }
        v.fetch(&t, &wake, &ctx, -1, &range);
        std::thread::sleep(std::time::Duration::from_millis(4));
    }
    println!("reverse: {} frames shown in {:.1}s, waited {:.2}s total, longest stall {:.2}s", shown_frames, start.elapsed().as_secs_f64(), waited, max_stall);
    assert!(max_stall < 1.5, "reverse playback stalls too long");
    assert!(waited < start.elapsed().as_secs_f64() * 0.5, "reverse playback spends most time waiting");
}

#[test]
fn export_range_reports_progress_and_cancels() {
    let Some((t, p)) = setup() else { return };
    let info = probe::probe(&t, &p).unwrap();
    let dir = t.temp.join("selftest_range");
    let _ = std::fs::remove_dir_all(&dir);
    let job = export::Job { video: p, out_dir: dir.clone(), start: 10, count: 60, fps: info.fps, vf: String::new(), format: Format::Png, high_bit: false, ai: None };
    let seen = std::sync::Mutex::new(Vec::<f32>::new());
    let files = export::run(&t, &job, &|f, _, _| seen.lock().unwrap().push(f), &std::sync::atomic::AtomicBool::new(false)).unwrap();
    assert_eq!(files.len(), 60);
    let v = seen.lock().unwrap().clone();
    println!("progress samples: {} first {:?} last {:?}", v.len(), v.first(), v.last());
    assert!(v.windows(2).all(|w| w[0] <= w[1]) && *v.last().unwrap() == 1.0);
    // cancel immediately
    let cancel = std::sync::atomic::AtomicBool::new(true);
    let r = export::run(&t, &job, &|_, _, _| {}, &cancel);
    assert!(matches!(r, Err(ref e) if e == "Annullato"), "{r:?}");
}

#[test]
fn reverse_loop_wraps_seamlessly() {
    let Some((t, _)) = setup() else { return };
    let p = std::env::var_os("VFA_REVERSE_VIDEO").map(std::path::PathBuf::from).unwrap_or_else(|| t.temp.join("selftest_long.mp4"));
    if !p.exists() {
        return; // created by reverse_playback_is_smooth
    }
    let info = probe::probe(&t, &p).unwrap();
    let ctx = eframe::egui::Context::default();
    let wake: crate::decoder::Waker = Arc::new(|| {});
    let mut v = crate::video::Video::new(p, info.clone(), String::new());
    // CUE-like range near the start so wraps happen quickly
    let (lo, hi) = (0u64, 150u64);
    let range = crate::video::PlayRange { lo, hi: Some(hi), looping: true, speed: 1.0 };
    v.want = 40;
    let t0 = std::time::Instant::now();
    while v.cur != v.want || v.shown.is_none() {
        v.fetch(&t, &wake, &ctx, 0, &range);
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert!(t0.elapsed().as_secs() < 10);
    }
    let mut pos = 40.0f64;
    let mut last_tick = std::time::Instant::now();
    let (mut wraps, mut max_stall, mut stall_start) = (0, 0.0f64, None::<std::time::Instant>);
    let start = std::time::Instant::now();
    while wraps < 3 && start.elapsed().as_secs() < 40 {
        let now = std::time::Instant::now();
        let dt = now.duration_since(last_tick).as_secs_f64().min(0.1);
        last_tick = now;
        if v.cur == v.want {
            if let Some(s) = stall_start.take() {
                max_stall = max_stall.max(s.elapsed().as_secs_f64());
            }
            pos -= dt * info.fps;
            if pos < lo as f64 {
                pos = hi as f64;
                v.wrap_to_end(hi);
                wraps += 1;
                if wraps == 1 {
                    max_stall = 0.0;
                }
                println!("wrap {wraps} at {:.1}s", start.elapsed().as_secs_f64());
            }
            v.want = pos.floor().max(0.0) as u64;
        } else if stall_start.is_none() {
            stall_start = Some(now);
        }
        v.fetch(&t, &wake, &ctx, -1, &range);
        std::thread::sleep(std::time::Duration::from_millis(4));
    }
    println!("reverse loop: {wraps} wraps in {:.1}s, longest stall {:.2}s", start.elapsed().as_secs_f64(), max_stall);
    assert!(wraps >= 3);
    assert!(max_stall < 0.5, "reverse loop stalls: {max_stall}");
}

/// Reverse loop over the whole file when the container duration over-estimates the frame count.
#[test]
fn reverse_loop_with_wrong_frame_estimate() {
    let Some((t, p)) = setup() else { return };
    let mut info = probe::probe(&t, &p).unwrap();
    info.frames = Some(info.frames.unwrap() + 7); // 127 believed, 120 real
    let ctx = eframe::egui::Context::default();
    let wake: crate::decoder::Waker = Arc::new(|| {});
    let mut v = crate::video::Video::new(p, info.clone(), String::new());
    let range = crate::video::PlayRange { lo: 0, hi: None, looping: true, speed: 1.0 };
    v.want = 30;
    let t0 = std::time::Instant::now();
    while v.cur != v.want || v.shown.is_none() {
        v.fetch(&t, &wake, &ctx, 0, &range);
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert!(t0.elapsed().as_secs() < 10);
    }
    let mut pos = 30.0f64;
    let mut last_tick = std::time::Instant::now();
    let (mut wraps, mut stall_start, mut max_stall) = (0, None::<std::time::Instant>, 0.0f64);
    let start = std::time::Instant::now();
    let mut shown = std::collections::BTreeSet::new();
    while wraps < 3 && start.elapsed().as_secs() < 30 {
        let now = std::time::Instant::now();
        let dt = now.duration_since(last_tick).as_secs_f64().min(0.1);
        last_tick = now;
        if v.cur == v.want {
            if let Some(s) = stall_start.take() {
                max_stall = max_stall.max(s.elapsed().as_secs_f64());
            }
            pos -= dt * info.fps;
            if pos < 0.0 {
                let h = v.last_frame().unwrap();
                pos = h as f64;
                v.wrap_to_end(h);
                wraps += 1;
                println!("wrap {wraps} at {:.1}s, end believed {h}", start.elapsed().as_secs_f64());
            }
            v.want = v.clamp_want(pos.floor().max(0.0) as i64);
        } else if stall_start.is_none() {
            stall_start = Some(now);
        } else if stall_start.unwrap().elapsed().as_secs() > 5 {
            panic!("stuck: want {} cur {} eof {:?} err {:?}", v.want, v.cur, v.eof, v.error);
        }
        v.fetch(&t, &wake, &ctx, -1, &range);
        if let Some(s) = &v.shown {
            shown.insert(s.index);
        }
        std::thread::sleep(std::time::Duration::from_millis(4));
    }
    println!("{wraps} wraps in {:.1}s, longest stall {:.2}s, eof {:?}, shown {} distinct frames", start.elapsed().as_secs_f64(), max_stall, v.eof, shown.len());
    assert!(wraps >= 3, "reverse loop got stuck");
    assert!(shown.contains(&119) && shown.contains(&0));
}

#[test]
fn benchmark_window_engine() {
    let Some((t, _)) = setup() else { return };
    if t.realesrgan_cpu.is_none() {
        return;
    }
    use crate::bench::{Msg, Plan, Spec};
    let mut a = crate::nn::default_settings(&t);
    a.engine = crate::filters::Engine::Esrgan;
    a.model = "realesr-animevideov3-x2".into();
    let mut b = crate::nn::default_settings(&t);
    b.engine = crate::filters::Engine::Esrgan;
    b.model = "realesrgan-x4plus".into();
    let plan = Plan {
        specs: vec![
            Spec { label: "animevideov3-x2".into(), ai: a, devices: vec![-1, 0] },
            Spec { label: "x4plus (timeout)".into(), ai: b, devices: vec![-1] },
        ],
        heights: vec![180, 360],
        timeout_secs: 2,
    };
    let (tx, rx) = std::sync::mpsc::channel();
    let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
    crate::bench::run(&t, &plan, None, &cancel, &tx, &|| {});
    let mut cells = Vec::new();
    let mut finished = false;
    while let Ok(m) = rx.try_recv() {
        match m {
            Msg::Cell { spec, res, dev, cell } => cells.push((spec, res, dev, cell)),
            Msg::Finished => finished = true,
        }
    }
    for c in &cells {
        println!("{:?}", c);
    }
    assert!(finished);
    assert_eq!(cells.len(), plan.total());
    // animevideov3 completes everywhere; x4plus on CPU times out at the first size and the next one is skipped
    // (a software Vulkan device such as llvmpipe may legitimately exceed the short 2 s limit — and then the larger size is
    // skipped — the CPU may not)
    assert!(cells.iter().filter(|c| c.0 == 0 && c.2 < 0).all(|c| c.3.secs.is_some()));
    assert!(cells.iter().filter(|c| c.0 == 0 && c.2 >= 0).all(|c| c.3.secs.is_some() || c.3.note.starts_with('>') || c.3.note == "saltato"));
    let x4: Vec<_> = cells.iter().filter(|c| c.0 == 1).collect();
    assert!(x4[0].3.note.starts_with('>') && x4[1].3.note == "saltato", "{x4:?}");
}

/// Random playback session (seeks, speed and direction changes, pauses) against the real clock + fetch logic.
/// Reports every period where the picture froze although playback was running.
#[test]
fn playback_stays_fluid_under_random_operations() {
    let Some((t, _)) = setup() else { return };
    let p = std::env::var_os("VFA_REVERSE_VIDEO").map(std::path::PathBuf::from).unwrap_or_else(|| t.temp.join("selftest_long.mp4"));
    if !p.exists() {
        return;
    }
    use crate::player::Playback;
    let info = probe::probe(&t, &p).unwrap();
    let ctx = eframe::egui::Context::default();
    let wake: crate::decoder::Waker = Arc::new(|| {});
    let mut v = crate::video::Video::new(p, info.clone(), String::new());
    let mut pb = Playback::new();
    pb.looping = true;
    let seed = std::env::var("VFA_SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(7u64);
    let mut rng = seed;
    let mut rnd = move |n: u64| {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (rng >> 33) % n
    };
    let last = info.frames.unwrap() - 1;
    let range = |pb: &Playback| pb.range(None);
    let t_start = std::time::Instant::now();
    let total = std::time::Duration::from_secs(std::env::var("VFA_SECS").ok().and_then(|s| s.parse().ok()).unwrap_or(50));
    let mut next_op = std::time::Instant::now();
    let mut last_op = String::from("start");
    let (mut frozen_since, mut last_cur, mut frozen_total, mut worst) = (None::<std::time::Instant>, u64::MAX, 0.0f64, 0.0f64);
    let mut resume_at: Option<std::time::Instant> = None;
    let mut play_secs = 0.0f64;
    let mut tick = std::time::Instant::now();
    let (mut changes, mut sample_at) = (0u32, std::time::Instant::now() + std::time::Duration::from_secs(1));
    let mut seen_cur = u64::MAX;
    while t_start.elapsed() < total {
        let now = std::time::Instant::now();
        if let Some(r) = resume_at {
            if now >= r {
                resume_at = None;
                let rg = range(&pb);
                pb.toggle(&mut v, false, &rg);
                last_op = "riprendi".into();
            }
        }
        if now >= next_op {
            next_op = now + std::time::Duration::from_millis(1200 + rnd(2500));
            let rg = range(&pb);
            match rnd(5) {
                0 => {
                    let f = rnd(last);
                    pb.playing = false;
                    v.want = f;
                    pb.play_pos = f as f64;
                    resume_at = Some(now + std::time::Duration::from_millis(300));
                    last_op = format!("salto a {f} + pausa");
                }
                1 => {
                    pb.speed = [0.25, 0.5, 1.0, 1.0, 2.0, 3.0][rnd(6) as usize];
                    last_op = format!("velocita {}", pb.speed);
                }
                2 => {
                    let rev = !pb.reverse;
                    pb.playing = false;
                    pb.toggle(&mut v, rev, &rg);
                    last_op = format!("verso {}", if rev { "indietro" } else { "avanti" });
                }
                3 => {
                    pb.playing = false;
                    resume_at = Some(now + std::time::Duration::from_millis(600));
                    last_op = "pausa breve".into();
                }
                _ => {
                    let f = rnd(last);
                    v.want = f;
                    pb.play_pos = f as f64;
                    last_op = format!("salto a {f} (in play)");
                }
            }
        }
        let rg = range(&pb);
        pb.advance(&mut v, now, &rg);
        v.fetch(&t, &wake, &ctx, pb.dir(), &rg);
        if pb.playing {
            let dt = now.duration_since(tick).as_secs_f64();
            play_secs += dt;
            if v.cur != last_cur {
                if let Some(s) = frozen_since.take() {
                    let d = s.elapsed().as_secs_f64();
                    if d > 0.4 {
                        println!("  freeze {:.2}s dopo '{}' (dir {}, vel {}, cur {})", d, last_op, pb.dir(), pb.speed, v.cur);
                    }
                    worst = worst.max(d);
                }
                last_cur = v.cur;
            } else if frozen_since.is_none() {
                frozen_since = Some(now);
            }
            if frozen_since.map_or(false, |s| s.elapsed().as_secs_f64() > 0.0) {
                frozen_total += dt;
            }
        } else {
            frozen_since = None;
            last_cur = v.cur;
        }
        if v.cur != seen_cur {
            seen_cur = v.cur;
            changes += 1;
        }
        if now >= sample_at {
            sample_at = now + std::time::Duration::from_secs(1);
            if pb.playing {
                println!("  [{:4.0}s] dir {:2} vel {:4} cambi/s {:3} (atteso {:3.0}) ritardo {:4.0} fotogrammi  cur {} ({})", t_start.elapsed().as_secs_f64(), pb.dir(), pb.speed, changes, info.fps * pb.speed as f64, (pb.play_pos - v.cur as f64).abs(), v.cur, last_op);
            }
            changes = 0;
        }
        tick = now;
        std::thread::sleep(std::time::Duration::from_millis(4));
    }
    for l in crate::log::snapshot().iter().filter(|l| l.contains("[rev]") || l.contains("[play]")) {
        println!("LOG {l}");
    }
    println!("playback {:.1}s, frozen {:.1}s ({:.0}%), worst freeze {:.2}s", play_secs, frozen_total, 100.0 * frozen_total / play_secs.max(0.1), worst);
    assert!(worst < 2.0, "freeze of {worst}s");
}

#[test]
fn filter_generation_tracks_applied_chain() {
    let Some((t, p)) = setup() else { return };
    let info = probe::probe(&t, &p).unwrap();
    let ctx = eframe::egui::Context::default();
    let wake: crate::decoder::Waker = Arc::new(|| {});
    let range = crate::video::PlayRange { lo: 0, hi: None, looping: false, speed: 1.0 };
    let mut v = crate::video::Video::new(p, info, String::new());
    v.want = 20;
    let t0 = std::time::Instant::now();
    while v.shown_gen != v.gen || v.shown.as_ref().map(|s| s.index) != Some(20) {
        v.fetch(&t, &wake, &ctx, 0, &range);
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert!(t0.elapsed().as_secs() < 8);
    }
    // a valid chain: not applied until a frame of the new generation is displayed, and it keeps the position
    v.set_filters("unsharp=lx=5:ly=5:la=1.0:cx=5:cy=5:ca=0.5,scale=iw*2:ih*2".into());
    assert_ne!(v.shown_gen, v.gen);
    while v.shown_gen != v.gen {
        v.fetch(&t, &wake, &ctx, 0, &range);
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert!(t0.elapsed().as_secs() < 16);
    }
    assert_eq!(v.cur, 20);
    assert_eq!(v.shown.as_ref().map(|s| (s.w, s.h)), Some((640, 480)));
    // a broken chain never becomes visible but reports an error so the UI can unlock
    v.set_filters("nosuchfilter=1".into());
    let t1 = std::time::Instant::now();
    while v.error.is_none() && t1.elapsed().as_secs() < 8 {
        v.fetch(&t, &wake, &ctx, 0, &range);
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert!(v.error.is_some() && v.shown_gen != v.gen, "{:?}", v.error);
}

/// The texture on screen must always be the frame that `shown` holds, also when the same frame index is
/// re-decoded with a different filter chain.
#[test]
fn texture_follows_filter_change_on_same_frame() {
    let Some((t, p)) = setup() else { return };
    let info = probe::probe(&t, &p).unwrap();
    let ctx = eframe::egui::Context::default();
    let wake: crate::decoder::Waker = Arc::new(|| {});
    let range = crate::video::PlayRange { lo: 0, hi: None, looping: false, speed: 1.0 };
    let mut v = crate::video::Video::new(p, info, String::new());
    let pump = |v: &mut crate::video::Video, what: &str| {
        let t0 = std::time::Instant::now();
        while v.shown_gen != v.gen || v.shown.as_ref().map(|s| s.index) != Some(v.want) {
            v.fetch(&t, &wake, &ctx, 0, &range);
            v.update_texture(&ctx, false);
            std::thread::sleep(std::time::Duration::from_millis(5));
            assert!(t0.elapsed().as_secs() < 8, "{what}");
        }
        v.update_texture(&ctx, false);
    };
    v.want = 20;
    pump(&mut v, "plain");
    assert_eq!(v.tex.as_ref().unwrap().size(), [320, 240]);
    for (chain, size) in [("scale=iw*2:ih*2", [640, 480]), ("", [320, 240]), ("scale=iw*2:ih*2", [640, 480]), ("", [320, 240])] {
        v.set_filters(chain.into());
        // the UI redraws while the old frame is still on screen (this is what used to poison the texture cache)
        v.update_texture(&ctx, false);
        pump(&mut v, chain);
        let shown = v.shown.as_ref().unwrap();
        assert_eq!([shown.w, shown.h], size, "frame for chain '{chain}'");
        assert_eq!(v.tex.as_ref().unwrap().size(), size, "texture for chain '{chain}'");
    }
}

#[test]
fn export_applies_filter_chain_only_when_given() {
    let Some((t, p)) = setup() else { return };
    let info = probe::probe(&t, &p).unwrap();
    let dir = t.temp.join("selftest_filters");
    let _ = std::fs::remove_dir_all(&dir);
    for (vf, size) in [("", (320u32, 240u32)), ("scale=iw*2:ih*2", (640, 480))] {
        let job = export::Job { video: p.clone(), out_dir: dir.join(if vf.is_empty() { "off" } else { "on" }), start: 12, count: 1, fps: info.fps, vf: vf.into(), format: Format::Png, high_bit: false, ai: None };
        let files = export::run(&t, &job, &|_, _, _| {}, &std::sync::atomic::AtomicBool::new(false)).unwrap();
        let img = image::open(&files[0]).unwrap();
        assert_eq!((img.width(), img.height()), size, "chain '{vf}'");
    }
}

/// Width of the right column over many frames while every slider keeps changing. `fixed_layout = false`
/// reproduces the old widths (slider wider than the card) to prove that the test detects the runaway growth.
fn column_widths(fixed_layout: bool) -> Vec<f32> {
    use eframe::egui;
    let ctx = egui::Context::default();
    let mut filters = crate::filters::default_filters();
    for f in &mut filters {
        f.enabled = true;
    }
    let mut widths = Vec::new();
    for frame in 0..45 {
        for f in &mut filters {
            for p in &mut f.params {
                if let crate::filters::Kind::Float { min, max } = p.kind {
                    p.value = min + (max - min) * ((frame * 3 % 11) as f32 / 11.0);
                }
            }
        }
        let raw = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1360.0, 820.0))), ..Default::default() };
        let mut w = 0.0;
        let _ = ctx.run_ui(raw, |ui| {
            let r = egui::Panel::right("side").resizable(true).default_size(380.0).size_range(300.0..=760.0).show(ui, |ui| {
                let cw = (ui.available_width() - 30.0).max(240.0);
                let slider_w = if fixed_layout { crate::widgets::slider_width(cw) } else { (cw - 70.0 - 70.0).clamp(70.0, 420.0) + 120.0 };
                let area = if fixed_layout { egui::ScrollArea::both() } else { egui::ScrollArea::vertical() };
                area.auto_shrink([false, false]).show(ui, |ui| {
                    for f in &mut filters {
                        crate::widgets::filter_card(ui, f, cw, slider_w);
                    }
                });
            });
            w = r.response.rect.width();
        });
        widths.push(w);
    }
    widths
}

#[test]
fn right_column_never_changes_width() {
    let good = column_widths(true);
    let (lo, hi) = (good.iter().cloned().fold(f32::MAX, f32::min), good.iter().cloned().fold(0.0, f32::max));
    println!("fixed layout: width {lo:.1} .. {hi:.1} over {} frames", good.len());
    assert!(hi - lo < 0.5, "column width changed: {lo} .. {hi}");
    assert!((lo - 380.0).abs() < 8.0, "unexpected width {lo}");
    let bad = column_widths(false);
    let (blo, bhi) = (bad.iter().cloned().fold(f32::MAX, f32::min), bad.iter().cloned().fold(0.0, f32::max));
    println!("old layout: width {blo:.1} .. {bhi:.1}");
    assert!(bhi - blo > 20.0, "the test must be able to detect a growing column");
}

/// ETA quality: compare every estimate with the time that really remained.
fn eta_error(job: &export::Job, t: &tools::Tools) -> (usize, f64, f64) {
    let samples = std::sync::Mutex::new(Vec::<(f64, f32, Option<f64>)>::new());
    let t0 = std::time::Instant::now();
    export::run(t, job, &|f, _, eta| samples.lock().unwrap().push((t0.elapsed().as_secs_f64(), f, eta)), &std::sync::atomic::AtomicBool::new(false)).unwrap();
    let total = t0.elapsed().as_secs_f64();
    let s = samples.lock().unwrap().clone();
    let with_eta: Vec<_> = s.iter().filter(|x| x.2.is_some() && x.0 > total * 0.3 && x.0 < total * 0.95).collect();
    let mut worst = 0.0f64;
    let mut sum = 0.0;
    for (at, frac, eta) in &with_eta {
        let real = total - at;
        let est = eta.unwrap();
        let rel = (est - real).abs() / real.max(0.5);
        println!("  t={at:6.1}s bar={:3.0}% estimated {est:6.1}s real {real:6.1}s  err {:3.0}%", frac * 100.0, rel * 100.0);
        worst = worst.max(rel);
        sum += rel;
    }
    (with_eta.len(), sum / with_eta.len().max(1) as f64, worst)
}

#[test]
fn export_eta_is_accurate() {
    let Some((t, _)) = setup() else { return };
    let long = t.temp.join("selftest_long.mp4");
    if !long.exists() {
        return;
    }
    let info = probe::probe(&t, &long).unwrap();
    let dir = t.temp.join("selftest_eta");
    let _ = std::fs::remove_dir_all(&dir);
    // plain export of many 1080p frames
    let job = export::Job { video: long, out_dir: dir.clone(), start: 0, count: 450, fps: info.fps, vf: String::new(), format: Format::Png, high_bit: false, ai: None };
    let (n, mean, worst) = eta_error(&job, &t);
    println!("plain export: {n} estimates, mean error {:.0}%, worst {:.0}%", mean * 100.0, worst * 100.0);
    assert!(n >= 3 && mean < 0.25, "plain ETA too inaccurate");
    // neural export: per-frame time is measured, so the estimate converges quickly
    if t.realesrgan_cpu.is_some() {
        let p = t.temp.join("selftest.mp4");
        let info = probe::probe(&t, &p).unwrap();
        let mut ai = crate::nn::default_settings(&t);
        ai.dev_esrgan = -1;
        let job = export::Job { video: p, out_dir: dir.join("ai"), start: 0, count: 24, fps: info.fps, vf: String::new(), format: Format::Png, high_bit: false, ai: Some(ai) };
        let (n, mean, worst) = eta_error(&job, &t);
        println!("neural export: {n} estimates, mean error {:.0}%, worst {:.0}%", mean * 100.0, worst * 100.0);
        assert!(n >= 3 && mean < 0.30, "neural ETA too inaccurate");
    }
}
