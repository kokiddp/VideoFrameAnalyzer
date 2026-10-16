use crate::decoder::Waker;
use crate::export::{self, Format};
use crate::bench::{self, Cell, Msg as BenchMsg, Plan, Spec};
use crate::filters::{self, AiSettings, Engine, Filter};
use crate::probe;
use crate::nn::{self, Benchmark, Device};
use crate::player::Playback;
use crate::tools::{self, Tools};
use crate::video::{PlayRange, Video};
use crate::widgets::{filter_card, icon_button, knob, range_bars, slider_width, Icon, LABEL_WIDTH};
use eframe::egui::{self, Align2, Color32, FontId, Key, Modifiers, RichText, Sense, Vec2};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Default width of the right column (user-resizable); card widths derive from it.
const SIDE_WIDTH: f32 = 380.0;

/// State of the benchmark window.
struct BenchUi {
    open: bool,
    /// (engine, model/dir name, selected)
    models: Vec<(Engine, String, bool)>,
    /// (height, selected)
    heights: Vec<(u32, bool)>,
    /// (device id, name, selected)
    devices: Vec<(i32, String, bool)>,
    timeout: u64,
    running: bool,
    rx: Option<Receiver<BenchMsg>>,
    cancel: Arc<AtomicBool>,
    plan: Option<Plan>,
    cells: HashMap<(usize, usize, i32), Cell>,
    done: usize,
    total: usize,
    started: Instant,
    /// device columns of the running / last plan
    columns: Vec<(i32, String)>,
}

impl BenchUi {
    fn new() -> BenchUi {
        BenchUi {
            open: false,
            models: Vec::new(),
            heights: vec![(360, true), (720, true), (1080, false)],
            devices: Vec::new(),
            timeout: 60,
            running: false,
            rx: None,
            cancel: Arc::new(AtomicBool::new(false)),
            plan: None,
            cells: HashMap::new(),
            done: 0,
            total: 0,
            started: Instant::now(),
            columns: Vec::new(),
        }
    }
}

enum Dialog {
    Video(PathBuf),
    OutDir(PathBuf),
}

enum ExportMsg {
    Progress(f32, String, Option<f64>),
    Done(Result<Vec<PathBuf>, String>),
}

pub struct App {
    tools: Option<Tools>,
    prep_rx: Option<Receiver<Result<Tools, String>>>,
    prep_err: Option<String>,
    wake: Waker,

    video: Option<Video>,
    filters: Vec<Filter>,
    filters_on: bool,
    ai: AiSettings,

    pb: Playback,
    frame_mode: bool,

    zoom: f32,
    fit: bool,
    pan: Vec2,
    nearest: bool,

    format: Format,
    out_dir: Option<PathBuf>,
    export_count: u64,
    export_rx: Option<Receiver<ExportMsg>>,
    export_status: Option<(String, bool)>,
    export_progress: Option<(f32, String, Instant)>,
    /// smoothed remaining-time estimate: (seconds remaining when measured, when)
    export_eta: Option<(f64, Instant)>,
    export_cancel: Arc<AtomicBool>,
    /// CUE range (inclusive frame indices): playback is confined to it while set.
    cue: Option<(u64, u64)>,
    last_export: Option<PathBuf>,

    dialog_tx: Sender<Dialog>,
    dialog_rx: Receiver<Dialog>,
    status: Option<(String, bool)>,
    goto_text: String,
    show_log: bool,
    bench: BenchUi,
    devices: Vec<(Engine, Vec<Device>)>,
    bench_rx: Option<Receiver<Benchmark>>,
    device_user_set: [bool; 2],
    c_held: bool,
    /// debounce of the filter chain: last value seen and when it last changed
    vf_seen: String,
    vf_seen_at: Instant,
    /// true from the moment a new filter chain starts being applied until the frame shows it
    filters_busy: bool,
    /// filters were switched off for playback and must come back on pause
    filters_resume: bool,
    prev_playing: bool,
    force_apply: bool,
    busy_since: Option<Instant>,
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>, initial: Option<PathBuf>) -> App {
        let ctx = cc.egui_ctx.clone();
        let wake: Waker = {
            let ctx = ctx.clone();
            Arc::new(move || ctx.request_repaint())
        };
        let (ptx, prx) = channel();
        {
            let wake = wake.clone();
            std::thread::spawn(move || {
                let _ = ptx.send(tools::prepare());
                wake();
            });
        }
        let (dialog_tx, dialog_rx) = channel();
        let app = App {
            tools: None,
            prep_rx: Some(prx),
            prep_err: None,
            wake,
            video: None,
            filters: filters::default_filters(),
            filters_on: true,
            ai: AiSettings {
                enabled: false,
                engine: Engine::Esrgan,
                model: "realesr-animevideov3-x2".into(),
                scale: 2,
                w2x_model: "upconv_7_photo".into(),
                w2x_scale: 2,
                w2x_noise: 1,
                dev_esrgan: 0,
                dev_w2x: 0,
            },
            pb: Playback::new(),
            frame_mode: false,
            zoom: 1.0,
            fit: true,
            pan: Vec2::ZERO,
            nearest: false,
            format: Format::Png,
            out_dir: None,
            export_count: 1,
            export_rx: None,
            export_status: None,
            export_progress: None,
            export_eta: None,
            export_cancel: Arc::new(AtomicBool::new(false)),
            cue: None,
            last_export: None,
            dialog_tx,
            dialog_rx,
            status: None,
            goto_text: String::new(),
            show_log: true,
            bench: BenchUi::new(),
            devices: Vec::new(),
            bench_rx: None,
            device_user_set: [false; 2],
            c_held: false,
            vf_seen: String::new(),
            vf_seen_at: Instant::now(),
            filters_busy: false,
            filters_resume: false,
            prev_playing: false,
            force_apply: false,
            busy_since: None,
        };
        if let Some(p) = initial {
            app.dialog_tx.send(Dialog::Video(p)).ok();
        }
        app
    }

    /// Filter chain used for the preview (empty while the A/B "original" view is active).
    fn vf(&self) -> String {
        if self.filters_on && !self.c_held {
            filters::chain(&self.filters)
        } else {
            String::new()
        }
    }

    fn open_video(&mut self, path: PathBuf) {
        let Some(tools) = &self.tools else { return };
        match probe::probe(tools, &path) {
            Ok(info) => {
                self.status = None;
                self.pb.playing = false;
                self.pb.reverse = false;
                self.fit = true;
                self.pan = Vec2::ZERO;
                self.out_dir = None;
                let vf = self.vf();
                self.video = Some(Video::new(path, info, vf));
            }
            Err(e) => self.status = Some((format!("Impossibile aprire il filmato: {e}"), true)),
        }
    }

    fn pick_video(&self) {
        let tx = self.dialog_tx.clone();
        let wake = self.wake.clone();
        std::thread::spawn(move || {
            if let Some(p) = rfd::FileDialog::new().set_title("Apri filmato").pick_file() {
                let _ = tx.send(Dialog::Video(p));
                wake();
            }
        });
    }

    fn pick_dir(&self) {
        let tx = self.dialog_tx.clone();
        let wake = self.wake.clone();
        std::thread::spawn(move || {
            if let Some(p) = rfd::FileDialog::new().set_title("Cartella di esportazione").pick_folder() {
                let _ = tx.send(Dialog::OutDir(p));
                wake();
            }
        });
    }

    fn effective_out_dir(&self) -> Option<PathBuf> {
        if let Some(d) = &self.out_dir {
            return Some(d.clone());
        }
        let v = self.video.as_ref()?;
        let stem = v.path.file_stem()?.to_string_lossy().to_string();
        Some(v.path.parent()?.join(format!("{stem}_frames")))
    }

    fn ai_available(&self) -> bool {
        self.tools.as_ref().map_or(false, |t| Engine::ALL.into_iter().any(|e| nn::engine_available(t, e)))
    }

    fn devices_for(&self, e: Engine) -> &[Device] {
        self.devices.iter().find(|(x, _)| *x == e).map_or(&[], |(_, v)| v.as_slice())
    }

    fn start_bench(&mut self, tools: &Tools, force: bool) {
        if self.bench_rx.is_some() || !Engine::ALL.into_iter().any(|e| nn::engine_available(tools, e)) {
            return;
        }
        let (tx, rx) = channel();
        let (tools, wake) = (tools.clone(), self.wake.clone());
        self.bench_rx = Some(rx);
        std::thread::spawn(move || {
            let _ = tx.send(nn::benchmark(&tools, force));
            wake();
        });
    }

    /// AI stage shown in the preview.
    fn ai_active(&self) -> bool {
        self.ai.enabled
            && self.filters_on
            && !self.c_held
            && self.bench_rx.is_none()
            && self.tools.as_ref().map_or(false, |t| nn::engine_available(t, self.ai.engine))
            && !self.devices_for(self.ai.engine).is_empty()
    }

    fn ai_key(&self) -> String {
        self.ai.key()
    }

    fn start_export(&mut self, start: u64, count: u64) {
        let (Some(tools), Some(v), Some(dir)) = (self.tools.clone(), self.video.as_ref(), self.effective_out_dir()) else { return };
        if self.export_rx.is_some() {
            return;
        }
        // the export follows the "Applica" switch: on = active filters (+ neural stage), off = untouched frames.
        // (while playing the switch is temporarily off for the preview, so the remembered state counts)
        let filters_enabled = self.filters_on || self.filters_resume;
        let job = export::Job {
            video: v.path.clone(),
            out_dir: dir,
            start,
            count: count.max(1),
            fps: v.info.fps,
            vf: if filters_enabled { filters::chain(&self.filters) } else { String::new() },
            format: self.format,
            high_bit: v.info.high_bit_depth,
            ai: if filters_enabled && self.ai.enabled && !self.devices_for(self.ai.engine).is_empty() { Some(self.ai.clone()) } else { None },
        };
        let (tx, rx) = channel();
        let wake = self.wake.clone();
        self.export_rx = Some(rx);
        self.export_status = None;
        self.export_progress = Some((0.0, "Avvio…".into(), Instant::now()));
        self.export_eta = None;
        self.export_cancel.store(false, Ordering::Relaxed);
        let cancel = self.export_cancel.clone();
        std::thread::spawn(move || {
            let tx2 = tx.clone();
            let w2 = wake.clone();
            let res = export::run(
                &tools,
                &job,
                &move |f, s, eta| {
                    let _ = tx2.send(ExportMsg::Progress(f, s, eta));
                    w2();
                },
                &cancel,
            );
            let _ = tx.send(ExportMsg::Done(res));
            wake();
        });
    }

    fn poll(&mut self, ctx: &egui::Context) {
        if let Some(rx) = &self.prep_rx {
            if let Ok(r) = rx.try_recv() {
                match r {
                    Ok(t) => {
                        let enabled = self.ai.enabled;
                        self.ai = nn::default_settings(&t);
                        self.ai.enabled = enabled;
                        self.start_bench(&t, false);
                        self.tools = Some(t);
                    }
                    Err(e) => self.prep_err = Some(e),
                }
                self.prep_rx = None;
            }
        }
        if let Some(rx) = &self.bench_rx {
            if let Ok(bench) = rx.try_recv() {
                self.bench_rx = None;
                let current = self.ai.engine;
                for (engine, devs) in &bench {
                    let Some(best) = devs.first() else { continue };
                    self.ai.engine = *engine;
                    let user = self.device_user_set[Engine::ALL.iter().position(|e| e == engine).unwrap()];
                    if !user || !devs.iter().any(|d| d.id == self.ai.device()) {
                        self.ai.set_device(best.id);
                    }
                    crate::log::log(format!("[bench] {}: processore predefinito {} ({:.2}s)", engine.label(), best.name, best.secs));
                }
                self.ai.engine = current;
                // keep the selected engine usable
                if let Some(t) = &self.tools {
                    if !nn::engine_available(t, self.ai.engine) {
                        if let Some(e) = Engine::ALL.into_iter().find(|e| nn::engine_available(t, *e)) {
                            self.ai.engine = e;
                        }
                    }
                }
                self.devices = bench;
            }
        }
        if let Some(rx) = &self.bench.rx {
            let mut finished = false;
            while let Ok(m) = rx.try_recv() {
                match m {
                    BenchMsg::Cell { spec, res, dev, cell } => {
                        self.bench.cells.insert((spec, res, dev), cell);
                        self.bench.done += 1;
                    }
                    BenchMsg::Finished => finished = true,
                }
            }
            if finished {
                self.bench.running = false;
                self.bench.rx = None;
            }
        }
        while let Ok(d) = self.dialog_rx.try_recv() {
            match d {
                Dialog::Video(p) => {
                    if self.tools.is_some() {
                        self.open_video(p)
                    } else {
                        // tools not ready yet: retry shortly
                        let _ = self.dialog_tx.send(Dialog::Video(p));
                        ctx.request_repaint_after(std::time::Duration::from_millis(200));
                        break;
                    }
                }
                Dialog::OutDir(p) => self.out_dir = Some(p),
            }
        }
        if let Some(rx) = &self.export_rx {
            let mut done = false;
            while let Ok(m) = rx.try_recv() {
                match m {
                    ExportMsg::Progress(f, s, eta) => {
                        if let Some(p) = &mut self.export_progress {
                            p.0 = f;
                            p.1 = s;
                        }
                        if let Some(rem) = eta {
                            // blend with what the countdown currently shows so the figure never jumps around
                            let shown = self.export_eta.map(|(v, at)| (v - at.elapsed().as_secs_f64()).max(0.0));
                            let blended = match shown {
                                Some(c) if rem > 0.0 => 0.65 * c + 0.35 * rem,
                                _ => rem,
                            };
                            self.export_eta = Some((blended, Instant::now()));
                        }
                    }
                    ExportMsg::Done(Ok(files)) => {
                        let last = files.last().cloned();
                        self.export_status = Some((
                            format!("Esportati {} file. Ultimo: {}", files.len(), last.as_ref().map(|p| p.display().to_string()).unwrap_or_default()),
                            false,
                        ));
                        self.last_export = last;
                        done = true;
                    }
                    ExportMsg::Done(Err(e)) => {
                        self.export_status = Some((e, true));
                        done = true;
                    }
                }
            }
            if done {
                self.export_rx = None;
                self.export_progress = None;
            }
        }
        let dropped = ctx.input(|i| i.raw.dropped_files.clone());
        if let Some(p) = dropped.into_iter().next().map(|f| f.path().to_path_buf()) {
            self.open_video(p);
        }
    }

    // ---- playback ---------------------------------------------------------------------------

    fn step(&mut self, delta: i64) {
        self.pb.playing = false;
        if let Some(v) = &mut self.video {
            v.want = v.clamp_want(v.cur as i64 + delta);
        }
    }

    /// Playback bounds: the CUE range if set, otherwise the whole file.
    fn range(&self) -> PlayRange {
        self.pb.range(self.cue)
    }

    fn seek(&mut self, idx: u64) {
        if let Some(v) = &mut self.video {
            v.want = v.clamp_want(idx as i64);
            self.pb.play_pos = v.want as f64;
        }
    }

    fn toggle_play(&mut self, reverse: bool) {
        let range = self.range();
        if let Some(v) = &mut self.video {
            self.pb.toggle(v, reverse, &range);
            if self.pb.playing {
                self.frame_mode = false;
            }
        }
    }

    fn advance_clock(&mut self) {
        let range = self.range();
        match &mut self.video {
            Some(v) => self.pb.advance(v, Instant::now(), &range),
            None => self.pb.last_tick = Instant::now(),
        }
    }

    fn handle_keys(&mut self, ctx: &egui::Context) {
        if ctx.egui_wants_keyboard_input() {
            return;
        }
        let (space, right, left, home, end, enter, export, shift, ctrl, f_key, o_key) = ctx.input(|i| {
            (
                i.key_pressed(Key::Space),
                i.key_pressed(Key::ArrowRight) || i.key_pressed(Key::Period),
                i.key_pressed(Key::ArrowLeft) || i.key_pressed(Key::Comma),
                i.key_pressed(Key::Home),
                i.key_pressed(Key::End),
                i.key_pressed(Key::Enter),
                i.key_pressed(Key::E) || (i.modifiers.command && i.key_pressed(Key::S)),
                i.modifiers.shift,
                i.modifiers.command,
                i.key_pressed(Key::F),
                i.modifiers.matches_logically(Modifiers::COMMAND) && i.key_pressed(Key::O),
            )
        });
        let (k_in, k_out, k_esc) = ctx.input(|i| (i.key_pressed(Key::I), i.key_pressed(Key::O) && !i.modifiers.command, i.key_pressed(Key::Escape)));
        if let Some(v) = &self.video {
            let (cur, last) = (v.cur, v.last_frame().unwrap_or(v.cur));
            if k_in {
                self.cue = Some((cur, self.cue.map_or(last, |c| c.1.max(cur))));
            }
            if k_out {
                self.cue = Some((self.cue.map_or(0, |c| c.0.min(cur)), cur));
            }
            if k_esc {
                self.cue = None;
            }
        }
        if ctx.input(|i| i.key_pressed(Key::L)) {
            self.show_log = !self.show_log;
        }
        if o_key {
            self.pick_video();
        }
        if self.video.is_none() {
            return;
        }
        let fps = self.video.as_ref().unwrap().info.fps.round() as i64;
        let amount = if ctrl { fps.max(1) } else if shift { 10 } else { 1 };
        if space {
            if self.frame_mode {
                self.step(1);
            } else {
                self.toggle_play(false);
            }
        }
        if enter && self.frame_mode {
            self.step(1);
        }
        if right {
            self.step(amount);
        }
        if left {
            self.step(-amount);
        }
        if home {
            self.pb.playing = false;
            self.seek(0);
        }
        if end {
            if let Some(l) = self.video.as_ref().and_then(|v| v.last_frame()) {
                self.pb.playing = false;
                self.seek(l);
            }
        }
        if f_key {
            self.frame_mode = !self.frame_mode;
            if self.frame_mode {
                self.pb.playing = false;
            }
        }
        if export {
            if let Some(v) = &self.video {
                let (start, count) = (v.cur, self.export_count);
                self.start_export(start, count);
            }
        }
    }

    // ---- UI pieces ----------------------------------------------------------------------------

    fn top_bar(&mut self, ui: &mut egui::Ui) {
        egui::Panel::top("top").show(ui, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Apri filmato…").on_hover_text("Ctrl+O, oppure trascina il file nella finestra").clicked() {
                    self.pick_video();
                }
                if let Some(v) = &self.video {
                    let i = &v.info;
                    ui.label(RichText::new(v.path.file_name().unwrap_or_default().to_string_lossy()).strong());
                    ui.label(format!(
                        "{}×{}  {:.3} fps  {}  {}{}",
                        i.width,
                        i.height,
                        i.fps,
                        i.codec,
                        i.pix_fmt,
                        i.duration.map(|d| format!("  {}", fmt_time(d))).unwrap_or_default()
                    ));
                }
                if let Some((s, err)) = &self.status {
                    ui.colored_label(if *err { Color32::LIGHT_RED } else { Color32::LIGHT_GREEN }, s);
                }
            });
        });
    }

    fn transport(&mut self, ui: &mut egui::Ui) {
        if self.video.is_none() {
            return;
        }
        egui::Panel::bottom("transport").show(ui, |ui| {
            let (cur, last, fps, loading) = {
                let v = self.video.as_ref().unwrap();
                (v.cur, v.last_frame(), v.info.fps, v.loading)
            };
            // CUE range bar + seek bar
            let pos = self.video.as_ref().unwrap().want;
            let max = last.unwrap_or(pos.max(1)).max(1);
            let bars = range_bars(ui, pos, max, self.cue);
            if let Some(c) = bars.cue {
                self.cue = c;
            }
            if let Some(f) = bars.seek {
                self.pb.playing = false;
                self.seek(f);
            }
            if let Some((a, b)) = self.cue {
                let fps = self.video.as_ref().map_or(25.0, |v| v.info.fps);
                let n = b - a + 1;
                ui.horizontal(|ui| {
                    ui.colored_label(Color32::from_rgb(255, 200, 60), format!("CUE {a} – {b}  ({n} fotogrammi, {})", fmt_time(n as f64 / fps)));
                    if ui.small_button("Vai all'inizio").clicked() {
                        self.pb.playing = false;
                        self.seek(a);
                    }
                    if ui.small_button("Deseleziona (Esc)").clicked() {
                        self.cue = None;
                    }
                    let busy = self.export_rx.is_some();
                    if ui.add_enabled(!busy, egui::Button::new(RichText::new(format!("Esporta intervallo ({n})")).strong())).clicked() {
                        self.start_export(a, n);
                    }
                });
            }
            ui.horizontal(|ui| {
                if icon_button(ui, Icon::First, "Inizio (Home)") {
                    self.pb.playing = false;
                    self.seek(0);
                }
                let rev_icon = if self.pb.playing && self.pb.reverse { Icon::Pause } else { Icon::PlayRev };
                if icon_button(ui, rev_icon, "Riproduci all'indietro") {
                    self.toggle_play(true);
                }
                if icon_button(ui, Icon::StepBack, "Fotogramma precedente (←)") {
                    self.step(-1);
                }
                let fwd_icon = if self.pb.playing && !self.pb.reverse { Icon::Pause } else { Icon::Play };
                if icon_button(ui, fwd_icon, "Riproduci / pausa (Spazio)") {
                    self.toggle_play(false);
                }
                if icon_button(ui, Icon::StepFwd, "Fotogramma successivo (→)") {
                    self.step(1);
                }
                if icon_button(ui, Icon::Last, "Fine (End)") {
                    if let Some(l) = last {
                        self.pb.playing = false;
                        self.seek(l);
                    }
                }
                ui.separator();
                ui.label(format!("Fotogramma {cur}{}", last.map(|l| format!(" / {l}")).unwrap_or_default()));
                ui.label(fmt_time(cur as f64 / fps));
                ui.add(egui::TextEdit::singleline(&mut self.goto_text).desired_width(60.0).hint_text("vai a #"));
                if ui.input(|i| i.key_pressed(Key::Enter)) && ui.memory(|m| m.focused().is_some()) {
                    if let Ok(n) = self.goto_text.trim().parse::<u64>() {
                        self.pb.playing = false;
                        self.seek(n);
                        self.goto_text.clear();
                    }
                }
                if loading {
                    ui.spinner();
                }
            });
            ui.horizontal(|ui| {
                ui.label("Velocità");
                let mut changed = knob(ui, &mut self.pb.speed, 0.05, 16.0);
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        changed |= ui
                            .add(egui::DragValue::new(&mut self.pb.speed).range(0.05..=16.0).speed(0.01).suffix("×").max_decimals(3))
                            .changed();
                        for p in [0.1f32, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0] {
                            if ui.small_button(format!("{p}×")).clicked() {
                                self.pb.speed = p;
                            }
                        }
                    });
                    ui.spacing_mut().slider_width = 360.0;
                    changed |= ui.add(egui::Slider::new(&mut self.pb.speed, 0.05..=16.0).logarithmic(true).show_value(false)).changed();
                });
                let _ = changed;
                ui.separator();
                ui.checkbox(&mut self.pb.looping, "Ripeti");
                if ui
                    .checkbox(&mut self.frame_mode, "Fotogramma per fotogramma")
                    .on_hover_text("Spazio / Invio / → avanzano di un fotogramma, ← indietro, E o Ctrl+S esporta (tasto F per attivare)")
                    .changed()
                    && self.frame_mode
                {
                    self.pb.playing = false;
                }
            });
        });
    }

    fn side_panel(&mut self, ui: &mut egui::Ui) {
        egui::Panel::right("side").resizable(true).default_size(SIDE_WIDTH).size_range(300.0..=760.0).show(ui, |ui| {
            // measured once, before any content is laid out, so content can never feed back into the width
            let cw = (ui.available_width() - 30.0).max(240.0);
            let slider_w = slider_width(cw);
            self.log_panel(ui);
            egui::ScrollArea::both().auto_shrink([false, false]).show(ui, |ui| {
                let locked = self.filters_busy || self.pb.playing;
                let n_active = self.filters.iter().filter(|f| f.enabled).count() + self.ai.enabled as usize;
                let applying = self.video.as_ref().map_or(false, |v| v.shown_gen != v.gen || (v.loading && !self.pb.playing));
                ui.horizontal(|ui| {
                    ui.heading("Filtri");
                    ui.label(format!("{n_active} attivi"));
                    if applying {
                        ui.spinner();
                    }
                });
                ui.add_enabled_ui(!locked, |ui| {
                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.filters_on, "Applica")
                        .on_hover_text("Disattiva per vedere l'originale. Tieni premuto C per un confronto rapido.");
                    if self.c_held {
                        ui.colored_label(Color32::YELLOW, "originale (C)");
                    }
                    if ui.small_button("Tutti on").clicked() {
                        self.filters.iter_mut().for_each(|f| f.enabled = true);
                        self.filters_on = true;
                    }
                    if ui.small_button("Tutti off").clicked() {
                        self.filters.iter_mut().for_each(|f| f.enabled = false);
                        self.ai.enabled = false;
                    }
                });
                if let Some(v) = &self.video {
                    if let (Some(e), false) = (&v.error, self.vf().is_empty()) {
                        ui.colored_label(Color32::LIGHT_RED, format!("Errore filtri: {}", e.lines().last().unwrap_or("")));
                    }
                }
                ui.add_space(2.0);
                for f in self.filters.iter_mut() {
                    filter_card(ui, f, cw, slider_w);
                }
                });
                if self.pb.playing {
                    ui.small("Filtri sospesi durante la riproduzione: tornano attivi alla pausa o in modalità fotogramma per fotogramma.");
                } else if locked {
                    ui.small("Applicazione dei filtri in corso: attendi che il fotogramma sia aggiornato…");
                }

                ui.add_space(2.0);
                let has_ai = self.ai_available();
                let ai_status = self.video.as_ref().and_then(|v| {
                    if !self.ai.enabled {
                        None
                    } else if v.ai_busy {
                        Some((format!("elaborazione… {:.0}s", v.ai_started.map_or(0.0, |t| t.elapsed().as_secs_f32())), Color32::YELLOW))
                    } else if let Some(e) = &v.ai_err {
                        Some((format!("errore: {}", e.lines().next().unwrap_or("")), Color32::LIGHT_RED))
                    } else if self.pb.playing {
                        Some(("in pausa durante la riproduzione".to_string(), Color32::GRAY))
                    } else if v.ai_key_matches(&self.ai_key()) {
                        Some(("applicata".to_string(), Color32::LIGHT_GREEN))
                    } else {
                        None
                    }
                });
                let stroke = if self.ai.enabled { egui::Stroke::new(1.5, ui.visuals().selection.bg_fill) } else { ui.visuals().widgets.noninteractive.bg_stroke };
                egui::Frame::group(ui.style()).stroke(stroke).show(ui, |ui| {
                    ui.set_width(cw);
                    ui.set_max_width(cw);
                    ui.add_enabled_ui(has_ai, |ui| {
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut self.ai.enabled, RichText::new("Rete neurale").strong())
                                .on_hover_text("Super-risoluzione / pulizia con rete neurale (GPU Vulkan o CPU).\nIn anteprima viene applicata solo al fotogramma fermo.");
                        });
                        if let Some((t, c)) = &ai_status {
                            ui.add(egui::Label::new(RichText::new(t).color(*c)).wrap());
                        }
                        if self.ai.enabled {
                            let tools = self.tools.clone();
                            let avail = |e: Engine| tools.as_ref().map_or(false, |t| nn::engine_available(t, e));
                            ui.horizontal(|ui| {
                                ui.add_sized([LABEL_WIDTH, 18.0], egui::Label::new("Motore"));
                                for e in Engine::ALL {
                                    ui.add_enabled_ui(avail(e), |ui| {
                                        ui.selectable_value(&mut self.ai.engine, e, e.label());
                                    });
                                }
                            });
                            let engine = self.ai.engine;
                            // processor (benchmarked per engine)
                            ui.horizontal(|ui| {
                                ui.add_sized([LABEL_WIDTH, 18.0], egui::Label::new("Processore"));
                                if self.bench_rx.is_some() {
                                    ui.spinner();
                                    ui.small("test in corso…");
                                } else {
                                    let devs: Vec<Device> = self.devices_for(engine).to_vec();
                                    let cur_dev = self.ai.device();
                                    let sel = devs.iter().find(|d| d.id == cur_dev).map(|d| d.name.clone()).unwrap_or_else(|| "nessuno disponibile".into());
                                    let mut chosen = cur_dev;
                                    egui::ComboBox::from_id_salt("aidev").width(cw - LABEL_WIDTH - 70.0).truncate().selected_text(sel).show_ui(ui, |ui| {
                                        for (i, d) in devs.iter().enumerate() {
                                            let label = format!("{} · {:.1}s{}", d.name, d.secs, if i == 0 { " · più veloce" } else { "" });
                                            ui.selectable_value(&mut chosen, d.id, label);
                                        }
                                    });
                                    if chosen != cur_dev {
                                        self.ai.set_device(chosen);
                                        self.device_user_set[Engine::ALL.iter().position(|e| *e == engine).unwrap()] = true;
                                    }
                                    if ui.small_button("Test").on_hover_text("Ripeti il test prestazioni (entrambi i motori)").clicked() {
                                        if let Some(t) = self.tools.clone() {
                                            self.devices.clear();
                                            self.device_user_set = [false; 2];
                                            self.start_bench(&t, true);
                                        }
                                    }
                                }
                            });
                            match engine {
                                Engine::Esrgan => {
                                    let models = self.tools.as_ref().map(|t| t.models.clone()).unwrap_or_default();
                                    ui.horizontal(|ui| {
                                        ui.add_sized([LABEL_WIDTH, 18.0], egui::Label::new("Modello"));
                                        egui::ComboBox::from_id_salt("aimodel").width(cw - LABEL_WIDTH - 40.0).truncate().selected_text(self.ai.model.clone()).show_ui(ui, |ui| {
                                            for m in &models {
                                                ui.selectable_value(&mut self.ai.model, m.clone(), m);
                                            }
                                        });
                                    });
                                    let fixed = self.ai.model.rsplit_once("-x").map_or(false, |(_, r)| r.starts_with(|c: char| c.is_ascii_digit()));
                                    ui.add_enabled_ui(!fixed, |ui| {
                                        ui.horizontal(|ui| {
                                            ui.add_sized([LABEL_WIDTH, 18.0], egui::Label::new("Fattore"));
                                            for s in [2u32, 3, 4] {
                                                ui.selectable_value(&mut self.ai.scale, s, format!("{s}×"));
                                            }
                                            if fixed {
                                                ui.small("fisso");
                                            }
                                        });
                                    });
                                    ui.add(egui::Label::new(RichText::new("animevideov3 = veloce · x4plus = massima qualità").small()).wrap());
                                }
                                Engine::Waifu2x => {
                                    let models = self.tools.as_ref().map(|t| t.w2x_models.clone()).unwrap_or_default();
                                    let nice = |m: &str| match m {
                                        "upconv_7_photo" => "foto (veloce)".to_string(),
                                        "upconv_7_anime_style_art_rgb" => "anime/disegni (veloce)".to_string(),
                                        "cunet" => "cunet (qualità, più lento)".to_string(),
                                        o => o.to_string(),
                                    };
                                    ui.horizontal(|ui| {
                                        ui.add_sized([LABEL_WIDTH, 18.0], egui::Label::new("Modello"));
                                        egui::ComboBox::from_id_salt("w2xmodel").width(cw - LABEL_WIDTH - 40.0).truncate().selected_text(nice(&self.ai.w2x_model)).show_ui(ui, |ui| {
                                            for m in &models {
                                                ui.selectable_value(&mut self.ai.w2x_model, m.clone(), nice(m));
                                            }
                                        });
                                    });
                                    ui.horizontal(|ui| {
                                        ui.add_sized([LABEL_WIDTH, 18.0], egui::Label::new("Pulizia"));
                                        egui::ComboBox::from_id_salt("w2xnoise").selected_text(if self.ai.w2x_noise < 0 { "nessuna".to_string() } else { format!("livello {}", self.ai.w2x_noise) }).show_ui(ui, |ui| {
                                            ui.selectable_value(&mut self.ai.w2x_noise, -1, "nessuna");
                                            for n in 0..=3 {
                                                ui.selectable_value(&mut self.ai.w2x_noise, n, format!("livello {n}"));
                                            }
                                        });
                                    });
                                    let cunet = self.ai.w2x_model == "cunet";
                                    ui.horizontal(|ui| {
                                        ui.add_sized([LABEL_WIDTH, 18.0], egui::Label::new("Fattore"));
                                        ui.add_enabled_ui(cunet && self.ai.w2x_noise >= 0, |ui| {
                                            ui.selectable_value(&mut self.ai.w2x_scale, 1, "1× (solo pulizia)");
                                        });
                                        ui.selectable_value(&mut self.ai.w2x_scale, 2, "2×");
                                        ui.selectable_value(&mut self.ai.w2x_scale, 4, "4×");
                                    });
                                    ui.add(egui::Label::new(RichText::new("waifu2x: più leggero di Real-ESRGAN; la pulizia riduce rumore/blocchi di compressione").small()).wrap());
                                }
                            }
                        }
                    });
                    if !has_ai {
                        ui.small("Nessun motore neurale incluso in questa build.");
                    }
                });
                if ui.button("Benchmark…").on_hover_text("Confronta modelli, risoluzioni e processori (CPU/GPU)").clicked() {
                    self.open_bench();
                }

                ui.separator();
                ui.heading("Esporta fotogrammi");
                egui::ComboBox::from_id_salt("fmt").selected_text(self.format.label()).show_ui(ui, |ui| {
                    for f in Format::ALL {
                        ui.selectable_value(&mut self.format, f, f.label());
                    }
                });
                {
                    let on = self.filters_on || self.filters_resume;
                    let n = self.filters.iter().filter(|f| f.enabled).count() + self.ai.enabled as usize;
                    let (txt, col) = if on && n > 0 {
                        (format!("Si esporta con {n} filtri/elaborazioni attivi (come l'anteprima)"), Color32::LIGHT_GREEN)
                    } else if on {
                        ("«Applica» è attivo ma nessun filtro è selezionato: fotogrammi originali".to_string(), Color32::GRAY)
                    } else {
                        ("«Applica» è disattivato: si esportano i fotogrammi originali".to_string(), Color32::YELLOW)
                    };
                    ui.add(egui::Label::new(RichText::new(txt).color(col).small()).wrap());
                }
                ui.horizontal(|ui| {
                    ui.label("Fotogrammi consecutivi");
                    ui.add(egui::DragValue::new(&mut self.export_count).range(1..=100000));
                });
                if let Some(d) = self.effective_out_dir() {
                    ui.horizontal_wrapped(|ui| {
                        ui.small(d.display().to_string());
                    });
                }
                if ui.button("Cartella…").clicked() {
                    self.pick_dir();
                }
                let busy = self.export_rx.is_some();
                ui.add_enabled_ui(self.video.is_some() && !busy, |ui| {
                    if ui
                        .add_sized([ui.available_width(), 34.0], egui::Button::new(RichText::new("Esporta fotogramma corrente").strong()))
                        .on_hover_text("E oppure Ctrl+S")
                        .clicked()
                    {
                        if let Some(v) = &self.video {
                            let (start, count) = (v.cur, self.export_count);
                            self.start_export(start, count);
                        }
                    }
                });
                if let Some((s, err)) = &self.export_status {
                    ui.colored_label(if *err { Color32::LIGHT_RED } else { Color32::LIGHT_GREEN }, s);
                }
                if let Some(p) = &self.last_export {
                    if ui.small_button("Mostra nella cartella").clicked() {
                        reveal(p);
                    }
                }
                ui.add_space(8.0);
                ui.collapsing("Scorciatoie", |ui| {
                    ui.small("Spazio  riproduci/pausa (o avanti 1 in modalità fotogramma)\n← →  fotogramma ±1   Maiusc ±10   Ctrl ±1 s\nHome / End  inizio / fine\nF  modalità fotogramma per fotogramma\nE o Ctrl+S  esporta\nI / O  ingresso / uscita CUE   Esc  deseleziona CUE\nRotella: zoom   Trascina: sposta   Doppio clic: adatta");
                });
                ui.checkbox(&mut self.nearest, "Pixel esatti (zoom senza smussare)");
            });
        });
    }

    fn open_bench(&mut self) {
        let Some(t) = self.tools.clone() else { return };
        // rebuild option lists from the current tools / detected processors
        let old: Vec<(Engine, String, bool)> = self.bench.models.clone();
        self.bench.models.clear();
        for m in &t.models {
            let on = old.iter().find(|o| o.0 == Engine::Esrgan && &o.1 == m).map_or(m == "realesr-animevideov3-x2" || m == "realesrgan-x4plus", |o| o.2);
            self.bench.models.push((Engine::Esrgan, m.clone(), on));
        }
        for m in &t.w2x_models {
            let on = old.iter().find(|o| o.0 == Engine::Waifu2x && &o.1 == m).map_or(m != "upconv_7_anime_style_art_rgb", |o| o.2);
            self.bench.models.push((Engine::Waifu2x, m.clone(), on));
        }
        let mut devs: Vec<(i32, String, bool)> = Vec::new();
        for (_, list) in &self.devices {
            for d in list {
                if !devs.iter().any(|x| x.0 == d.id) {
                    let on = self.bench.devices.iter().find(|o| o.0 == d.id).map_or(true, |o| o.2);
                    devs.push((d.id, d.name.clone(), on));
                }
            }
        }
        devs.sort_by_key(|d| d.0);
        self.bench.devices = devs;
        self.bench.open = true;
    }

    fn start_benchmark(&mut self) {
        let Some(t) = self.tools.clone() else { return };
        let mut specs = Vec::new();
        for (engine, model, on) in self.bench.models.clone() {
            if !on {
                continue;
            }
            let mut ai = nn::default_settings(&t);
            ai.engine = engine;
            let label = match engine {
                Engine::Esrgan => {
                    ai.model = model.clone();
                    format!("Real-ESRGAN {model}")
                }
                Engine::Waifu2x => {
                    ai.w2x_model = model.clone();
                    ai.w2x_scale = 2;
                    ai.w2x_noise = 1;
                    format!("waifu2x {model}")
                }
            };
            let avail_devs: Vec<i32> = self
                .devices_for(engine)
                .iter()
                .map(|d| d.id)
                .filter(|id| self.bench.devices.iter().any(|d| d.0 == *id && d.2))
                .collect();
            let mut devices = avail_devs;
            devices.sort();
            if !devices.is_empty() {
                specs.push(Spec { label, ai, devices });
            }
        }
        let heights: Vec<u32> = self.bench.heights.iter().filter(|h| h.1).map(|h| h.0).collect();
        if specs.is_empty() || heights.is_empty() {
            return;
        }
        let plan = Plan { specs, heights, timeout_secs: self.bench.timeout };
        self.bench.columns = self.bench.devices.iter().filter(|d| d.2).map(|d| (d.0, d.1.clone())).collect();
        self.bench.cells.clear();
        self.bench.done = 0;
        self.bench.total = plan.total();
        self.bench.started = Instant::now();
        self.bench.running = true;
        self.bench.cancel.store(false, Ordering::Relaxed);
        self.bench.plan = Some(plan.clone());
        let source = self.video.as_ref().and_then(|v| v.shown.as_ref().map(|f| (f.w as u32, f.h as u32, f.rgba.clone())));
        let (tx, rx) = channel();
        self.bench.rx = Some(rx);
        let (cancel, wake) = (self.bench.cancel.clone(), self.wake.clone());
        std::thread::spawn(move || {
            let w2 = wake.clone();
            bench::run(&t, &plan, source, &cancel, &tx, &move || w2());
        });
    }

    fn bench_table_text(&self, csv: bool) -> String {
        let Some(plan) = &self.bench.plan else { return String::new() };
        let cols = &self.bench.columns;
        let sep = if csv { "," } else { " | " };
        let mut out = String::new();
        let mut header = vec!["Modello".to_string(), "Risoluzione".to_string()];
        header.extend(cols.iter().map(|c| c.1.clone()));
        out.push_str(&header.join(sep));
        out.push('\n');
        if !csv {
            out.push_str(&vec!["---"; header.len()].join(sep));
            out.push('\n');
        }
        for (si, spec) in plan.specs.iter().enumerate() {
            for (ri, h) in plan.heights.iter().enumerate() {
                let mut row = vec![spec.label.clone(), format!("{h}p")];
                for (id, _) in cols {
                    row.push(match self.bench.cells.get(&(si, ri, *id)) {
                        Some(Cell { secs: Some(t), .. }) => format!("{t:.2} s"),
                        Some(c) => c.note.clone(),
                        None if spec.devices.contains(id) => "…".into(),
                        None => "-".into(),
                    });
                }
                out.push_str(&row.join(sep));
                out.push('\n');
            }
        }
        out
    }

    fn bench_window(&mut self, ctx: &egui::Context) {
        if !self.bench.open {
            return;
        }
        let mut open = true;
        egui::Window::new("Benchmark rete neurale").open(&mut open).default_size([820.0, 560.0]).show(ctx, |ui| {
            let running = self.bench.running;
            ui.add_enabled_ui(!running, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.label(RichText::new("Modelli").strong());
                    for e in Engine::ALL {
                        ui.separator();
                        ui.label(e.label());
                        for m in self.bench.models.iter_mut().filter(|m| m.0 == e) {
                            ui.checkbox(&mut m.2, &m.1);
                        }
                    }
                });
                ui.horizontal_wrapped(|ui| {
                    ui.label(RichText::new("Risoluzione").strong());
                    for h in &mut self.bench.heights {
                        ui.checkbox(&mut h.1, format!("{}p", h.0));
                    }
                    ui.separator();
                    ui.label(RichText::new("Processori").strong());
                    for d in &mut self.bench.devices {
                        ui.checkbox(&mut d.2, &d.1);
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Tempo massimo per prova");
                    ui.add(egui::DragValue::new(&mut self.bench.timeout).range(10..=600).suffix(" s"));
                    ui.small("Oltre il limite la prova è interrotta e le risoluzioni maggiori dello stesso modello sono saltate.");
                });
            });
            ui.horizontal(|ui| {
                if running {
                    let frac = if self.bench.total > 0 { self.bench.done as f32 / self.bench.total as f32 } else { 0.0 };
                    let elapsed = self.bench.started.elapsed().as_secs_f64();
                    let eta = if frac > 0.02 { format!("rimanente ~{}", fmt_dur(elapsed * (1.0 - frac as f64) / frac as f64)) } else { "rimanente: calcolo…".into() };
                    ui.add(egui::ProgressBar::new(frac).desired_width(260.0).text(format!("{}/{}  ·  {eta}", self.bench.done, self.bench.total)));
                    if ui.button("Annulla").clicked() {
                        self.bench.cancel.store(true, Ordering::Relaxed);
                    }
                } else {
                    let ready = self.bench_ready();
                    if ui.add_enabled(ready, egui::Button::new(RichText::new("Avvia benchmark").strong())).clicked() {
                        self.start_benchmark();
                    }
                    if !ready {
                        ui.small(if self.bench_rx.is_some() { "attendi la fine del test iniziale…" } else { "seleziona almeno un modello, una risoluzione e un processore" });
                    }
                    if self.bench.plan.is_some() {
                        if ui.button("Copia (Markdown)").clicked() {
                            ui.ctx().copy_text(self.bench_table_text(false));
                        }
                        if ui.button("Copia (CSV)").clicked() {
                            ui.ctx().copy_text(self.bench_table_text(true));
                        }
                    }
                }
            });
            ui.small(match self.video.as_ref().and_then(|v| v.shown.as_ref()) {
                Some(_) => "Immagine di prova: il fotogramma corrente, ridimensionato a ogni risoluzione.",
                None => "Immagine di prova: sintetica (apri un filmato per usare un suo fotogramma).",
            });
            ui.separator();
            if let Some(plan) = self.bench.plan.clone() {
                let cols = self.bench.columns.clone();
                egui::ScrollArea::both().auto_shrink([false, false]).show(ui, |ui| {
                    egui::Grid::new("bench_grid").striped(true).spacing([14.0, 4.0]).show(ui, |ui| {
                        ui.label(RichText::new("Modello").strong());
                        ui.label(RichText::new("Risoluzione").strong());
                        for c in &cols {
                            ui.label(RichText::new(&c.1).strong());
                        }
                        ui.end_row();
                        for (si, spec) in plan.specs.iter().enumerate() {
                            for (ri, h) in plan.heights.iter().enumerate() {
                                ui.label(&spec.label);
                                ui.label(format!("{h}p"));
                                // fastest processor of this row
                                let best = cols
                                    .iter()
                                    .filter_map(|(id, _)| self.bench.cells.get(&(si, ri, *id)).and_then(|c| c.secs).map(|t| (*id, t)))
                                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                                    .map(|b| b.0);
                                for (id, _) in &cols {
                                    match self.bench.cells.get(&(si, ri, *id)) {
                                        Some(Cell { secs: Some(t), .. }) => {
                                            let txt = format!("{t:.2} s");
                                            if best == Some(*id) && cols.len() > 1 {
                                                ui.label(RichText::new(txt).strong().color(Color32::LIGHT_GREEN));
                                            } else {
                                                ui.label(txt);
                                            }
                                        }
                                        Some(c) => {
                                            ui.colored_label(Color32::GRAY, &c.note);
                                        }
                                        None if spec.devices.contains(id) => {
                                            ui.label("…");
                                        }
                                        None => {
                                            ui.colored_label(Color32::GRAY, "-");
                                        }
                                    }
                                }
                                ui.end_row();
                            }
                        }
                    });
                });
            } else {
                ui.label("Misura il tempo per fotogramma di ogni modello, a diverse risoluzioni, su CPU e GPU.");
            }
        });
        self.bench.open = open;
        if !open && self.bench.running {
            self.bench.cancel.store(true, Ordering::Relaxed);
        }
        if self.bench.running {
            ctx.request_repaint_after(std::time::Duration::from_millis(300));
        }
    }

    fn bench_ready(&self) -> bool {
        self.bench_rx.is_none()
            && self.bench.models.iter().any(|m| m.2)
            && self.bench.heights.iter().any(|h| h.1)
            && self.bench.devices.iter().any(|d| d.2)
    }

    /// Modal-style progress window shown while an export runs.
    fn export_window(&mut self, ctx: &egui::Context) {
        let Some((frac, text, started)) = self.export_progress.clone() else { return };
        let elapsed = started.elapsed().as_secs_f64();
        // countdown between estimates: remaining time as of the last measurement minus the time passed since
        let eta = self.export_eta.map(|(v, at)| (v - at.elapsed().as_secs_f64()).max(0.0));
        egui::Window::new("Esportazione in corso")
            .collapsible(false)
            .resizable(false)
            .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
            .fixed_size([420.0, 0.0])
            .show(ctx, |ui| {
                ui.add(egui::ProgressBar::new(frac.clamp(0.0, 1.0)).show_percentage().desired_width(ui.available_width()).animate(true));
                ui.label(text);
                ui.horizontal(|ui| {
                    ui.label(format!("Trascorso {}", fmt_dur(elapsed)));
                    ui.separator();
                    ui.label(match eta {
                        Some(e) if e < 1.5 => "Rimanente: pochi istanti".to_string(),
                        Some(e) => format!("Rimanente ~{}", fmt_dur(e)),
                        None => "Rimanente: calcolo…".to_string(),
                    });
                });
                if ui.button("Annulla").clicked() {
                    self.export_cancel.store(true, Ordering::Relaxed);
                }
            });
        ctx.request_repaint_after(std::time::Duration::from_millis(250));
    }

    /// Log box docked at the bottom of the right column.
    fn log_panel(&mut self, ui: &mut egui::Ui) {
        let lines = if self.show_log { crate::log::snapshot() } else { Vec::new() };
        let header = |this: &mut App, ui: &mut egui::Ui, lines: &[String]| {
            ui.horizontal(|ui| {
                ui.toggle_value(&mut this.show_log, "Log").on_hover_text("Mostra / nascondi (tasto L)");
                if this.show_log {
                    if ui.small_button("Copia").clicked() {
                        ui.ctx().copy_text(lines.join("\n"));
                    }
                    if ui.small_button("Pulisci").clicked() {
                        crate::log::clear();
                    }
                    ui.small(format!("{} righe", lines.len()));
                }
            });
        };
        if !self.show_log {
            egui::Panel::bottom("log_min").show(ui, |ui| header(self, ui, &lines));
            return;
        }
        egui::Panel::bottom("log").resizable(true).default_size(190.0).size_range(60.0..=600.0).show(ui, |ui| {
            header(self, ui, &lines);
            egui::ScrollArea::both().stick_to_bottom(true).auto_shrink([false, false]).show_rows(ui, 14.0, lines.len(), |ui, range| {
                for l in &lines[range] {
                    let col = if l.contains("rror") || l.contains("errore") || l.contains("fallit") { Color32::LIGHT_RED } else { ui.visuals().text_color() };
                    ui.add(egui::Label::new(RichText::new(l).monospace().size(11.0).color(col)).extend());
                }
            });
        });
        ui.ctx().request_repaint_after(std::time::Duration::from_millis(400));
    }

    fn viewer(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default().frame(egui::Frame::NONE.fill(Color32::from_gray(16))).show(ui, |ui| {
            let rect = ui.available_rect_before_wrap();
            let resp = ui.allocate_rect(rect, Sense::click_and_drag());
            let painter = ui.painter_at(rect);

            if let Some(msg) = &self.prep_err {
                painter.text(rect.center(), Align2::CENTER_CENTER, format!("Errore: {msg}"), FontId::proportional(18.0), Color32::LIGHT_RED);
                return;
            }
            if self.tools.is_none() {
                painter.text(rect.center(), Align2::CENTER_CENTER, "Preparazione componenti (solo al primo avvio)…", FontId::proportional(18.0), Color32::GRAY);
                return;
            }
            let ai_active = self.ai_active();
            let ai_key = self.ai_key();
            let playing = self.pb.playing;
            let nearest = self.nearest;
            let Some(v) = &self.video else {
                painter.text(rect.center(), Align2::CENTER_CENTER, "Trascina qui un filmato\noppure premi «Apri filmato…»", FontId::proportional(22.0), Color32::GRAY);
                return;
            };

            let use_ai = ai_active && !playing && v.ai_key_matches(&ai_key);
            let tex = if use_ai { v.ai.as_ref().map(|a| &a.tex) } else { v.tex.as_ref() };
            if let Some(tex) = tex {
                let size = tex.size_vec2();
                let fit_scale = (rect.width() / size.x).min(rect.height() / size.y);
                let mut scale = if self.fit { fit_scale } else { self.zoom };
                if resp.hovered() {
                    let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                    if scroll != 0.0 {
                        let new = (scale * (1.0 + scroll / 400.0)).clamp(0.02, 64.0);
                        if let Some(c) = ui.input(|i| i.pointer.hover_pos()) {
                            let rel = c - rect.center();
                            self.pan = rel - (rel - self.pan) * (new / scale);
                        }
                        scale = new;
                        self.zoom = new;
                        self.fit = false;
                    }
                }
                if resp.double_clicked() {
                    self.fit = true;
                    self.pan = Vec2::ZERO;
                    scale = fit_scale;
                }
                if resp.dragged() {
                    self.pan += resp.drag_delta();
                    self.fit = false;
                    self.zoom = scale;
                }
                let img_rect = egui::Rect::from_center_size(rect.center() + if self.fit { Vec2::ZERO } else { self.pan }, size * scale);
                painter.image(tex.id(), img_rect, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), Color32::WHITE);

                let mut badge = format!("{}×{}  ({:.0}%)", size.x as u32, size.y as u32, scale * 100.0);
                if use_ai {
                    badge.push_str("  · AI");
                }
                let _ = nearest;
                painter.text(rect.left_top() + Vec2::new(8.0, 6.0), Align2::LEFT_TOP, badge, FontId::monospace(13.0), Color32::from_rgba_unmultiplied(255, 255, 255, 200));
            } else if v.loading {
                painter.text(rect.center(), Align2::CENTER_CENTER, "Caricamento…", FontId::proportional(18.0), Color32::GRAY);
            }
            // processing indicator: from a filter / frame change until the correct filtered frame is on screen
            let filters_pending = v.shown_gen != v.gen || (v.loading && (!playing || v.hard_wait));
            let ai_pending = ai_active && !playing && v.ai_err.is_none() && !v.ai_key_matches(&ai_key);
            if filters_pending || ai_pending {
                let label = if v.ai_busy && ai_pending && !filters_pending {
                    format!("Rete neurale… {:.0}s", v.ai_started.map_or(0.0, |t| t.elapsed().as_secs_f32()))
                } else if filters_pending {
                    "Applicazione filtri…".to_string()
                } else {
                    "Rete neurale…".to_string()
                };
                let galley = painter.layout_no_wrap(label, FontId::proportional(14.0), Color32::WHITE);
                let w = galley.size().x + 44.0;
                let pill = egui::Rect::from_min_size(egui::pos2(rect.right() - w - 10.0, rect.top() + 10.0), Vec2::new(w, 30.0));
                painter.rect_filled(pill, 15.0, Color32::from_rgba_unmultiplied(20, 20, 20, 215));
                painter.galley(egui::pos2(pill.left() + 36.0, pill.center().y - galley.size().y / 2.0), galley, Color32::WHITE);
                egui::Spinner::new().size(18.0).color(Color32::WHITE).paint_at(ui, egui::Rect::from_center_size(egui::pos2(pill.left() + 18.0, pill.center().y), Vec2::splat(18.0)));
            }
            let mut y = 24.0;
            let mut note = |text: String, col: Color32| {
                painter.text(rect.left_top() + Vec2::new(8.0, y), Align2::LEFT_TOP, text, FontId::proportional(14.0), col);
                y += 18.0;
            };
            if ai_active && playing {
                note("Rete neurale attiva solo a fotogramma fermo".into(), Color32::YELLOW);
            }
            if let Some(e) = &v.ai_err {
                note(format!("AI: {}", e.lines().next().unwrap_or("")), Color32::LIGHT_RED);
            }
            if let Some(e) = &v.error {
                for l in e.lines().take(6) {
                    note(l.to_string(), Color32::LIGHT_RED);
                }
            }
        });
    }
}

impl eframe::App for App {
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll(ctx);
        if self.tools.is_some() {
            self.handle_keys(ctx);
            self.advance_clock();

            // filter chain changed -> restart decoding at the current frame (not while a slider is being dragged)
            let c_held = !ctx.egui_wants_keyboard_input() && ctx.input(|i| i.key_down(Key::C) && !i.modifiers.command);
            self.c_held = c_held;
            // Debounce: a new chain is only applied once the pointer is released (slider let go) and the value has
            // been stable for a moment; every change restarts the wait.
            if self.pb.playing != self.prev_playing {
                self.prev_playing = self.pb.playing;
                if self.pb.playing {
                    if self.filters_on {
                        self.filters_on = false;
                        self.filters_resume = true;
                        self.force_apply = true;
                    }
                } else if self.filters_resume {
                    self.filters_on = true;
                    self.filters_resume = false;
                    self.force_apply = true;
                }
            }
            let vf = self.vf();
            if vf != self.vf_seen {
                self.vf_seen = vf.clone();
                self.vf_seen_at = Instant::now();
            }
            let settle = std::time::Duration::from_millis(120);
            let pointer_down = ctx.input(|i| i.pointer.any_down());
            if let Some(v) = &mut self.video {
                if v.vf != vf {
                    if self.force_apply || (!pointer_down && self.vf_seen_at.elapsed() >= settle) {
                        self.force_apply = false;
                        v.set_filters(vf);
                    } else {
                        ctx.request_repaint_after(std::time::Duration::from_millis(40));
                    }
                }
            }
            let tools = self.tools.clone().unwrap();
            let wake = self.wake.clone();
            let (nearest, playing) = (self.nearest, self.pb.playing);
            let dir: i8 = if !self.pb.playing { 0 } else if self.pb.reverse { -1 } else { 1 };
            let range = self.range();
            let ai_wanted = if self.ai_active() { Some((self.ai_key(), self.ai.clone())) } else { None };
            if let Some(v) = &mut self.video {
                v.fetch(&tools, &wake, ctx, dir, &range);
                v.update_texture(ctx, nearest);
                let settled = !playing && v.cur == v.want;
                v.drive_ai(&tools, &wake, ctx, ai_wanted, settled);
            }
            // filters are "being applied" from set_filters() until a frame of the new generation is on screen;
            // a failing chain (ffmpeg error) or a very slow one releases the lock so it can be corrected
            let applying = self.video.as_ref().map_or(false, |v| v.shown_gen != v.gen && v.error.is_none());
            if applying {
                let since = *self.busy_since.get_or_insert_with(Instant::now);
                self.filters_busy = since.elapsed().as_secs() < 30;
            } else {
                self.busy_since = None;
                self.filters_busy = false;
            }
            if self.pb.playing {
                ctx.request_repaint();
            }
        }
        if self.video.as_ref().map_or(false, |v| v.loading) {
            ctx.request_repaint_after(std::time::Duration::from_millis(30));
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.top_bar(ui);
        self.transport(ui);
        self.side_panel(ui);
        self.viewer(ui);
        self.export_window(&ui.ctx().clone());
        self.bench_window(&ui.ctx().clone());
    }
}

fn fmt_dur(s: f64) -> String {
    let t = s.max(0.0).round() as u64;
    if t >= 3600 {
        format!("{}:{:02}:{:02}", t / 3600, t / 60 % 60, t % 60)
    } else {
        format!("{}:{:02}", t / 60, t % 60)
    }
}

fn fmt_time(s: f64) -> String {
    let ms = (s * 1000.0).round() as u64;
    format!("{:02}:{:02}:{:02}.{:03}", ms / 3_600_000, ms / 60_000 % 60, ms / 1000 % 60, ms % 1000)
}

fn reveal(p: &std::path::Path) {
    #[cfg(windows)]
    {
        let _ = std::process::Command::new("explorer").arg(format!("/select,{}", p.display())).spawn();
    }
    #[cfg(target_os = "linux")]
    {
        let _ = std::process::Command::new("xdg-open").arg(p.parent().unwrap_or(p)).spawn();
    }
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("open").arg("-R").arg(p).spawn();
    }
}
