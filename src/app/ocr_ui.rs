//! OCR user interface: side-panel section, overlay on the viewer, region selection and the background job plumbing.
use super::*;
use crate::ocr_job::{self, SharedEngine};
use crate::ocr_vote::{Hit, Kind};

#[derive(Clone, Copy, PartialEq)]
pub enum Mode {
    Text,
    Plates,
    Both,
}

#[derive(Clone, Copy, PartialEq)]
pub enum Scope {
    Frame,
    Cue,
    Next,
}

pub enum OcrMsg {
    Progress(u64, u64),
    Done(Result<Vec<Hit>, String>),
}

pub struct OcrState {
    engine: SharedEngine,
    mode: Mode,
    scope: Scope,
    next_n: u64,
    step: u64,
    use_ai: bool,
    auto: bool,
    /// region of interest, fractions of the frame
    roi: Option<[f32; 4]>,
    drag_from: Option<egui::Pos2>,
    rx: Option<Receiver<OcrMsg>>,
    cancel: Arc<AtomicBool>,
    progress: (u64, u64),
    started: Option<Instant>,
    hits: Vec<Hit>,
    selected: Option<usize>,
    min_conf: f32,
    only_valid: bool,
    status: Option<(String, bool)>,
    /// auto mode: frame being watched and since when, and the last frame already analysed
    auto_seen: (u64, Instant),
    auto_done: Option<u64>,
}

impl OcrState {
    pub fn new() -> OcrState {
        OcrState {
            engine: Arc::new(std::sync::Mutex::new(None)),
            mode: Mode::Both,
            scope: Scope::Frame,
            next_n: 30,
            step: 1,
            use_ai: false,
            auto: false,
            roi: None,
            drag_from: None,
            rx: None,
            cancel: Arc::new(AtomicBool::new(false)),
            progress: (0, 0),
            started: None,
            hits: Vec::new(),
            selected: None,
            min_conf: 0.4,
            only_valid: false,
            status: None,
            auto_seen: (u64::MAX, Instant::now()),
            auto_done: None,
        }
    }

    /// Forget results (a different video was opened).
    pub fn clear(&mut self) {
        self.hits.clear();
        self.selected = None;
        self.roi = None;
        self.status = None;
        self.auto_done = None;
    }

    fn visible(&self, h: &Hit) -> bool {
        h.conf >= self.min_conf && (!self.only_valid || h.kind == Kind::Text || h.format.is_some())
    }
}

fn hit_rect_at(h: &Hit, cur: u64, slack: u64) -> Option<crate::ocr::Rect> {
    if cur < h.first || cur > h.last + slack {
        return None;
    }
    h.trail.iter().rev().find(|(f, _)| *f <= cur).or(h.trail.first()).map(|t| t.1)
}

fn hit_color(h: &Hit) -> Color32 {
    match (h.kind, h.format.is_some()) {
        (Kind::Plate, true) => Color32::from_rgb(90, 220, 110),
        (Kind::Plate, false) => Color32::from_rgb(255, 170, 60),
        (Kind::Text, _) => Color32::from_rgb(80, 200, 255),
    }
}

/// Draws the results and the region of interest over the picture, and handles Shift+drag (region) and clicks (copy).
pub fn viewer_overlay(st: &mut OcrState, ui: &egui::Ui, painter: &egui::Painter, resp: &egui::Response, img_rect: egui::Rect, cur: u64) {
    let to_screen = |x: f32, y: f32| egui::pos2(img_rect.left() + x * img_rect.width(), img_rect.top() + y * img_rect.height());
    let shift = ui.input(|i| i.modifiers.shift);
    if shift && resp.drag_started() {
        st.drag_from = ui.input(|i| i.pointer.press_origin());
    }
    if let (Some(a), true) = (st.drag_from, resp.dragged() || resp.drag_stopped()) {
        if let Some(b) = ui.input(|i| i.pointer.latest_pos()) {
            let r = egui::Rect::from_two_pos(a, b).intersect(img_rect);
            if resp.drag_stopped() {
                st.drag_from = None;
                if r.width() > 12.0 && r.height() > 12.0 {
                    st.roi = Some([
                        (r.left() - img_rect.left()) / img_rect.width(),
                        (r.top() - img_rect.top()) / img_rect.height(),
                        (r.right() - img_rect.left()) / img_rect.width(),
                        (r.bottom() - img_rect.top()) / img_rect.height(),
                    ]);
                }
            } else {
                painter.rect_stroke(r, 0.0, egui::Stroke::new(2.0, Color32::YELLOW), egui::StrokeKind::Middle);
            }
        }
    }
    if let Some(r) = st.roi {
        let rr = egui::Rect::from_min_max(to_screen(r[0], r[1]), to_screen(r[2], r[3]));
        painter.rect_stroke(rr, 0.0, egui::Stroke::new(1.5, Color32::YELLOW), egui::StrokeKind::Middle);
        painter.text(rr.left_top() + Vec2::new(3.0, -2.0), Align2::LEFT_BOTTOM, "area OCR", FontId::proportional(11.0), Color32::YELLOW);
    }
    let slack = st.step.max(1) * 2;
    let click = if resp.clicked() { ui.input(|i| i.pointer.interact_pos()) } else { None };
    let mut clicked_hit = None;
    for (i, h) in st.hits.iter().enumerate() {
        if !st.visible(h) {
            continue;
        }
        let Some(r) = hit_rect_at(h, cur, slack) else { continue };
        let rr = egui::Rect::from_min_max(to_screen(r.x0, r.y0), to_screen(r.x1, r.y1));
        let col = hit_color(h);
        let sel = st.selected == Some(i);
        painter.rect_stroke(rr, 0.0, egui::Stroke::new(if sel { 3.0 } else { 1.5 }, if sel { Color32::YELLOW } else { col }), egui::StrokeKind::Middle);
        let label = format!("{}  {:.0}%", h.text, h.conf * 100.0);
        let galley = painter.layout_no_wrap(label, FontId::proportional(13.0), Color32::BLACK);
        let tag = egui::Rect::from_min_size(egui::pos2(rr.left(), rr.top() - galley.size().y - 4.0), galley.size() + Vec2::new(8.0, 4.0));
        painter.rect_filled(tag, 3.0, col);
        painter.galley(tag.min + Vec2::new(4.0, 2.0), galley, Color32::BLACK);
        if click.is_some_and(|p| rr.contains(p) || tag.contains(p)) {
            clicked_hit = Some(i);
        }
    }
    if let Some(i) = clicked_hit {
        st.selected = Some(i);
        ui.ctx().copy_text(st.hits[i].text.clone());
        st.status = Some((format!("Copiato: {}", st.hits[i].text), false));
    }
}

impl App {
    fn ocr_ready(&self) -> bool {
        self.tools.as_ref().and_then(|t| t.ocr_dir.as_deref()).is_some_and(crate::ocr::available)
    }

    /// Panning with the mouse is suspended while the region of interest is being drawn (Shift held).
    pub(super) fn ocr_idle_drag(&self, ui: &egui::Ui) -> bool {
        self.ocr.drag_from.is_none() && !ui.input(|i| i.modifiers.shift)
    }

    /// Starts a job. `frame_only` forces the current frame regardless of the chosen scope.
    pub(super) fn ocr_start(&mut self, frame_only: bool) {
        if !self.ocr_ready() || self.ocr.rx.is_some() {
            return;
        }
        let (Some(tools), Some(v)) = (self.tools.clone(), self.video.as_ref()) else { return };
        let (cur, fps, path) = (v.cur, v.info.fps, v.path.clone());
        let last = v.last_frame().unwrap_or(cur);
        let (start, count) = if frame_only {
            (cur, 1)
        } else {
            match self.ocr.scope {
                Scope::Frame => (cur, 1),
                Scope::Next => (cur, self.ocr.next_n.max(1).min(last + 1 - cur)),
                Scope::Cue => match self.cue {
                    Some((a, b)) => (a, b.saturating_sub(a) + 1),
                    None => {
                        self.ocr.status = Some(("Nessun intervallo CUE selezionato (tasti I / O)".into(), true));
                        return;
                    }
                },
            }
        };
        let filters_enabled = self.filters_on || self.filters_resume;
        let ai = if self.ocr.use_ai && self.ai.enabled && !self.devices_for(self.ai.engine).is_empty() { Some(self.ai.clone()) } else { None };
        let spec = ocr_job::Spec {
            video: path,
            fps,
            vf: if filters_enabled { filters::chain(&self.filters) } else { String::new() },
            ai,
            start,
            count,
            step: if count > 1 { self.ocr.step.max(1) } else { 1 },
            roi: self.ocr.roi,
            text: self.ocr.mode != Mode::Plates,
            plates: self.ocr.mode != Mode::Text,
        };
        self.pb.playing = false;
        let (tx, rx) = channel();
        let (wake, engine) = (self.wake.clone(), self.ocr.engine.clone());
        self.ocr.cancel.store(false, Ordering::Relaxed);
        let cancel = self.ocr.cancel.clone();
        self.ocr.rx = Some(rx);
        self.ocr.progress = (0, 0);
        self.ocr.started = Some(Instant::now());
        self.ocr.status = None;
        self.ocr.auto_done = Some(cur);
        std::thread::spawn(move || {
            let (tx2, w2) = (tx.clone(), wake.clone());
            let res = ocr_job::run(&tools, &engine, &spec, &cancel, &move |d, t| {
                let _ = tx2.send(OcrMsg::Progress(d, t));
                w2();
            });
            let _ = tx.send(OcrMsg::Done(res));
            wake();
        });
    }

    pub(super) fn ocr_poll(&mut self) {
        let mut finished = false;
        if let Some(rx) = &self.ocr.rx {
            while let Ok(m) = rx.try_recv() {
                match m {
                    OcrMsg::Progress(d, t) => self.ocr.progress = (d, t),
                    OcrMsg::Done(Ok(hits)) => {
                        let secs = self.ocr.started.map_or(0.0, |s| s.elapsed().as_secs_f32());
                        let plates = hits.iter().filter(|h| h.kind == Kind::Plate).count();
                        self.ocr.status = Some((format!("{} risultati ({} targhe) in {:.1}s", hits.len(), plates, secs), false));
                        self.ocr.hits = hits;
                        self.ocr.selected = None;
                        finished = true;
                    }
                    OcrMsg::Done(Err(e)) => {
                        self.ocr.status = Some((e, true));
                        finished = true;
                    }
                }
            }
        }
        if finished {
            self.ocr.rx = None;
        }
    }

    /// Automatic OCR when the picture has been still for a moment.
    pub(super) fn ocr_auto(&mut self, ctx: &egui::Context) {
        if !self.ocr.auto || self.pb.playing || self.ocr.rx.is_some() || !self.ocr_ready() {
            return;
        }
        let Some(v) = &self.video else { return };
        if v.cur != v.want || v.loading || v.shown_gen != v.gen {
            return;
        }
        let cur = v.cur;
        if self.ocr.auto_seen.0 != cur {
            self.ocr.auto_seen = (cur, Instant::now());
        }
        if self.ocr.auto_done == Some(cur) {
            return;
        }
        if self.ocr.auto_seen.1.elapsed() >= std::time::Duration::from_millis(400) {
            self.ocr_start(true);
        } else {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
    }

    fn ocr_csv(&self) -> String {
        let q = |s: &str| format!("\"{}\"", s.replace('"', "\"\""));
        let mut out = String::from("tipo,testo,confidenza,formato,frame_inizio,frame_fine,frame_migliore,tempo_s,letture,alternative\n");
        let fps = self.video.as_ref().map_or(25.0, |v| v.info.fps);
        for h in self.ocr.hits.iter().filter(|h| self.ocr.visible(h)) {
            out.push_str(&format!(
                "{},{},{:.3},{},{},{},{},{:.3},{},{}\n",
                if h.kind == Kind::Plate { "targa" } else { "testo" },
                q(&h.text),
                h.conf,
                q(h.format.unwrap_or("")),
                h.first,
                h.last,
                h.best,
                h.best as f64 / fps,
                h.readings,
                q(&h.alts.join(" | "))
            ));
        }
        out
    }

    pub(super) fn ocr_section(&mut self, ui: &mut egui::Ui) {
        ui.heading("OCR");
        if !self.ocr_ready() {
            ui.label("Modelli OCR non inclusi in questa build.");
            return;
        }
        let busy = self.ocr.rx.is_some();
        let have_video = self.video.is_some();
        ui.add_enabled_ui(!busy, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.ocr.mode, Mode::Text, "Testo");
                ui.selectable_value(&mut self.ocr.mode, Mode::Plates, "Targhe");
                ui.selectable_value(&mut self.ocr.mode, Mode::Both, "Entrambi");
            });
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.ocr.scope, Scope::Frame, "Fotogramma");
                ui.selectable_value(&mut self.ocr.scope, Scope::Cue, "Intervallo CUE");
                ui.selectable_value(&mut self.ocr.scope, Scope::Next, "Serie");
            });
            if self.ocr.scope == Scope::Next {
                ui.horizontal(|ui| {
                    ui.label("Fotogrammi:");
                    ui.add(egui::DragValue::new(&mut self.ocr.next_n).range(2..=100_000));
                });
            }
            if self.ocr.scope != Scope::Frame {
                ui.horizontal(|ui| {
                    ui.label("Analizza uno ogni:");
                    ui.add(egui::DragValue::new(&mut self.ocr.step).range(1..=1000))
                        .on_hover_text("Più fotogrammi = voto più affidabile, ma più lento (circa 1 s per fotogramma Full HD su CPU).");
                });
            }
            ui.horizontal(|ui| {
                let ai_ok = self.ai.enabled && !self.devices_for(self.ai.engine).is_empty();
                ui.add_enabled(ai_ok, egui::Checkbox::new(&mut self.ocr.use_ai, "Usa fotogramma potenziato (AI)"))
                    .on_hover_text("Passa ogni fotogramma dalla rete neurale prima della lettura: aiuta su scritte piccole e sfocate, ma è molto più lento. Richiede l'AI attiva nei filtri.");
            });
            ui.checkbox(&mut self.ocr.auto, "OCR automatico a fotogramma fermo");
            ui.horizontal(|ui| {
                match self.ocr.roi {
                    Some(_) => {
                        ui.label("Area: selezionata");
                        if ui.small_button("Tutto il fotogramma").clicked() {
                            self.ocr.roi = None;
                        }
                    }
                    None => {
                        ui.label("Area: tutto il fotogramma");
                        ui.weak("(Maiusc+trascina sull'immagine per limitarla)");
                    }
                };
            });
        });
        ui.horizontal(|ui| {
            if busy {
                let (d, t) = self.ocr.progress;
                ui.add(egui::ProgressBar::new(if t > 0 { d as f32 / t as f32 } else { 0.0 }).desired_width(ui.available_width() - 80.0).text(format!("{d}/{t}")));
                if ui.button("Annulla").clicked() {
                    self.ocr.cancel.store(true, Ordering::Relaxed);
                }
            } else {
                ui.add_enabled_ui(have_video, |ui| {
                    if ui.add_sized([ui.available_width(), 30.0], egui::Button::new(RichText::new("Leggi (T)").strong())).clicked() {
                        self.ocr_start(false);
                    }
                });
            }
        });
        if let Some((s, err)) = &self.ocr.status {
            ui.colored_label(if *err { Color32::LIGHT_RED } else { Color32::LIGHT_GREEN }, s);
        }
        if self.ocr.hits.is_empty() {
            return;
        }
        ui.separator();
        ui.horizontal(|ui| {
            ui.label("Confidenza minima");
            ui.add(egui::Slider::new(&mut self.ocr.min_conf, 0.0..=1.0).custom_formatter(|v, _| format!("{:.0}%", v * 100.0)));
        });
        ui.checkbox(&mut self.ocr.only_valid, "Solo targhe con formato valido");
        let fps = self.video.as_ref().map_or(25.0, |v| v.info.fps);
        let mut goto = None;
        let mut copy = None;
        let rows: Vec<usize> = (0..self.ocr.hits.len()).filter(|&i| self.ocr.visible(&self.ocr.hits[i])).collect();
        egui::ScrollArea::vertical().id_salt("ocr_hits").max_height(260.0).auto_shrink([false, true]).show(ui, |ui| {
            for &i in &rows {
                let h = &self.ocr.hits[i];
                ui.horizontal(|ui| {
                    let tag = if h.kind == Kind::Plate { "TARGA" } else { "testo" };
                    ui.label(RichText::new(tag).small().color(hit_color(h)));
                    let r = ui.selectable_label(self.ocr.selected == Some(i), RichText::new(&h.text).monospace()).on_hover_ui(|ui| {
                        if let Some(f) = h.format {
                            ui.label(format!("Formato: {f}"));
                        }
                        ui.label(format!("{} letture, fotogrammi {}–{} (migliore: {})", h.readings, h.first, h.last, h.best));
                        if !h.alts.is_empty() {
                            ui.label(format!("Altre letture: {}", h.alts.join(", ")));
                        }
                        ui.small("Clic: vai al fotogramma · clic destro: copia");
                    });
                    if r.clicked() {
                        goto = Some(i);
                    }
                    if r.secondary_clicked() {
                        copy = Some(h.text.clone());
                    }
                    ui.weak(format!("{:.0}%  {}", h.conf * 100.0, fmt_time(h.best as f64 / fps)));
                });
            }
        });
        if let Some(i) = goto {
            self.ocr.selected = Some(i);
            let best = self.ocr.hits[i].best;
            self.pb.playing = false;
            self.seek(best);
        }
        if let Some(t) = copy {
            ui.ctx().copy_text(t);
        }
        ui.horizontal(|ui| {
            if ui.button("Copia tutto").clicked() {
                let all: Vec<&str> = rows.iter().map(|&i| self.ocr.hits[i].text.as_str()).collect();
                ui.ctx().copy_text(all.join("\n"));
                self.ocr.status = Some((format!("Copiati {} risultati", all.len()), false));
            }
            if ui.button("Salva CSV").clicked() {
                match (self.effective_out_dir(), self.video.as_ref()) {
                    (Some(dir), Some(v)) => {
                        let stem = v.path.file_stem().and_then(|s| s.to_str()).unwrap_or("video");
                        let p = dir.join(format!("{stem}_ocr.csv"));
                        self.ocr.status = Some(match std::fs::write(&p, self.ocr_csv()) {
                            Ok(()) => {
                                self.last_export = Some(p.clone());
                                (format!("Salvato {}", p.display()), false)
                            }
                            Err(e) => (format!("CSV non salvato: {e}"), true),
                        });
                    }
                    _ => self.ocr.status = Some(("Scegli prima la cartella di esportazione".into(), true)),
                }
            }
        });
    }
}
