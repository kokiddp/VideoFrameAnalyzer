use eframe::egui::{self, Color32, Pos2, Sense, Stroke, Vec2};

/// Rotary knob mapping a drag onto a logarithmic range. Returns true when the value changed.
pub fn knob(ui: &mut egui::Ui, value: &mut f32, min: f32, max: f32) -> bool {
    let size = 64.0;
    let (rect, resp) = ui.allocate_exact_size(Vec2::splat(size), Sense::click_and_drag());
    let (lmin, lmax) = (min.ln(), max.ln());
    let mut t = ((value.max(min).ln() - lmin) / (lmax - lmin)).clamp(0.0, 1.0);
    let mut changed = false;
    if resp.dragged() {
        let d = resp.drag_delta();
        t = (t + (d.x - d.y) / 220.0).clamp(0.0, 1.0);
        *value = (lmin + t * (lmax - lmin)).exp();
        changed = true;
    }
    if resp.double_clicked() {
        *value = 1.0;
        t = (0.0 - lmin) / (lmax - lmin);
        changed = true;
    }
    if resp.hovered() {
        let s = ui.input(|i| i.smooth_scroll_delta.y);
        if s != 0.0 {
            t = (t + s / 1500.0).clamp(0.0, 1.0);
            *value = (lmin + t * (lmax - lmin)).exp();
            changed = true;
        }
    }
    let center = rect.center();
    let r = size * 0.38;
    let p = ui.painter();
    let vis = ui.visuals();
    let a0 = std::f32::consts::PI * 0.75;
    let sweep = std::f32::consts::PI * 1.5;
    let arc = |from: f32, to: f32, col: Color32, w: f32| {
        let n = 40;
        let pts: Vec<Pos2> = (0..=n)
            .map(|i| {
                let a = a0 + (from + (to - from) * i as f32 / n as f32) * sweep;
                center + Vec2::new(a.cos(), a.sin()) * (r + 6.0)
            })
            .collect();
        p.add(egui::Shape::line(pts, Stroke::new(w, col)));
    };
    arc(0.0, 1.0, vis.widgets.noninteractive.bg_stroke.color, 3.0);
    arc(0.0, t, vis.selection.bg_fill, 3.0);
    p.circle(center, r, vis.widgets.inactive.bg_fill, Stroke::new(1.5, vis.widgets.inactive.fg_stroke.color));
    let ang = a0 + t * sweep;
    let dir = Vec2::new(ang.cos(), ang.sin());
    p.line_segment([center + dir * (r * 0.35), center + dir * (r * 0.95)], Stroke::new(3.0, vis.strong_text_color()));
    resp.on_hover_text("Trascina (su/destra = più veloce) o usa la rotella. Doppio clic: 1×");
    changed
}

#[derive(Clone, Copy, PartialEq)]
pub enum Icon {
    First,
    Last,
    Play,
    PlayRev,
    Pause,
    StepBack,
    StepFwd,
}

/// Transport button with a vector-drawn icon (no dependency on font glyphs).
pub fn icon_button(ui: &mut egui::Ui, icon: Icon, tip: &str) -> bool {
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(38.0, 30.0), Sense::click());
    let vis = ui.style().interact(&resp);
    let p = ui.painter();
    p.rect(rect, 4.0, vis.bg_fill, vis.bg_stroke, egui::StrokeKind::Inside);
    let col = vis.fg_stroke.color;
    let c = rect.center();
    let h = 7.0; // half height of glyphs
    let w = 6.0; // triangle width
    // right-pointing triangle with its left edge at x
    let tri = |x: f32, right: bool| {
        let (a, b) = if right { (x, x + w) } else { (x + w, x) };
        p.add(egui::Shape::convex_polygon(
            vec![Pos2::new(a, c.y - h), Pos2::new(b, c.y), Pos2::new(a, c.y + h)],
            col,
            Stroke::NONE,
        ));
    };
    let bar = |x: f32, wd: f32| {
        p.rect_filled(egui::Rect::from_min_size(Pos2::new(x, c.y - h), Vec2::new(wd, 2.0 * h)), 0.0, col);
    };
    match icon {
        Icon::Play => tri(c.x - 4.0, true),
        Icon::PlayRev => tri(c.x - 2.0, false),
        Icon::Pause => {
            bar(c.x - 5.5, 3.5);
            bar(c.x + 2.0, 3.5);
        }
        Icon::First => {
            bar(c.x - 9.0, 2.5);
            tri(c.x - 5.0, false);
            tri(c.x + 1.0, false);
        }
        Icon::Last => {
            tri(c.x - 7.0, true);
            tri(c.x - 1.0, true);
            bar(c.x + 7.0, 2.5);
        }
        Icon::StepBack => {
            bar(c.x - 6.0, 2.5);
            tri(c.x - 2.0, false);
        }
        Icon::StepFwd => {
            tri(c.x - 4.0, true);
            bar(c.x + 4.0, 2.5);
        }
    }
    resp.on_hover_text(tip).clicked()
}

#[derive(Clone, Copy)]
enum CueDrag {
    New(u64),
    Start,
    End,
}

#[derive(Default)]
pub struct BarsOut {
    /// New playhead position requested by the user.
    pub seek: Option<u64>,
    /// Changed CUE selection (`Some(None)` = cleared).
    pub cue: Option<Option<(u64, u64)>>,
}

/// CUE-range bar stacked on top of the seek bar; both share the same horizontal scale.
pub fn range_bars(ui: &mut egui::Ui, pos: u64, max: u64, cue: Option<(u64, u64)>) -> BarsOut {
    let mut out = BarsOut::default();
    let max = max.max(1);
    let width = (ui.available_width() - 6.0).max(100.0);
    let (cue_rect, cue_resp) = ui.allocate_exact_size(Vec2::new(width, 16.0), Sense::click_and_drag());
    let (bar_rect, bar_resp) = ui.allocate_exact_size(Vec2::new(width, 24.0), Sense::click_and_drag());
    let x_of = |f: u64| cue_rect.left() + (f as f32 / max as f32).min(1.0) * cue_rect.width();
    let frame_at = |x: f32| (((x - cue_rect.left()) / cue_rect.width()).clamp(0.0, 1.0) * max as f32).round() as u64;

    // ---- CUE bar interaction
    let id = cue_resp.id;
    if cue_resp.drag_started() {
        if let Some(p) = cue_resp.interact_pointer_pos() {
            let f = frame_at(p.x);
            let mode = match cue {
                Some((a, _)) if (x_of(a) - p.x).abs() < 7.0 => CueDrag::Start,
                Some((_, b)) if (x_of(b) - p.x).abs() < 7.0 => CueDrag::End,
                _ => CueDrag::New(f),
            };
            ui.data_mut(|d| d.insert_temp(id, mode));
        }
    }
    if cue_resp.dragged() {
        if let (Some(p), Some(mode)) = (cue_resp.interact_pointer_pos(), ui.data(|d| d.get_temp::<CueDrag>(id))) {
            let g = frame_at(p.x);
            out.cue = Some(Some(match (mode, cue) {
                (CueDrag::New(a), _) => (a.min(g), a.max(g)),
                (CueDrag::Start, Some((_, b))) => (g.min(b), b),
                (CueDrag::End, Some((a, _))) => (a, g.max(a)),
                (_, c) => c.unwrap_or((g, g)),
            }));
        }
    } else if cue_resp.clicked() {
        out.cue = Some(None);
    }
    let shown_cue = match out.cue {
        Some(c) => c,
        None => cue,
    };

    // ---- seek bar interaction
    if bar_resp.dragged() || bar_resp.clicked() {
        if let Some(p) = bar_resp.interact_pointer_pos() {
            out.seek = Some(frame_at(p.x));
        }
    }

    // ---- painting
    let vis = ui.visuals();
    let accent = vis.selection.bg_fill;
    let p = ui.painter();
    p.rect_filled(cue_rect, 3.0, vis.extreme_bg_color);
    p.rect_stroke(cue_rect, 3.0, vis.widgets.noninteractive.bg_stroke, egui::StrokeKind::Inside);
    if shown_cue.is_none() {
        p.text(cue_rect.center(), egui::Align2::CENTER_CENTER, "CUE: trascina per selezionare un intervallo (I / O = punti di ingresso e uscita)",
            egui::FontId::proportional(10.5), vis.weak_text_color());
    }
    if let Some((a, b)) = shown_cue {
        let r = egui::Rect::from_min_max(Pos2::new(x_of(a), cue_rect.top()), Pos2::new(x_of(b).max(x_of(a) + 2.0), cue_rect.bottom()));
        p.rect_filled(r, 2.0, accent.gamma_multiply(0.85));
        for x in [r.left(), r.right()] {
            p.line_segment([Pos2::new(x, cue_rect.top() - 1.0), Pos2::new(x, cue_rect.bottom() + 1.0)], Stroke::new(3.0, Color32::from_rgb(255, 200, 60)));
        }
    }
    p.rect_filled(bar_rect.shrink2(Vec2::new(0.0, 7.0)), 3.0, vis.extreme_bg_color);
    if let Some((a, b)) = shown_cue {
        let r = egui::Rect::from_min_max(Pos2::new(x_of(a), bar_rect.top() + 7.0), Pos2::new(x_of(b), bar_rect.bottom() - 7.0));
        p.rect_filled(r, 2.0, accent.gamma_multiply(0.45));
    }
    let px = x_of(pos);
    let filled = egui::Rect::from_min_max(Pos2::new(bar_rect.left(), bar_rect.top() + 7.0), Pos2::new(px, bar_rect.bottom() - 7.0));
    p.rect_filled(filled, 3.0, vis.widgets.inactive.fg_stroke.color.gamma_multiply(0.35));
    p.line_segment([Pos2::new(px, bar_rect.top() + 1.0), Pos2::new(px, bar_rect.bottom() - 1.0)], Stroke::new(2.5, vis.strong_text_color()));
    p.circle_filled(Pos2::new(px, bar_rect.center().y), 6.0, vis.strong_text_color());
    cue_resp.on_hover_text("Trascina: seleziona l'intervallo · trascina i bordi gialli: regola · clic: deseleziona");
    out
}

use crate::filters::{Filter, Kind};

pub const LABEL_WIDTH: f32 = 70.0;

/// Slider width that makes `label | slider | value box | Reset` fit exactly inside a card of width `cw`.
/// (value box ~62 px, Reset ~48 px, four 8 px gaps)
pub fn slider_width(cw: f32) -> f32 {
    (cw - LABEL_WIDTH - 62.0 - 48.0 - 32.0).clamp(60.0, 420.0)
}

/// One filter as a fixed-width card. Every row has constant content (the Reset button is always present,
/// merely disabled at the default value) so that moving a slider can never change the layout width.
pub fn filter_card(ui: &mut egui::Ui, f: &mut Filter, cw: f32, slider_w: f32) {
    let stroke = if f.enabled { Stroke::new(1.5, ui.visuals().selection.bg_fill) } else { ui.visuals().widgets.noninteractive.bg_stroke };
    egui::Frame::group(ui.style()).stroke(stroke).show(ui, |ui| {
        ui.set_width(cw);
        ui.set_max_width(cw);
        ui.checkbox(&mut f.enabled, egui::RichText::new(f.name).strong()).on_hover_text(f.help);
        if f.enabled {
            let id = f.id;
            for p in &mut f.params {
                ui.horizontal(|ui| {
                    ui.add_sized([LABEL_WIDTH, 18.0], egui::Label::new(p.label));
                    match p.kind.clone() {
                        Kind::Float { min, max } => {
                            ui.spacing_mut().slider_width = slider_w;
                            ui.add(egui::Slider::new(&mut p.value, min..=max).clamping(egui::SliderClamping::Always).fixed_decimals(2));
                        }
                        Kind::Choice(opts) => {
                            let mut idx = p.value as usize;
                            egui::ComboBox::from_id_salt((id, p.label))
                                .width(slider_w)
                                .truncate()
                                .selected_text(opts[idx.min(opts.len() - 1)])
                                .show_ui(ui, |ui| {
                                    for (k, o) in opts.iter().enumerate() {
                                        ui.selectable_value(&mut idx, k, *o);
                                    }
                                });
                            p.value = idx as f32;
                        }
                    }
                    let changed = p.value != p.default;
                    if ui.add_enabled(changed, egui::Button::new("Reset").small()).on_hover_text("Ripristina il valore predefinito").clicked() {
                        p.value = p.default;
                    }
                });
            }
        }
    });
}
