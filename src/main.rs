#![cfg_attr(all(not(debug_assertions), windows), windows_subsystem = "windows")]

mod app;
mod bench;
mod decoder;
mod export;
mod filters;
mod i18n;
mod log;
mod nn;
mod ocr;
mod ocr_job;
mod ocr_vote;
mod player;
mod probe;
#[cfg(test)]
mod selftest;
mod tools;
mod update;
mod video;
mod widgets;

fn main() -> eframe::Result {
    i18n::init();
    let initial = std::env::args_os().nth(1).map(std::path::PathBuf::from);
    let icon = image::load_from_memory_with_format(include_bytes!("../icon/icon-256.png"), image::ImageFormat::Png)
        .ok()
        .map(|i| {
            let i = i.into_rgba8();
            eframe::egui::IconData { width: i.width(), height: i.height(), rgba: i.into_raw() }
        });
    let mut viewport = eframe::egui::ViewportBuilder::default()
            .with_inner_size([1360.0, 820.0])
            .with_min_inner_size([900.0, 560.0])
            .with_title("Video Frame Analyzer")
            .with_drag_and_drop(true);
    if let Some(icon) = icon {
        viewport = viewport.with_icon(icon);
    }
    let options = eframe::NativeOptions { viewport, ..Default::default() };
    eframe::run_native("Video Frame Analyzer", options, Box::new(move |cc| Ok(Box::new(app::App::new(cc, initial)))))
}
