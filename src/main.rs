#![cfg_attr(all(not(debug_assertions), windows), windows_subsystem = "windows")]

mod app;
mod bench;
mod decoder;
mod export;
mod filters;
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
mod video;
mod widgets;

fn main() -> eframe::Result {
    let initial = std::env::args_os().nth(1).map(std::path::PathBuf::from);
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1360.0, 820.0])
            .with_min_inner_size([900.0, 560.0])
            .with_title("Video Frame Analyzer")
            .with_drag_and_drop(true),
        ..Default::default()
    };
    eframe::run_native("Video Frame Analyzer", options, Box::new(move |cc| Ok(Box::new(app::App::new(cc, initial)))))
}
