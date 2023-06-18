#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use eframe::egui;
use serde_derive::*;
use ysc_utils::shared;

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "ysc-utils (gui)",
        native_options,
        Box::new(|cc| Box::new(Gui::new(cc))),
    )
}

#[derive(Default, Serialize, Deserialize)]
pub struct Gui {
    old_script: String,
    new_script: String,
    input: String,
    #[serde(skip)]
    output: String,
}

impl Gui {
    /// Called once before the first frame.
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        if std::path::Path::exists("./settings.json".as_ref()) {
            return match std::fs::read_to_string("./settings.json") {
                Ok(content) => {
                    match serde_json::from_str::<Self>(&content) {
                        Ok(s) => s,
                        _ => Default::default()
                    }
                }
                Err(_) => Default::default()
            }

        }
        Default::default()
    }
}

impl eframe::App for Gui {
    /// Called each time the UI needs repainting, which may be many times per second.
    /// Put your widgets into a `SidePanel`, `TopPanel`, `CentralPanel`, `Window` or `Area`.
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            // The central panel the region left after adding TopPanel's and SidePanel's

            ui.heading("ysc utils -- gui");

            ui.horizontal(|ui| {
                ui.label("Old .ysc* script");
                ui.text_edit_singleline(&mut self.old_script);
                if ui.button("Browse").clicked() {
                    match tinyfiledialogs::open_file_dialog("Old script", "~", None) {
                        Some(file) => {
                            self.old_script = file;
                        }
                        None => {}
                    }
                }
            });

            ui.horizontal(|ui| {
                ui.label("New .ysc* script");
                ui.text_edit_singleline(&mut self.new_script);
                if ui.button("Browse").clicked() {
                    match tinyfiledialogs::open_file_dialog("New script", "~", None) {
                        Some(file) => {
                            self.new_script = file;
                        }
                        None => {}
                    }
                }
            });
            ui.horizontal(|ui| {
                ui.label("Input");
                let re = ui.text_edit_singleline(&mut self.input);
                if ui.button("Go").clicked() || re.lost_focus() {
                    match serde_json::to_string(&self) {
                        Ok(json) => {
                            let _ = std::fs::write("./settings.json", &json).is_ok();
                        }
                        Err(_) => {}
                    }

                    match shared::update_global(&self.old_script, &self.new_script, &self.input) {
                        Ok(val) => {
                            self.output = val;
                        }
                        Err(text) => {
                            self.output = format!("Error: {}", text);
                        }
                    }

                }
            });
            ui.horizontal(|ui| {
                ui.label("Output");
                ui.text_edit_singleline(&mut self.output);
                if ui.button("Copy to clipboard").clicked() {
                    // todo
                }
            });

        });
    }
}