// src-tauri/src/main.rs
// ─────────────────────────────────────────────────────────────────────────────
// DeepFake Defense – desktop binary entry point.
// ─────────────────────────────────────────────────────────────────────────────

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    deepfake_defense_lib::run();
}
