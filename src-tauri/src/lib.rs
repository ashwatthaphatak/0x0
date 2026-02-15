// src-tauri/src/lib.rs
// ─────────────────────────────────────────────────────────────────────────────
// Vaxel – shared Tauri app entry for desktop/mobile.
// ─────────────────────────────────────────────────────────────────────────────

mod commands;

use commands::{
    check_sidecar_ready,
    cleanup_temp_dir,
    get_app_version,
    run_local_deepfake_test,
    run_local_protection,
    write_temp_input_image,
};
use std::sync::Mutex;
use tauri::{Manager, RunEvent};

/// Shared application state – tracks the temp-dir path and any running child PIDs.
pub struct AppState {
    pub temp_dir: Mutex<String>,
    pub child_pids: Mutex<Vec<u32>>,
}

pub fn run() {
    env_logger::init();

    tauri::Builder::default()
        // ── Plugins ───────────────────────────────────────────────────────────
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_http::init())
        .plugin(tauri_plugin_process::init())
        // ── Managed State ────────────────────────────────────────────────────
        .manage(AppState {
            temp_dir: Mutex::new(String::new()),
            child_pids: Mutex::new(Vec::new()),
        })
        // ── Setup Hook ───────────────────────────────────────────────────────
        .setup(|app| {
            // Create a unique temp directory for this session.
            let tmp = tempfile::Builder::new()
                .prefix("deepfake-defense-")
                .tempdir()
                .expect("Failed to create temp directory");

            let tmp_path = tmp.path().to_string_lossy().to_string();
            log::info!("Session temp dir: {}", tmp_path);

            // Persist so the dir is not dropped before the app closes
            // (tempfile::TempDir is dropped at end of scope otherwise).
            app.manage(tmp); // keep TempDir alive via managed state

            let state: tauri::State<AppState> = app.state();
            *state.temp_dir.lock().unwrap() = tmp_path;

            Ok(())
        })
        // ── Commands ─────────────────────────────────────────────────────────
        .invoke_handler(tauri::generate_handler![
            run_local_protection,
            run_local_deepfake_test,
            write_temp_input_image,
            check_sidecar_ready,
            cleanup_temp_dir,
            get_app_version,
        ])
        // ── Run / Lifecycle ──────────────────────────────────────────────────
        .build(tauri::generate_context!())
        .expect("Error while building Tauri application")
        .run(|app_handle, event| {
            if let RunEvent::ExitRequested { .. } = event {
                // Kill any lingering sidecar processes.
                let state: tauri::State<AppState> = app_handle.state();
                let pids = state.child_pids.lock().unwrap().clone();
                for pid in pids {
                    log::info!("Killing child process {}", pid);
                    // Best-effort kill; ignore errors.
                    let _ = std::process::Command::new("kill")
                        .arg(pid.to_string())
                        .output();
                }

                // Temp cleanup handled by TempDir drop when app_handle is dropped.
                log::info!("Application exiting – cleanup complete.");
            }
        });
}
