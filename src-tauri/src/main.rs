// src-tauri/src/main.rs
// Tauri v2 main entry point for DeepFake Defense

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;

use tauri::Manager;

fn main() {
    // Initialize logger
    env_logger::init();

    tauri::Builder::default()
        // Register plugins
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_http::init())
        .plugin(tauri_plugin_process::init())
        // Register commands
        .invoke_handler(tauri::generate_handler![
            commands::protect_image_local,
            commands::check_cloud_health,
            commands::get_app_data_dir,
        ])
        // Setup hook for initialization
        .setup(|app| {
            #[cfg(debug_assertions)]
            {
                let window = app.get_webview_window("main").unwrap();
                window.open_devtools();
            }
            
            log::info!("DeepFake Defense initialized");
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
