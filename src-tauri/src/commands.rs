// src-tauri/src/commands.rs
// Tauri v2 compatible commands for DeepFake Defense

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tauri::Manager;

#[derive(Debug, Serialize, Deserialize)]
pub struct ProtectionResult {
    pub path: String,
    pub score: f64,
}

/// Call the local Python sidecar binary to protect an image
#[tauri::command]
pub async fn protect_image_local(
    app: tauri::AppHandle,
    input_path: String,
    epsilon: f64,
) -> Result<ProtectionResult, String> {
    // Get app data directory
    let app_data_dir = app
        .path()
        .app_local_data_dir()
        .map_err(|e| format!("Failed to get app data directory: {}", e))?;
    
    let output_dir = app_data_dir.join("protected");
    
    std::fs::create_dir_all(&output_dir)
        .map_err(|e| format!("Failed to create output directory: {}", e))?;
    
    let timestamp = chrono::Utc::now().timestamp();
    let output_path = output_dir.join(format!("protected-{}.png", timestamp));
    let output_path_str = output_path
        .to_str()
        .ok_or("Invalid output path")?
        .to_string();

    // Create sidecar command using tauri-plugin-shell
    let sidecar_command = tauri::async_runtime::spawn(async move {
        use tauri_plugin_shell::ShellExt;
        
        let sidecar = app
            .shell()
            .sidecar("defense-engine")
            .map_err(|e| format!("Failed to create sidecar command: {}", e))?;
        
        let output = sidecar
            .args([
                "--mode", "protect",
                "--input", &input_path,
                "--output", &output_path_str,
                "--level", &epsilon.to_string(),
                "--size", "512",
            ])
            .output()
            .await
            .map_err(|e| format!("Failed to execute sidecar: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("Sidecar failed: {}", stderr));
        }

        // Parse output for SUCCESS line
        let stdout = String::from_utf8_lossy(&output.stdout);
        for line in stdout.lines() {
            if line.starts_with("SUCCESS:") {
                if let Some(json_str) = line.strip_prefix("SUCCESS: ") {
                    let result: serde_json::Value = serde_json::from_str(json_str)
                        .map_err(|e| format!("Failed to parse result: {}", e))?;
                    
                    return Ok(ProtectionResult {
                        path: result["path"]
                            .as_str()
                            .unwrap_or(&output_path_str)
                            .to_string(),
                        score: result["score"].as_f64().unwrap_or(0.0),
                    });
                }
            }
        }

        Err("No success message found in output".to_string())
    });

    sidecar_command
        .await
        .map_err(|e| format!("Sidecar task failed: {}", e))?
}

/// Health check for Modal cloud backend
#[tauri::command]
pub async fn check_cloud_health() -> Result<bool, String> {
    let client = reqwest::Client::new();
    
    match client
        .get("https://akshay-3046--deepfake-defense-web.modal.run/health")
        .timeout(std::time::Duration::from_secs(5))
        .send()
        .await
    {
        Ok(response) => Ok(response.status().is_success()),
        Err(e) => {
            log::warn!("Modal health check failed: {}", e);
            Ok(false) // Return false instead of error for graceful degradation
        }
    }
}

/// Get the app's local data directory path
#[tauri::command]
pub async fn get_app_data_dir(app: tauri::AppHandle) -> Result<String, String> {
    let data_dir = app
        .path()
        .app_local_data_dir()
        .map_err(|e| format!("Failed to get app data directory: {}", e))?;
    
    data_dir
        .to_str()
        .ok_or("Invalid path")?
        .to_string()
        .pipe(Ok)
}

// Helper trait for pipe operations
trait Pipe: Sized {
    fn pipe<F, R>(self, f: F) -> R
    where
        F: FnOnce(Self) -> R,
    {
        f(self)
    }
}

impl<T> Pipe for T {}
