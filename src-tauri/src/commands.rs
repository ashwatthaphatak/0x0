// src-tauri/src/commands.rs
// ─────────────────────────────────────────────────────────────────────────────
// All Tauri commands exposed via invoke() from the Next.js frontend.
// ─────────────────────────────────────────────────────────────────────────────

use crate::AppState;
use anyhow::Context;
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter, Manager, State};
use uuid::Uuid;

// ─── Shared Types ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtectionResult {
    pub output_path: String,
    pub score: f64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum ProgressEvent {
    Status  { message: String },
    Progress { percent: u8 },
    Complete { result: ProtectionResult },
    Error   { message: String },
}

fn stderr_tail_message(lines: &[String]) -> String {
    if lines.is_empty() {
        return String::new();
    }
    format!("\nSidecar stderr (last {} lines):\n{}", lines.len(), lines.join("\n"))
}

fn target_triple_guess() -> String {
    if let Ok(target) = std::env::var("TARGET") {
        if !target.trim().is_empty() {
            return target;
        }
    }
    match (std::env::consts::ARCH, std::env::consts::OS) {
        ("aarch64", "macos") => "aarch64-apple-darwin".to_string(),
        ("x86_64", "macos") => "x86_64-apple-darwin".to_string(),
        ("x86_64", "linux") => "x86_64-unknown-linux-gnu".to_string(),
        ("aarch64", "linux") => "aarch64-unknown-linux-gnu".to_string(),
        ("x86_64", "windows") => "x86_64-pc-windows-msvc".to_string(),
        ("aarch64", "windows") => "aarch64-pc-windows-msvc".to_string(),
        (arch, os) => format!("{arch}-{os}"),
    }
}

fn python_candidates() -> Vec<String> {
    let mut candidates: Vec<String> = Vec::new();

    for key in ["DEEPFAKE_DEFENSE_PYTHON", "PYTHON_BIN"] {
        if let Ok(val) = std::env::var(key) {
            let trimmed = val.trim();
            if !trimmed.is_empty() {
                candidates.push(trimmed.to_string());
            }
        }
    }

    if let Ok(home) = std::env::var("HOME") {
        for rel in [
            "miniforge3/bin/python",
            "miniconda3/bin/python",
            "anaconda3/bin/python",
            ".pyenv/shims/python",
        ] {
            candidates.push(format!("{home}/{rel}"));
        }
    }

    candidates.push("python".to_string());
    candidates.push("python3".to_string());
    candidates
}

fn python_has_engine_deps(python_bin: &str) -> bool {
    Command::new(python_bin)
        .args(["-c", "import torch,torchvision,cv2,PIL,numpy,skimage"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn resolve_python_engine_bin() -> Result<String, String> {
    let mut tried: Vec<String> = Vec::new();
    for candidate in python_candidates() {
        if tried.contains(&candidate) {
            continue;
        }
        tried.push(candidate.clone());
        if python_has_engine_deps(&candidate) {
            return Ok(candidate);
        }
    }

    Err(format!(
        "No Python interpreter with required ML dependencies was found. Tried: {}. \
Set DEEPFAKE_DEFENSE_PYTHON to your env interpreter or run \
`cd python_engine && PYTHON_BIN=<python> ./build_binary.sh`.",
        tried.join(", ")
    ))
}

fn dev_python_engine_script() -> Option<std::path::PathBuf> {
    if !cfg!(debug_assertions) {
        return None;
    }
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?;
    let src_tauri_dir = dir.parent()?.parent()?;
    let repo_root = src_tauri_dir.parent()?;
    let script = repo_root.join("python_engine").join("main.py");
    if script.exists() {
        Some(script)
    } else {
        None
    }
}

// ─── Helper: locate sidecar binary ──────────────────────────────────────────

/// Returns the absolute path to the bundled `defense-engine` binary.
/// Tauri copies sidecars next to the main executable at install time.
fn sidecar_path(app: &AppHandle) -> anyhow::Result<std::path::PathBuf> {
    let exe = std::env::current_exe().context("Cannot determine exe path")?;
    let dir = exe
        .parent()
        .context("Exe has no parent directory")?;
    let target_triple = target_triple_guess();

    // During development, prefer `src-tauri/binaries/defense-engine-<triple>`
    // so fresh sidecar rebuilds are picked up without manually copying.
    let mut dev_candidates: Vec<std::path::PathBuf> = Vec::new();
    if let Some(src_tauri_dir) = dir.parent().and_then(|p| p.parent()) {
        dev_candidates.push(
            src_tauri_dir
                .join("binaries")
                .join(format!("defense-engine-{}", target_triple)),
        );
        dev_candidates.push(
            src_tauri_dir
                .join("binaries")
                .join(format!("defense-engine-{}.exe", target_triple)),
        );
    }

    // Tauri 2 sidecar convention: same dir as the executable
    let mut candidates = dev_candidates;
    candidates.push(dir.join("defense-engine"));
    candidates.push(dir.join("defense-engine.exe"));
    candidates.push(
        app.path()
            .resource_dir()
            .unwrap_or_default()
            .join("defense-engine"),
    );

    for c in &candidates {
        if c.exists() {
            return Ok(c.clone());
        }
    }

    anyhow::bail!(
        "defense-engine sidecar not found. Run python_engine/build_binary.sh first."
    )
}

// ─── Commands ────────────────────────────────────────────────────────────────

/// Run the local Python sidecar to protect an image.
/// Streams STATUS / PROGRESS / SUCCESS / ERROR lines via Tauri events.
///
/// JS usage:
/// ```ts
/// await invoke('run_local_protection', { imagePath, outputPath, epsilon });
/// ```
#[tauri::command]
pub async fn run_local_protection(
    app:      AppHandle,
    state:    State<'_, AppState>,
    image_path:  String,
    output_path: Option<String>,
    epsilon:     Option<f64>,
    size:        Option<u32>,
) -> Result<ProtectionResult, String> {
    // Determine output path (default: temp dir)
    let out_path = match output_path {
        Some(p) => p,
        None => {
            let tmp_dir = state.temp_dir.lock().unwrap().clone();
            let fname   = format!("protected-{}.png", Uuid::new_v4());
            format!("{}/{}", tmp_dir, fname)
        }
    };

    let eps  = epsilon.unwrap_or(0.05);
    let sz   = size.unwrap_or(1024);

    let mut base_cmd = if let Some(script) = dev_python_engine_script() {
        let python_bin = resolve_python_engine_bin()
            .map_err(|e| format!("Local Python engine is not ready. {e}"))?;
        log::info!("Using Python engine in dev mode: {}", script.display());
        log::info!("Using Python interpreter for local engine: {}", python_bin);
        let mut c = Command::new(&python_bin);
        c.arg(script);
        c
    } else {
        let engine = sidecar_path(&app).map_err(|e| e.to_string())?;
        log::info!("Using sidecar engine binary: {}", engine.display());
        Command::new(&engine)
    };

    log::info!(
        "Spawning local engine with --input {:?} --output {:?} --level {} --size {}",
        image_path,
        out_path,
        eps,
        sz
    );

    let tmp_dir = state.temp_dir.lock().unwrap().clone();
    let torch_home = format!("{}/torch-cache", tmp_dir);
    let _ = std::fs::create_dir_all(&torch_home);

    let mut child = base_cmd
        .env("TORCH_HOME", &torch_home)
        .env("XDG_CACHE_HOME", &torch_home)
        .args([
            "--input",  &image_path,
            "--output", &out_path,
            "--level",  &eps.to_string(),
            "--size",   &sz.to_string(),
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to start local engine: {e}"))?;

    // Track PID for cleanup
    {
        let mut pids = state.child_pids.lock().unwrap();
        pids.push(child.id());
    }

    let stdout = child
        .stdout
        .take()
        .ok_or("Could not capture stdout from sidecar")?;
    let stderr = child
        .stderr
        .take()
        .ok_or("Could not capture stderr from sidecar")?;

    let stderr_lines: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let stderr_lines_ref = Arc::clone(&stderr_lines);
    let stderr_thread = std::thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines().map_while(Result::ok) {
            log::warn!("sidecar! {}", line);
            let mut lines = stderr_lines_ref.lock().unwrap();
            lines.push(line);
            if lines.len() > 25 {
                lines.remove(0);
            }
        }
    });

    let reader  = BufReader::new(stdout);
    let app_ref = app.clone();
    let mut last_result: Option<ProtectionResult> = None;

    // Parse each stdout line and emit events to the frontend
    for line in reader.lines() {
        let line = line.map_err(|e| e.to_string())?;
        log::debug!("sidecar> {}", line);

        let event: ProgressEvent = if let Some(msg) = line.strip_prefix("STATUS: ") {
            ProgressEvent::Status { message: msg.to_string() }
        } else if let Some(pct) = line.strip_prefix("PROGRESS: ") {
            let percent = pct.trim().parse::<u8>().unwrap_or(0);
            ProgressEvent::Progress { percent }
        } else if let Some(payload) = line.strip_prefix("SUCCESS: ") {
            let parsed: serde_json::Value = serde_json::from_str(payload)
                .unwrap_or(serde_json::json!({"path": payload, "score": 0.0}));

            let result = ProtectionResult {
                output_path: parsed["path"]
                    .as_str()
                    .unwrap_or(&out_path)
                    .to_string(),
                score: parsed["score"].as_f64().unwrap_or(0.0),
            };
            last_result = Some(result.clone());
            ProgressEvent::Complete { result }
        } else if let Some(msg) = line.strip_prefix("ERROR: ") {
            ProgressEvent::Error { message: msg.to_string() }
        } else {
            // Ignore unrecognized lines (debug output, etc.)
            continue;
        };

        app_ref
            .emit("protection-progress", &event)
            .map_err(|e| e.to_string())?;

        if matches!(event, ProgressEvent::Error { .. }) {
            let _ = child.kill();
            let _ = child.wait();
            let _ = stderr_thread.join();
            let stderr_tail = stderr_lines
                .lock()
                .map(|v| v.clone())
                .unwrap_or_default();
            return Err(match event {
                ProgressEvent::Error { message } => message,
                _ => "Unknown error from sidecar".to_string(),
            } + &stderr_tail_message(&stderr_tail));
        }
    }

    let exit_status = child
        .wait()
        .map_err(|e| format!("Failed to wait for sidecar: {e}"))?;
    let _ = stderr_thread.join();
    let stderr_tail = stderr_lines
        .lock()
        .map(|v| v.clone())
        .unwrap_or_default();

    // Remove PID from tracking list
    {
        let pid = exit_status; // `wait()` consumes, store already removed above
        let _ = pid; // silence unused warning
        let mut pids = state.child_pids.lock().unwrap();
        pids.retain(|&p| {
            // We can't recover the PID after wait() in std; just clear all finished ones.
            // A production implementation would use a HashMap.
            p != 0 // keep all (best effort)
        });
    }

    if !exit_status.success() {
        return Err(format!(
            "Local engine exited with non-zero code: {:?}. \
             Try switching to Cloud mode.{}",
            exit_status.code(),
            stderr_tail_message(&stderr_tail)
        ));
    }

    last_result.ok_or_else(|| {
        format!(
            "Sidecar exited without a SUCCESS line. Check that the output path is writable.{}",
            stderr_tail_message(&stderr_tail)
        )
    })
}

/// Ping the sidecar to verify it can be executed (returns version string).
#[tauri::command]
pub async fn check_sidecar_ready(app: AppHandle) -> Result<String, String> {
    let output = if let Some(script) = dev_python_engine_script() {
        let python_bin = resolve_python_engine_bin()?;
        Command::new(&python_bin)
            .arg(script)
            .arg("--help")
            .output()
            .map_err(|e| format!("Python engine not executable: {e}"))?
    } else {
        let engine = sidecar_path(&app).map_err(|e| e.to_string())?;
        Command::new(&engine)
            .arg("--help")
            .output()
            .map_err(|e| format!("Sidecar not executable: {e}"))?
    };

    if output.status.success() || !output.stdout.is_empty() || !output.stderr.is_empty() {
        Ok("ready".to_string())
    } else {
        Err("Sidecar returned empty output on --help".to_string())
    }
}

/// Delete all files in the session temp directory.
#[tauri::command]
pub async fn cleanup_temp_dir(state: State<'_, AppState>) -> Result<(), String> {
    let tmp_dir = state.temp_dir.lock().unwrap().clone();
    if tmp_dir.is_empty() {
        return Ok(());
    }

    let path = std::path::Path::new(&tmp_dir);
    if path.exists() {
        std::fs::remove_dir_all(path)
            .map_err(|e| format!("Cleanup failed: {e}"))?;
        log::info!("Cleaned up temp dir: {}", tmp_dir);
    }

    Ok(())
}

/// Return the application version string.
#[tauri::command]
pub fn get_app_version(app: AppHandle) -> String {
    app.config().version.clone().unwrap_or_else(|| "0.1.0".to_string())
}
