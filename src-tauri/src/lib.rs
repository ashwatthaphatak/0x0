// src-tauri/src/lib.rs
// Library exports for DeepFake Defense

pub mod commands;

// Re-export commonly used items
pub use commands::{
    protect_image_local,
    check_cloud_health,
    get_app_data_dir,
    ProtectionResult,
};
