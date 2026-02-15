// src/lib/tauri-bridge.ts
// ─────────────────────────────────────────────────────────────────────────────
// Thin wrapper around Tauri's JS API so that every other file imports from
// here rather than directly from @tauri-apps/*. Enables easy mocking in tests
// and graceful degradation when running in a browser (dev mode without Tauri).
// ─────────────────────────────────────────────────────────────────────────────

import type { ProtectionResult, ProgressUpdate } from "@/types";

// Detect Tauri environment
export const IS_TAURI = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

// ── Dynamic imports ──────────────────────────────────────────────────────────

async function getInvoke() {
  if (!IS_TAURI) throw new Error("Not running inside Tauri");
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke;
}

async function getDialog() {
  if (!IS_TAURI) throw new Error("Not running inside Tauri");
  return import("@tauri-apps/plugin-dialog");
}

async function getListen() {
  if (!IS_TAURI) throw new Error("Not running inside Tauri");
  const { listen } = await import("@tauri-apps/api/event");
  return listen;
}

async function getConvertFileSrc() {
  if (!IS_TAURI) return (path: string) => path;
  const { convertFileSrc } = await import("@tauri-apps/api/core");
  return convertFileSrc;
}

// ── Commands ─────────────────────────────────────────────────────────────────

export async function checkSidecarReady(): Promise<boolean> {
  try {
    const invoke = await getInvoke();
    const result = await invoke<string>("check_sidecar_ready");
    return result === "ready";
  } catch {
    return false;
  }
}

export async function runLocalProtection(
  imagePath: string,
  epsilon: number,
  outputPath?: string,
  size?: number
): Promise<ProtectionResult> {
  const invoke = await getInvoke();
  const raw = await invoke<{ output_path: string; score: number }>(
    "run_local_protection",
    { imagePath, outputPath, epsilon, size }
  );
  return { outputPath: raw.output_path, score: raw.score, isLocal: true };
}

export async function cleanupTempDir(): Promise<void> {
  try {
    const invoke = await getInvoke();
    await invoke("cleanup_temp_dir");
  } catch {
    // Best effort
  }
}

export async function getAppVersion(): Promise<string> {
  try {
    const invoke = await getInvoke();
    return await invoke<string>("get_app_version");
  } catch {
    return "0.1.0";
  }
}

// ── Dialog ───────────────────────────────────────────────────────────────────

export async function openFileDialog(): Promise<string | null> {
  const dialog = await getDialog();
  const result = await dialog.open({
    multiple: false,
    filters: [{ name: "Images", extensions: ["jpg", "jpeg", "png", "webp"] }],
  });
  if (!result) return null;
  return typeof result === "string" ? result : result[0] ?? null;
}

export async function saveFileDialog(defaultName = "protected.png"): Promise<string | null> {
  const dialog = await getDialog();
  const result = await dialog.save({
    defaultPath: defaultName,
    filters: [{ name: "PNG Image", extensions: ["png"] }],
  });
  return result ?? null;
}

export async function showErrorDialog(title: string, message: string): Promise<void> {
  const dialog = await getDialog();
  await dialog.message(message, { title, kind: "error" });
}

// ── Event Listeners ──────────────────────────────────────────────────────────

type UnlistenFn = () => void;

export interface RawProgressEvent {
  type: string;
  data?: {
    message?: string;
    percent?: number;
    result?: { output_path: string; score: number };
  };
}

export async function listenProtectionProgress(
  handler: (update: ProgressUpdate) => void
): Promise<UnlistenFn> {
  const listen = await getListen();

  return listen<RawProgressEvent>("protection-progress", (event) => {
    const payload = event.payload;
    const type = payload.type;

    switch (type) {
      case "status":
        handler({ type: "status", message: payload.data?.message ?? "" });
        break;
      case "progress":
        handler({ type: "progress", percent: payload.data?.percent ?? 0 });
        break;
      case "complete": {
        const r = payload.data?.result;
        if (r) {
          handler({
            type: "complete",
            result: { outputPath: r.output_path, score: r.score, isLocal: true },
          });
        }
        break;
      }
      case "error":
        handler({ type: "error", message: payload.data?.message ?? "Unknown error" });
        break;
    }
  });
}

// ── File Utilities ───────────────────────────────────────────────────────────

/**
 * Convert a local file path to a URL that Tauri's webview can display.
 * Falls back to the raw path in browser mode.
 */
export async function localPathToUrl(filePath: string): Promise<string> {
  const convert = await getConvertFileSrc();
  return convert(filePath);
}

/**
 * Copy the source file/download the URL to the user-chosen destination.
 */
export async function saveResultToPath(
  source: string,
  destination: string,
  isLocal: boolean
): Promise<void> {
  if (isLocal) {
    // Use Tauri FS plugin to copy the temp file
    const { copyFile } = await import("@tauri-apps/plugin-fs");
    await copyFile(source, destination);
  } else {
    // Download from cloud URL and write to destination
    const response = await fetch(source);
    if (!response.ok) throw new Error(`Download failed: ${response.statusText}`);
    const buffer = await response.arrayBuffer();
    const { writeFile } = await import("@tauri-apps/plugin-fs");
    await writeFile(destination, new Uint8Array(buffer));
  }
}
