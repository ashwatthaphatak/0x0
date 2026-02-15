// src/page.tsx
"use client";

import { useState, useCallback, useEffect } from "react";
import { ImageDropzone }             from "@/components/image-dropzone";
import { ImageCropper }              from "@/components/image-cropper";
import { ComputeToggle }             from "@/components/compute-toggle";
import { ProtectionLevelSelector }   from "@/components/protection-level-selector";
import { ProgressTracker }           from "@/components/progress-tracker";
import { ResultViewer }              from "@/components/result-viewer";
import { AttackTester }              from "@/components/attack-tester";
import { useProtection }             from "@/hooks/useProtection";
import { IS_TAURI, cleanupTempDir, getAppVersion, localPathToUrl } from "@/lib/tauri-bridge";
import type { ComputeMode, ProtectionLevel } from "@/types";

// ─── Types & Constants ────────────────────────────────────────────────────────

type AppStage =
  | "upload"       // waiting for image
  | "crop"         // cropping the loaded image
  | "ready"        // image ready, waiting to start
  | "processing"   // sidecar / cloud running
  | "complete"     // done
  | "error";

const STORAGE_KEY_MODE  = "deepfake-defense:compute-mode";
const STORAGE_KEY_LEVEL = "deepfake-defense:protection-level";

function readPersisted<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function HomePage() {
  // Persisted settings
  const [mode,  setModeState]  = useState<ComputeMode>(()    => readPersisted(STORAGE_KEY_MODE,  "local"));
  const [level, setLevelState] = useState<ProtectionLevel>(() => readPersisted(STORAGE_KEY_LEVEL, "medium"));

  const setMode = useCallback((m: ComputeMode) => {
    setModeState(m);
    localStorage.setItem(STORAGE_KEY_MODE, JSON.stringify(m));
  }, []);

  const setLevel = useCallback((l: ProtectionLevel) => {
    setLevelState(l);
    localStorage.setItem(STORAGE_KEY_LEVEL, JSON.stringify(l));
  }, []);

  // App stage
  const [stage,         setStage]         = useState<AppStage>("upload");

  // Image data
  const [rawFile,       setRawFile]       = useState<File | null>(null);
  const [rawPath,       setRawPath]       = useState<string | undefined>();    // native path (Tauri local mode)
  const [rawPreviewUrl, setRawPreviewUrl] = useState<string>("");              // data URL for preview

  const [croppedBlob,   setCroppedBlob]   = useState<Blob | null>(null);
  const [croppedUrl,    setCroppedUrl]    = useState<string>("");              // data URL of cropped image

  const [version,       setVersion]       = useState("0.1.0");

  // Protection hook
  const {
    state: processingState,
    progress,
    statusMsg,
    result,
    error,
    protect,
    reset: resetProtection,
  } = useProtection({ mode, level });

  // Load version
  useEffect(() => {
    getAppVersion().then(setVersion).catch(() => {});
  }, []);

  // ── Image Loaded ────────────────────────────────────────────────────────────
  const handleImageLoaded = useCallback(
    (file: File | null, nativePath?: string) => {
      setRawFile(file);
      setRawPath(nativePath);

      if (file) {
        const url = URL.createObjectURL(file);
        setRawPreviewUrl(url);
        setStage("crop");
      } else if (nativePath) {
        // Tauri native path with no blob fallback.
        // Resolve through Tauri's converter so the webview can render it correctly.
        void localPathToUrl(nativePath)
          .then((url) => setRawPreviewUrl(url))
          .catch(() => setRawPreviewUrl(`asset://localhost/${encodeURIComponent(nativePath)}`));
        setStage("crop");
      }
    },
    []
  );

  // ── Crop Confirmed ──────────────────────────────────────────────────────────
  const handleCropConfirm = useCallback(
    (blob: Blob, dataUrl: string) => {
      setCroppedBlob(blob);
      setCroppedUrl(dataUrl);
      setStage("ready");
    },
    []
  );

  // ── Sanitize ────────────────────────────────────────────────────────────────
  const handleSanitize = useCallback(async () => {
    setStage("processing");

    if (mode === "local") {
      // Prefer native path for local mode (avoids writing the blob back to disk)
      const path = rawPath ?? null;
      if (!path) {
        // No native path: write cropped blob to a temp file via Tauri FS
        // For simplicity we pass the blob's object URL – the Python engine
        // cannot read this, so we fall back to a temp write approach.
        // In production, write the Blob to a temp file using tauri-plugin-fs
        // and pass that path. For now, show an informative error.
        alert("Local mode requires a file path. Please re-load the image from disk.");
        setStage("ready");
        return;
      }

      try {
        await protect(path);
        setStage("complete");
      } catch {
        setStage("error");
      }
    } else {
      // Cloud mode: needs a Blob
      const blob = croppedBlob;
      if (!blob) { setStage("ready"); return; }

      try {
        await protect(blob);
        setStage("complete");
      } catch {
        setStage("error");
      }
    }
  }, [mode, rawPath, croppedBlob, protect]);

  // ── Reset ────────────────────────────────────────────────────────────────────
  const handleReset = useCallback(() => {
    // Revoke any blob URLs to free memory
    if (rawPreviewUrl.startsWith("blob:")) URL.revokeObjectURL(rawPreviewUrl);
    if (croppedUrl.startsWith("blob:"))   URL.revokeObjectURL(croppedUrl);

    setRawFile(null);
    setRawPath(undefined);
    setRawPreviewUrl("");
    setCroppedBlob(null);
    setCroppedUrl("");
    setStage("upload");
    resetProtection();
  }, [rawPreviewUrl, croppedUrl, resetProtection]);

  // ── Cleanup on unmount ───────────────────────────────────────────────────────
  useEffect(() => {
    const cleanup = () => {
      if (IS_TAURI) cleanupTempDir();
    };
    window.addEventListener("beforeunload", cleanup);
    return () => window.removeEventListener("beforeunload", cleanup);
  }, []);

  // ── Derive actual stage from hook state ──────────────────────────────────────
  useEffect(() => {
    if (processingState === "complete") setStage("complete");
    if (processingState === "error")    setStage("error");
  }, [processingState]);

  const isProcessing = stage === "processing";
  const comparisonOriginal = mode === "local" ? (rawPreviewUrl || croppedUrl) : croppedUrl;

  // ─────────────────────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      {/* ── Header ── */}
      <header className="flex items-center justify-between px-6 py-4 border-b border-slate-800/60">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center">
            <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"/>
            </svg>
          </div>
          <div>
            <h1 className="font-bold text-base leading-none">DeepFake Defense</h1>
            <p className="text-[10px] text-slate-500 mt-0.5">Texture Feature Perturbation</p>
          </div>
        </div>
        <span className="text-xs text-slate-600 font-mono">v{version}</span>
      </header>

      {/* ── Main Content ── */}
      <main className="flex-1 flex flex-col lg:flex-row gap-0">

        {/* Left panel – settings */}
        <aside className="w-full lg:w-80 xl:w-96 border-b lg:border-b-0 lg:border-r border-slate-800/60 p-6 flex flex-col gap-6">
          <div>
            <h2 className="text-sm font-semibold text-slate-300 mb-1">How it works</h2>
            <p className="text-xs text-slate-500 leading-relaxed">
              Upload a photo and click <strong className="text-slate-400">Sanitize</strong>.
              We inject an imperceptible adversarial perturbation into texture regions,
              breaking deepfake generators <em>before</em> they can manipulate your image.
            </p>
          </div>

          <ComputeToggle
            mode={mode}
            onChange={setMode}
            disabled={isProcessing}
          />

          <ProtectionLevelSelector
            level={level}
            onChange={setLevel}
            disabled={isProcessing}
          />

          {/* Processing info */}
          <div className="flex flex-col gap-2 text-xs text-slate-500">
            <div className="flex items-start gap-2">
              <span className="text-indigo-400 mt-0.5">★</span>
              <span>Your image is processed at 1024 × 1024 px internally.</span>
            </div>
            <div className="flex items-start gap-2">
              <span className="text-indigo-400 mt-0.5">★</span>
              <span>Local mode never sends data off-device.</span>
            </div>
            <div className="flex items-start gap-2">
              <span className="text-indigo-400 mt-0.5">★</span>
              <span>Perturbation is invisible: PSNR ≥ 38 dB.</span>
            </div>
          </div>
        </aside>

        {/* Right panel – main workflow */}
        <section className="flex-1 p-6 flex flex-col gap-6">

          {/* Upload stage */}
          {stage === "upload" && (
            <div className="flex flex-col gap-4">
              <div>
                <h2 className="text-lg font-semibold">Upload Image</h2>
                <p className="text-sm text-slate-400 mt-0.5">
                  Drag & drop or browse to load your photo
                </p>
              </div>
              <ImageDropzone
                onImageLoaded={handleImageLoaded}
                disabled={false}
              />
            </div>
          )}

          {/* Crop stage */}
          {stage === "crop" && (
            <div className="flex flex-col gap-4">
              <div>
                <h2 className="text-lg font-semibold">Crop Image</h2>
                <p className="text-sm text-slate-400 mt-0.5">
                  Adjust the crop area then confirm
                </p>
              </div>
              <ImageCropper
                imageSrc={rawPreviewUrl}
                onConfirm={handleCropConfirm}
                onCancel={handleReset}
              />
            </div>
          )}

          {/* Ready stage */}
          {stage === "ready" && (
            <div className="flex flex-col gap-6">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-lg font-semibold">Ready to Protect</h2>
                  <p className="text-sm text-slate-400 mt-0.5">
                    Image loaded — click Sanitize to inject protection
                  </p>
                </div>
                <button
                  onClick={handleReset}
                  className="text-xs text-slate-500 hover:text-slate-300 transition-colors"
                >
                  Change image
                </button>
              </div>

              {croppedUrl && (
                <div className="relative w-full h-64 rounded-2xl overflow-hidden bg-slate-900">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={croppedUrl}
                    alt="Cropped preview"
                    className="w-full h-full object-contain"
                  />
                  <div className="absolute top-3 left-3 px-2 py-1 rounded-md bg-black/60 text-xs text-white font-medium">
                    Preview · {mode === "local" ? "Local" : "Cloud"} ·{" "}
                    {["low","medium","high"].find((l) => l === level)}
                  </div>
                </div>
              )}

              <button
                onClick={handleSanitize}
                className="w-full py-4 rounded-2xl bg-gradient-to-r from-indigo-600 to-violet-600
                           hover:from-indigo-500 hover:to-violet-500 text-white font-bold text-lg
                           shadow-lg shadow-indigo-900/40 transition-all duration-200 active:scale-[0.98]"
              >
                🛡️ Sanitize Image
              </button>
            </div>
          )}

          {/* Processing stage */}
          {stage === "processing" && (
            <div className="flex flex-col gap-6">
              <div>
                <h2 className="text-lg font-semibold">Processing…</h2>
                <p className="text-sm text-slate-400 mt-0.5">
                  {mode === "local" ? "Running local defense engine" : "Processing in the cloud"}
                </p>
              </div>

              {croppedUrl && (
                <div className="relative w-full h-48 rounded-2xl overflow-hidden bg-slate-900">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={croppedUrl} alt="Processing" className="w-full h-full object-contain opacity-60" />
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="w-16 h-16 rounded-full border-4 border-indigo-500 border-t-transparent animate-spin"/>
                  </div>
                </div>
              )}

              <ProgressTracker
                state={processingState}
                percent={progress}
                statusMsg={statusMsg}
                error={error}
                mode={mode}
              />
            </div>
          )}

          {/* Complete stage */}
          {stage === "complete" && result && (
            <div className="flex flex-col gap-6">
              <div>
                <h2 className="text-lg font-semibold">Protection Applied ✓</h2>
                <p className="text-sm text-slate-400 mt-0.5">
                  Compare the original and sanitized images side by side, then run a deepfake test.
                </p>
              </div>
              <ResultViewer
                original={comparisonOriginal}
                result={result}
                onReset={handleReset}
              />
              <AttackTester
                originalPath={rawPath}
                sanitizedPath={result.outputPath}
                isLocalResult={result.isLocal}
              />
            </div>
          )}

          {/* Safety fallback: avoid blank panel if complete has no result */}
          {stage === "complete" && !result && (
            <div className="flex flex-col gap-4">
              <div className="p-5 rounded-2xl bg-amber-950/40 border border-amber-700">
                <h3 className="text-amber-300 font-semibold mb-2">No Result Available</h3>
                <p className="text-sm text-amber-200">
                  Processing finished without an output image. Please try again.
                </p>
                {error && (
                  <p className="text-xs text-amber-100/80 font-mono break-words mt-2">
                    {error}
                  </p>
                )}
              </div>
              <div className="flex gap-3">
                <button
                  onClick={() => { resetProtection(); setStage("ready"); }}
                  className="flex-1 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium transition-colors"
                >
                  Try Again
                </button>
                <button
                  onClick={handleReset}
                  className="flex-1 py-2.5 rounded-xl border border-slate-600 text-slate-300 hover:bg-slate-700 text-sm font-medium transition-colors"
                >
                  Start Over
                </button>
              </div>
            </div>
          )}

          {/* Error stage */}
          {stage === "error" && (
            <div className="flex flex-col gap-4">
              <div className="p-5 rounded-2xl bg-red-950/50 border border-red-800">
                <h3 className="text-red-300 font-semibold mb-2">Protection Failed</h3>
                <p className="text-sm text-red-400 font-mono break-words">
                  {error ?? "An unexpected error occurred."}
                </p>
                {mode === "local" && (
                  <p className="text-xs text-slate-500 mt-3">
                    Tip: Try switching to Cloud mode if the local engine is not responding.
                  </p>
                )}
              </div>
              <div className="flex gap-3">
                <button
                  onClick={() => { resetProtection(); setStage("ready"); }}
                  className="flex-1 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium transition-colors"
                >
                  Try Again
                </button>
                <button
                  onClick={handleReset}
                  className="flex-1 py-2.5 rounded-xl border border-slate-600 text-slate-300 hover:bg-slate-700 text-sm font-medium transition-colors"
                >
                  Start Over
                </button>
              </div>
            </div>
          )}
        </section>
      </main>

      {/* Footer */}
      <footer className="px-6 py-3 border-t border-slate-800/60 text-center text-xs text-slate-600">
        DeepFake Defense · Texture Feature Perturbation · Based on Zhang et al., 2025
      </footer>
    </div>
  );
}
