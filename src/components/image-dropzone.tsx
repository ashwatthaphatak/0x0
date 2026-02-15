// src/components/image-dropzone.tsx
"use client";

import { useRef, useState, useCallback } from "react";
import { IS_TAURI, openFileDialog } from "@/lib/tauri-bridge";

interface ImageDropzoneProps {
  onImageLoaded: (file: File | null, nativePath?: string) => void;
  disabled?: boolean;
  previewUrl?: string;
}

const ACCEPTED_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);
const MAX_SIZE_MB     = 50;
const MIME_BY_EXT: Record<string, string> = {
  jpg: "image/jpeg",
  jpeg: "image/jpeg",
  png: "image/png",
  webp: "image/webp",
};

function inferMimeType(path: string): string {
  const ext = path.split(".").pop()?.toLowerCase() ?? "";
  return MIME_BY_EXT[ext] ?? "application/octet-stream";
}

export function ImageDropzone({ onImageLoaded, disabled, previewUrl }: ImageDropzoneProps) {
  const [dragOver, setDragOver]  = useState(false);
  const [errMsg,   setErrMsg]    = useState<string | null>(null);
  const inputRef                 = useRef<HTMLInputElement>(null);

  const validateAndLoad = useCallback(
    (file: File, nativePath?: string) => {
      setErrMsg(null);

      if (!ACCEPTED_TYPES.has(file.type)) {
        setErrMsg("Unsupported format. Please use JPG, PNG, or WebP.");
        return;
      }
      if (file.size > MAX_SIZE_MB * 1024 * 1024) {
        setErrMsg(`File is too large. Maximum size is ${MAX_SIZE_MB} MB.`);
        return;
      }

      onImageLoaded(file, nativePath);
    },
    [onImageLoaded]
  );

  // ── Drop handling ──────────────────────────────────────────────────────────
  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setDragOver(false);
      if (disabled) return;

      const file = e.dataTransfer.files[0];
      if (!file) return;

      // Tauri: the native path is available via e.dataTransfer.items[0].webkitGetAsEntry()
      // For simplicity, we use the File object (Blob) and let the Tauri bridge handle the rest.
      validateAndLoad(file);
    },
    [disabled, validateAndLoad]
  );

  const handleDragOver = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      if (!disabled) setDragOver(true);
    },
    [disabled]
  );

  // ── Browser file input ─────────────────────────────────────────────────────
  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) validateAndLoad(file);
      // Reset so re-selecting the same file triggers onChange
      e.target.value = "";
    },
    [validateAndLoad]
  );

  // ── Native Tauri file picker ───────────────────────────────────────────────
  const handleNativePicker = useCallback(async () => {
    if (IS_TAURI) {
      const path = await openFileDialog();
      if (!path) return;

      try {
        const { readFile } = await import("@tauri-apps/plugin-fs");
        const bytes = await readFile(path);
        const name = path.split(/[\\/]/).pop() ?? "image";
        const file = new File([bytes], name, { type: inferMimeType(path) });
        validateAndLoad(file, path);
        return;
      } catch {
        // Fall back: pass null file with native path so upstream can invoke Rust directly.
        onImageLoaded(null, path);
      }
    } else {
      inputRef.current?.click();
    }
  }, [onImageLoaded, validateAndLoad]);

  return (
    <div className="flex flex-col gap-3">
      {/* Drop zone */}
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={() => setDragOver(false)}
        onClick={disabled ? undefined : handleNativePicker}
        className={[
          "relative flex flex-col items-center justify-center",
          "w-full rounded-2xl border-2 border-dashed transition-all duration-200",
          "cursor-pointer select-none",
          previewUrl ? "h-64" : "h-56",
          dragOver
            ? "border-indigo-400 bg-indigo-950/40 scale-[1.01]"
            : "border-slate-600 bg-slate-800/40 hover:border-indigo-500 hover:bg-indigo-950/20",
          disabled ? "opacity-50 cursor-not-allowed" : "",
        ].join(" ")}
      >
        {previewUrl ? (
          // Image preview
          <>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={previewUrl}
              alt="Loaded image preview"
              className="absolute inset-0 w-full h-full object-contain rounded-2xl p-2"
            />
            <div className="absolute inset-0 flex items-center justify-center rounded-2xl bg-black/40 opacity-0 hover:opacity-100 transition-opacity">
              <span className="text-white text-sm font-medium px-3 py-1 rounded-full bg-black/50 backdrop-blur">
                Click to change image
              </span>
            </div>
          </>
        ) : (
          // Empty state
          <div className="flex flex-col items-center gap-3 text-center px-6">
            <div className="w-14 h-14 rounded-full bg-slate-700 flex items-center justify-center">
              <svg className="w-7 h-7 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                  d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
            </div>
            <div>
              <p className="font-semibold text-slate-300">
                Drop image here or{" "}
                <span className="text-indigo-400 underline underline-offset-2">browse</span>
              </p>
              <p className="text-xs text-slate-500 mt-1">JPG · PNG · WebP · up to {MAX_SIZE_MB} MB</p>
            </div>
          </div>
        )}

        {dragOver && (
          <div className="absolute inset-0 flex items-center justify-center rounded-2xl border-2 border-indigo-400 bg-indigo-950/60 backdrop-blur-sm">
            <p className="text-indigo-300 font-semibold text-lg">Drop to load</p>
          </div>
        )}
      </div>

      {/* Hidden browser file input (fallback) */}
      <input
        ref={inputRef}
        type="file"
        accept=".jpg,.jpeg,.png,.webp"
        className="hidden"
        onChange={handleFileInput}
      />

      {/* Error message */}
      {errMsg && (
        <p className="text-red-400 text-sm flex items-center gap-1.5">
          <svg className="w-4 h-4 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          {errMsg}
        </p>
      )}
    </div>
  );
}
