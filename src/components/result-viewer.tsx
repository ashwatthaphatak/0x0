"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { ProtectionResult } from "@/types";
import { localPathToUrl, saveFileDialog, saveResultToPath } from "@/lib/tauri-bridge";

interface ResultViewerProps {
  original: string;
  result: ProtectionResult;
  onReset: () => void;
}

export function ResultViewer({ original, result, onReset }: ResultViewerProps) {
  const [protectedUrl, setProtectedUrl] = useState<string>("");
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState<string | null>(null);
  const [previewErr, setPreviewErr] = useState<string | null>(null);
  const objectUrlRef = useRef<string | null>(null);

  const releaseObjectUrl = useCallback(() => {
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = null;
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    setProtectedUrl("");
    setPreviewErr(null);
    releaseObjectUrl();

    const resolveProtectedUrl = async () => {
      if (!result.isLocal) {
        if (!cancelled) setProtectedUrl(result.outputPath);
        return;
      }

      try {
        const { readFile } = await import("@tauri-apps/plugin-fs");
        const bytes = await readFile(result.outputPath);
        const blobUrl = URL.createObjectURL(new Blob([bytes], { type: "image/png" }));

        if (cancelled) {
          URL.revokeObjectURL(blobUrl);
          return;
        }

        objectUrlRef.current = blobUrl;
        setProtectedUrl(blobUrl);
        return;
      } catch {
        // Fallback for environments where direct FS read is unavailable.
      }

      try {
        const converted = await localPathToUrl(result.outputPath);
        if (!cancelled) setProtectedUrl(converted);
      } catch (err: unknown) {
        if (!cancelled) {
          setPreviewErr(err instanceof Error ? err.message : "Could not preview sanitized image.");
        }
      }
    };

    void resolveProtectedUrl();

    return () => {
      cancelled = true;
      releaseObjectUrl();
    };
  }, [releaseObjectUrl, result]);

  const handleSave = useCallback(async () => {
    setSaving(true);
    setSaveMsg(null);
    try {
      const dest = await saveFileDialog("sanitized.png");
      if (!dest) {
        setSaving(false);
        return;
      }
      await saveResultToPath(result.outputPath, dest, result.isLocal);
      setSaveMsg(`Saved to ${dest.split(/[\\/]/).pop()}`);
    } catch (err: unknown) {
      setSaveMsg(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }, [result]);

  const scoreColor =
    result.score >= 80 ? "text-emerald-400" : result.score >= 50 ? "text-yellow-400" : "text-red-400";

  return (
    <div className="flex flex-col gap-5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="flex flex-col">
            <span className="text-xs text-slate-500 uppercase tracking-widest">Protection Score</span>
            <span className={`text-3xl font-bold tabular-nums ${scoreColor}`}>
              {result.score.toFixed(1)}%
            </span>
          </div>
          <div
            className={[
              "px-3 py-1 rounded-full text-sm font-semibold border",
              result.score >= 80
                ? "bg-emerald-900/60 text-emerald-300 border-emerald-700"
                : result.score >= 50
                  ? "bg-yellow-900/60 text-yellow-300 border-yellow-700"
                  : "bg-red-900/60 text-red-300 border-red-700",
            ].join(" ")}
          >
            {result.score >= 80 ? "Protected" : result.score >= 50 ? "Partial" : "Weak"}
          </div>
        </div>
        <span className="text-xs text-slate-500">{result.isLocal ? "Local" : "Cloud"}</span>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div className="rounded-2xl border border-slate-700/80 bg-slate-900/70 p-3">
          <div className="mb-2 text-xs font-medium uppercase tracking-wider text-slate-400">Original</div>
          <div className="h-64 overflow-hidden rounded-xl bg-black">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={original} alt="Original" className="h-full w-full object-contain" />
          </div>
        </div>

        <div className="rounded-2xl border border-indigo-700/70 bg-indigo-950/20 p-3">
          <div className="mb-2 text-xs font-medium uppercase tracking-wider text-indigo-300">Sanitized</div>
          <div className="h-64 overflow-hidden rounded-xl bg-black">
            {protectedUrl ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={protectedUrl}
                alt="Sanitized"
                className="h-full w-full object-contain"
                onError={() => {
                  setProtectedUrl("");
                  setPreviewErr("Could not preview sanitized image.");
                }}
              />
            ) : (
              <div className="flex h-full items-center justify-center text-xs text-slate-500">Loading preview...</div>
            )}
          </div>
        </div>
      </div>

      {previewErr && <p className="text-center text-xs text-amber-400">{previewErr}</p>}

      <div className="flex gap-3">
        <button
          onClick={onReset}
          className="flex-1 rounded-xl border border-slate-600 px-4 py-2.5 text-sm font-medium text-slate-300 transition-colors hover:bg-slate-700"
        >
          New Image
        </button>
        <button
          onClick={handleSave}
          disabled={saving || !result.outputPath}
          className="flex flex-1 items-center justify-center gap-2 rounded-xl bg-indigo-600 px-4 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-indigo-500 disabled:opacity-50"
        >
          {saving ? "Saving..." : "Save Sanitized Image"}
        </button>
      </div>

      {saveMsg && <p className="text-center text-xs text-slate-400">{saveMsg}</p>}
    </div>
  );
}
