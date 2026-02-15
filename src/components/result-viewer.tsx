// src/components/result-viewer.tsx
"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import type { ProtectionResult } from "@/types";
import { localPathToUrl, saveResultToPath, saveFileDialog } from "@/lib/tauri-bridge";

interface ResultViewerProps {
  original:   string;           // data URL or local path URL
  result:     ProtectionResult;
  onReset:    () => void;
}

export function ResultViewer({ original, result, onReset }: ResultViewerProps) {
  const [protectedUrl, setProtectedUrl] = useState<string>("");
  const [sliderPos, setSliderPos]       = useState(50);         // 0–100
  const [saving,    setSaving]          = useState(false);
  const [saveMsg,   setSaveMsg]         = useState<string | null>(null);
  const containerRef                    = useRef<HTMLDivElement>(null);
  const dragging                        = useRef(false);

  // Resolve protected image URL
  useEffect(() => {
    if (result.isLocal) {
      localPathToUrl(result.outputPath).then(setProtectedUrl);
    } else {
      setProtectedUrl(result.outputPath);
    }
  }, [result]);

  // ── Slider drag logic ──────────────────────────────────────────────────────
  const updateSlider = useCallback((clientX: number) => {
    const el = containerRef.current;
    if (!el) return;
    const { left, width } = el.getBoundingClientRect();
    const pct = Math.max(0, Math.min(100, ((clientX - left) / width) * 100));
    setSliderPos(pct);
  }, []);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    dragging.current = true;
    updateSlider(e.clientX);
  }, [updateSlider]);

  useEffect(() => {
    const onMove = (e: MouseEvent)  => { if (dragging.current) updateSlider(e.clientX); };
    const onUp   = ()               => { dragging.current = false; };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup",   onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup",   onUp);
    };
  }, [updateSlider]);

  // Touch support
  const handleTouchMove = useCallback(
    (e: React.TouchEvent) => updateSlider(e.touches[0].clientX),
    [updateSlider]
  );

  // ── Save ──────────────────────────────────────────────────────────────────
  const handleSave = useCallback(async () => {
    setSaving(true);
    setSaveMsg(null);
    try {
      const dest = await saveFileDialog("protected.png");
      if (!dest) { setSaving(false); return; }
      await saveResultToPath(result.outputPath, dest, result.isLocal);
      setSaveMsg(`✓ Saved to ${dest.split(/[\\/]/).pop()}`);
    } catch (err: unknown) {
      setSaveMsg(`✗ ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setSaving(false);
    }
  }, [result]);

  const scoreColor =
    result.score >= 80 ? "text-emerald-400"
    : result.score >= 50 ? "text-yellow-400"
    : "text-red-400";

  return (
    <div className="flex flex-col gap-5">
      {/* Score badge */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="flex flex-col">
            <span className="text-xs text-slate-500 uppercase tracking-widest">Protection Score</span>
            <span className={`text-3xl font-bold tabular-nums ${scoreColor}`}>
              {result.score.toFixed(1)}%
            </span>
          </div>
          <div className={[
            "px-3 py-1 rounded-full text-sm font-semibold",
            result.score >= 80
              ? "bg-emerald-900/60 text-emerald-300 border border-emerald-700"
              : result.score >= 50
              ? "bg-yellow-900/60 text-yellow-300 border border-yellow-700"
              : "bg-red-900/60 text-red-300 border border-red-700",
          ].join(" ")}>
            {result.score >= 80 ? "🛡️ Protected" : result.score >= 50 ? "⚠️ Partial" : "❌ Weak"}
          </div>
        </div>
        <span className="text-xs text-slate-500">
          {result.isLocal ? "🖥 Local" : "☁️ Cloud"}
        </span>
      </div>

      {/* Comparison slider */}
      <div
        ref={containerRef}
        className="relative w-full h-72 rounded-2xl overflow-hidden cursor-ew-resize select-none bg-black"
        onMouseDown={handleMouseDown}
        onTouchMove={handleTouchMove}
        onTouchStart={(e) => updateSlider(e.touches[0].clientX)}
      >
        {/* Protected (right side) */}
        {protectedUrl && (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={protectedUrl}
            alt="Protected"
            className="absolute inset-0 w-full h-full object-contain"
            draggable={false}
          />
        )}

        {/* Original (left clip) */}
        <div
          className="absolute inset-0 overflow-hidden"
          style={{ width: `${sliderPos}%` }}
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={original}
            alt="Original"
            className="absolute inset-0 w-full h-full object-contain"
            style={{ width: containerRef.current?.clientWidth ?? "100%" }}
            draggable={false}
          />
        </div>

        {/* Divider line */}
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-white/80 shadow-lg pointer-events-none"
          style={{ left: `${sliderPos}%` }}
        >
          {/* Handle */}
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2
                          w-8 h-8 rounded-full bg-white shadow-xl flex items-center justify-center">
            <svg className="w-4 h-4 text-slate-700" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5}
                d="M8 9l-3 3 3 3m8-6l3 3-3 3"/>
            </svg>
          </div>
        </div>

        {/* Labels */}
        <div className="absolute top-3 left-3 px-2 py-1 rounded-md bg-black/60 backdrop-blur text-xs text-white font-medium pointer-events-none">
          Original
        </div>
        <div className="absolute top-3 right-3 px-2 py-1 rounded-md bg-indigo-900/80 backdrop-blur text-xs text-indigo-200 font-medium pointer-events-none">
          🛡️ Protected
        </div>
      </div>

      <p className="text-xs text-slate-500 text-center">Drag the slider to compare · the protection is imperceptible</p>

      {/* Action buttons */}
      <div className="flex gap-3">
        <button
          onClick={onReset}
          className="flex-1 px-4 py-2.5 rounded-xl border border-slate-600 text-slate-300
                     hover:bg-slate-700 transition-colors text-sm font-medium"
        >
          ← New Image
        </button>
        <button
          onClick={handleSave}
          disabled={saving || !protectedUrl}
          className="flex-1 px-4 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500
                     text-white text-sm font-semibold transition-colors disabled:opacity-50
                     flex items-center justify-center gap-2"
        >
          {saving ? (
            <>
              <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
              </svg>
              Saving…
            </>
          ) : (
            <>
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M8 7H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-3m-1 4l-3 3m0 0l-3-3m3 3V4"/>
              </svg>
              Save Image
            </>
          )}
        </button>
      </div>

      {saveMsg && (
        <p className={`text-sm text-center ${saveMsg.startsWith("✓") ? "text-emerald-400" : "text-red-400"}`}>
          {saveMsg}
        </p>
      )}
    </div>
  );
}
