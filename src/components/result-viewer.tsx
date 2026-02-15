"use client";

import { useCallback, useEffect, useRef, useState, type CSSProperties, type PointerEvent } from "react";
import type { ProtectionResult } from "@/types";
import { localPathToUrl, saveFileDialog, saveResultToPath } from "@/lib/tauri-bridge";

interface ResultViewerProps {
  original: string;
  result: ProtectionResult;
  onReset: () => void;
}

const RIPPLE_RADIUS_MIN_PX = 40;
const RIPPLE_RADIUS_MAX_PX = 90;
const RIPPLE_FALLOFF_MIN_PX = 105;
const RIPPLE_FALLOFF_MAX_PX = 220;

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function jetColor(value: number): { r: number; g: number; b: number } {
  const x = clamp01(value);
  const stops = [
    { p: 0.0, c: [8, 35, 255] },
    { p: 0.35, c: [0, 196, 255] },
    { p: 0.65, c: [255, 230, 0] },
    { p: 1.0, c: [255, 55, 0] },
  ] as const;

  for (let i = 0; i < stops.length - 1; i++) {
    const left = stops[i];
    const right = stops[i + 1];
    if (x >= left.p && x <= right.p) {
      const t = (x - left.p) / (right.p - left.p || 1);
      return {
        r: Math.round(left.c[0] + (right.c[0] - left.c[0]) * t),
        g: Math.round(left.c[1] + (right.c[1] - left.c[1]) * t),
        b: Math.round(left.c[2] + (right.c[2] - left.c[2]) * t),
      };
    }
  }

  return { r: 255, g: 55, b: 0 };
}

async function loadBitmap(src: string): Promise<ImageBitmap> {
  const response = await fetch(src);
  if (!response.ok) {
    throw new Error(`Failed to load image: ${response.status}`);
  }
  const blob = await response.blob();
  return createImageBitmap(blob);
}

async function buildHeatmapOverlay(originalSrc: string, protectedSrc: string): Promise<string> {
  const [originalBitmap, protectedBitmap] = await Promise.all([
    loadBitmap(originalSrc),
    loadBitmap(protectedSrc),
  ]);

  try {
    const width = Math.max(64, Math.min(1024, protectedBitmap.width || originalBitmap.width));
    const height = Math.max(64, Math.min(1024, protectedBitmap.height || originalBitmap.height));
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;

    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) {
      throw new Error("Canvas context unavailable");
    }

    ctx.drawImage(originalBitmap, 0, 0, width, height);
    const originalPixels = ctx.getImageData(0, 0, width, height);

    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(protectedBitmap, 0, 0, width, height);
    const protectedPixels = ctx.getImageData(0, 0, width, height);

    const output = ctx.createImageData(width, height);
    const deltas = new Float32Array(width * height);
    let maxDelta = 0;

    for (let idx = 0, px = 0; idx < originalPixels.data.length; idx += 4, px++) {
      const dr = Math.abs(originalPixels.data[idx] - protectedPixels.data[idx]);
      const dg = Math.abs(originalPixels.data[idx + 1] - protectedPixels.data[idx + 1]);
      const db = Math.abs(originalPixels.data[idx + 2] - protectedPixels.data[idx + 2]);

      const delta = (dr + dg + db) / 765;
      deltas[px] = delta;
      if (delta > maxDelta) maxDelta = delta;
    }

    const floor = Math.max(0.01, maxDelta * 0.08);
    const span = Math.max(0.0001, maxDelta - floor);

    for (let idx = 0, px = 0; idx < output.data.length; idx += 4, px++) {
      const normalized = clamp01((deltas[px] - floor) / span);
      if (normalized <= 0) {
        output.data[idx + 3] = 0;
        continue;
      }

      const intensity = Math.pow(normalized, 0.35);
      const color = jetColor(Math.min(1, intensity * 1.45));
      const originalR = originalPixels.data[idx];
      const originalG = originalPixels.data[idx + 1];
      const originalB = originalPixels.data[idx + 2];
      const blend = 0.99;

      output.data[idx] = Math.round(originalR * (1 - blend) + color.r * blend);
      output.data[idx + 1] = Math.round(originalG * (1 - blend) + color.g * blend);
      output.data[idx + 2] = Math.round(originalB * (1 - blend) + color.b * blend);
      output.data[idx + 3] = Math.round(255 * Math.pow(intensity, 0.5));
    }

    ctx.clearRect(0, 0, width, height);
    ctx.putImageData(output, 0, 0);
    return canvas.toDataURL("image/png");
  } finally {
    originalBitmap.close();
    protectedBitmap.close();
  }
}

interface InteractiveHeatmapPreviewProps {
  originalSrc: string;
  protectedSrc: string;
  onImageError: () => void;
}

function InteractiveHeatmapPreview({ originalSrc, protectedSrc, onImageError }: InteractiveHeatmapPreviewProps) {
  const [heatmapUrl, setHeatmapUrl] = useState<string | null>(null);
  const [isHovering, setIsHovering] = useState(false);
  const [cursor, setCursor] = useState({ x: 50, y: 50 });
  const [canHover, setCanHover] = useState(false);
  const [rippleSize, setRippleSize] = useState({
    radius: 46,
    falloff: 120,
    ringA: 7,
    ringB: 13,
    ringC: 22,
  });
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (typeof window === "undefined") return;
    setCanHover(window.matchMedia("(hover: hover) and (pointer: fine)").matches);
  }, []);

  useEffect(() => {
    let cancelled = false;
    setHeatmapUrl(null);

    const build = async () => {
      try {
        const overlay = await buildHeatmapOverlay(originalSrc, protectedSrc);
        if (!cancelled) {
          setHeatmapUrl(overlay);
        }
      } catch {
        if (!cancelled) {
          setHeatmapUrl(null);
        }
      }
    };

    void build();
    return () => {
      cancelled = true;
    };
  }, [originalSrc, protectedSrc]);

  useEffect(() => {
    if (!containerRef.current || typeof ResizeObserver === "undefined") return;

    const element = containerRef.current;
    const updateSize = (width: number, height: number) => {
      const minDim = Math.max(120, Math.min(width, height));
      const radius = Math.round(clamp(minDim * 0.11, RIPPLE_RADIUS_MIN_PX, RIPPLE_RADIUS_MAX_PX));
      const falloff = Math.round(clamp(minDim * 0.29, RIPPLE_FALLOFF_MIN_PX, RIPPLE_FALLOFF_MAX_PX));
      const ringA = Math.max(6, Math.round(falloff * 0.06));
      const ringB = ringA + Math.max(5, Math.round(falloff * 0.05));
      const ringC = ringB + Math.max(7, Math.round(falloff * 0.075));
      setRippleSize({ radius, falloff, ringA, ringB, ringC });
    };

    updateSize(element.clientWidth, element.clientHeight);

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) return;
      updateSize(entry.contentRect.width, entry.contentRect.height);
    });

    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const updateCursor = useCallback((event: PointerEvent<HTMLDivElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    if (!rect.width || !rect.height) return;

    const x = clamp01((event.clientX - rect.left) / rect.width) * 100;
    const y = clamp01((event.clientY - rect.top) / rect.height) * 100;
    setCursor({ x, y });
  }, []);

  const overlayMask: CSSProperties | undefined =
    heatmapUrl && canHover
      ? {
          WebkitMaskImage: `radial-gradient(circle ${rippleSize.falloff}px at ${cursor.x}% ${cursor.y}%, rgba(0, 0, 0, 1) 0%, rgba(0, 0, 0, 1) ${Math.round((rippleSize.radius / rippleSize.falloff) * 100)}%, rgba(0, 0, 0, 0.72) 38%, rgba(0, 0, 0, 0.2) 48%, rgba(0, 0, 0, 0) 58%)`,
          maskImage: `radial-gradient(circle ${rippleSize.falloff}px at ${cursor.x}% ${cursor.y}%, rgba(0, 0, 0, 1) 0%, rgba(0, 0, 0, 1) ${Math.round((rippleSize.radius / rippleSize.falloff) * 100)}%, rgba(0, 0, 0, 0.72) 38%, rgba(0, 0, 0, 0.2) 48%, rgba(0, 0, 0, 0) 58%)`,
        }
      : undefined;
  const dimmingOverlay: CSSProperties | undefined =
    heatmapUrl && canHover
      ? {
          ...overlayMask,
          backgroundImage: [
            `radial-gradient(circle at ${cursor.x}% ${cursor.y}%, rgba(0, 0, 0, 0.82) 0%, rgba(0, 0, 0, 0.62) 18%, rgba(0, 0, 0, 0.36) 34%, rgba(0, 0, 0, 0.22) 49%, rgba(0, 0, 0, 0.1) 60%, rgba(0, 0, 0, 0) 74%)`,
            `repeating-radial-gradient(circle at ${cursor.x}% ${cursor.y}%, rgba(0, 0, 0, 0.16) 0px, rgba(0, 0, 0, 0.16) ${rippleSize.ringA}px, rgba(0, 0, 0, 0.04) ${rippleSize.ringB}px, rgba(0, 0, 0, 0.04) ${rippleSize.ringC}px)`,
          ].join(", "),
        }
      : undefined;

  return (
    <div
      ref={containerRef}
      className="relative h-full w-full"
      onPointerEnter={(event) => {
        if (!heatmapUrl || !canHover) return;
        setIsHovering(true);
        updateCursor(event);
      }}
      onPointerMove={(event) => {
        if (!heatmapUrl || !canHover) return;
        updateCursor(event);
      }}
      onPointerLeave={() => setIsHovering(false)}
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={protectedSrc}
        alt="Sanitized"
        className="h-full w-full object-contain"
        onError={onImageError}
      />

      {heatmapUrl && canHover && (
        <>
          <div
            aria-hidden
            className={[
              "pointer-events-none absolute inset-0 transition-opacity duration-150",
              isHovering ? "opacity-100" : "opacity-0",
            ].join(" ")}
            style={dimmingOverlay}
          />
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={heatmapUrl}
            alt=""
            aria-hidden
            className={[
              "pointer-events-none absolute inset-0 h-full w-full object-contain transition-opacity duration-150",
              isHovering ? "opacity-100" : "opacity-0",
            ].join(" ")}
            style={overlayMask}
          />
          <div className="pointer-events-none absolute bottom-2 left-2 rounded-md bg-black/60 px-2 py-1 text-[10px] font-medium text-indigo-200">
            Hover to inspect affected-pixel heatmap
          </div>
        </>
      )}
    </div>
  );
}

export function ResultViewer({ original, result, onReset }: ResultViewerProps) {
  const [protectedUrl, setProtectedUrl] = useState<string>("");
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState<string | null>(null);
  const [previewErr, setPreviewErr] = useState<string | null>(null);
  const [expandedView, setExpandedView] = useState<"original" | "sanitized" | null>(null);
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
    setExpandedView(null);
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

  useEffect(() => {
    if (!expandedView) return;

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setExpandedView(null);
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [expandedView]);

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
        <div
          className="rounded-2xl border border-slate-700/80 bg-slate-900/70 p-3 transition-colors hover:border-slate-500/80 cursor-zoom-in"
          onClick={() => setExpandedView("original")}
          role="button"
          tabIndex={0}
          onKeyDown={(event) => {
            if (event.key === "Enter" || event.key === " ") {
              event.preventDefault();
              setExpandedView("original");
            }
          }}
          aria-label="Enlarge original image"
        >
          <div className="mb-2 text-xs font-medium uppercase tracking-wider text-slate-400">Original</div>
          <div className="relative h-64 overflow-hidden rounded-xl bg-black">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={original} alt="Original" className="h-full w-full object-contain" />
          </div>
        </div>

        <div
          className={[
            "rounded-2xl border border-indigo-700/70 bg-indigo-950/20 p-3 transition-colors",
            protectedUrl ? "cursor-zoom-in hover:border-indigo-500/80" : "cursor-default",
          ].join(" ")}
          onClick={() => {
            if (protectedUrl) setExpandedView("sanitized");
          }}
          role={protectedUrl ? "button" : undefined}
          tabIndex={protectedUrl ? 0 : -1}
          onKeyDown={(event) => {
            if (!protectedUrl) return;
            if (event.key === "Enter" || event.key === " ") {
              event.preventDefault();
              setExpandedView("sanitized");
            }
          }}
          aria-label={protectedUrl ? "Enlarge sanitized image" : undefined}
        >
          <div className="mb-2 text-xs font-medium uppercase tracking-wider text-indigo-300">Sanitized</div>
          <div className="relative h-64 overflow-hidden rounded-xl bg-black">
            {protectedUrl ? (
              <InteractiveHeatmapPreview
                originalSrc={original}
                protectedSrc={protectedUrl}
                onImageError={() => {
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

      {expandedView && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/85 p-4"
          onClick={() => setExpandedView(null)}
        >
          <div
            className="relative flex h-[92vh] w-[95vw] max-w-7xl items-center justify-center overflow-hidden rounded-2xl border border-slate-700 bg-black"
            onClick={(event) => event.stopPropagation()}
          >
            {expandedView === "original" ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={original} alt="Original expanded" className="h-full w-full object-contain" />
            ) : (
              protectedUrl && (
                <InteractiveHeatmapPreview
                  originalSrc={original}
                  protectedSrc={protectedUrl}
                  onImageError={() => {
                    setExpandedView(null);
                    setProtectedUrl("");
                    setPreviewErr("Could not preview sanitized image.");
                  }}
                />
              )
            )}

            <button
              type="button"
              onClick={() => setExpandedView(null)}
              className="absolute right-3 top-3 rounded-md border border-white/20 bg-black/60 px-2.5 py-1.5 text-xs font-medium text-white transition-colors hover:bg-black/80"
              aria-label="Close expanded preview"
              title="Close"
            >
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
