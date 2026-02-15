"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  DEEPFAKE_ATTACK_OPTIONS,
  type DeepfakeAttackResult,
  type DeepfakeAttackType,
} from "@/types";
import { localPathToUrl, runLocalDeepfakeTest } from "@/lib/tauri-bridge";

interface AttackTesterProps {
  originalPath?: string;
  sanitizedPath: string;
  isLocalResult: boolean;
}

export function AttackTester({ originalPath, sanitizedPath, isLocalResult }: AttackTesterProps) {
  const [attackType, setAttackType] = useState<DeepfakeAttackType>("blonde_hair_female_old");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<DeepfakeAttackResult | null>(null);
  const [originalFakeUrl, setOriginalFakeUrl] = useState<string>("");
  const [sanitizedFakeUrl, setSanitizedFakeUrl] = useState<string>("");
  const [expandedView, setExpandedView] = useState<"original" | "sanitized" | null>(null);
  const objectUrlsRef = useRef<string[]>([]);

  const releaseObjectUrls = useCallback(() => {
    for (const url of objectUrlsRef.current) URL.revokeObjectURL(url);
    objectUrlsRef.current = [];
  }, []);

  const resolveLocalImage = useCallback(async (path: string): Promise<string> => {
    try {
      const { readFile } = await import("@tauri-apps/plugin-fs");
      const bytes = await readFile(path);
      const blobUrl = URL.createObjectURL(new Blob([bytes], { type: "image/png" }));
      objectUrlsRef.current.push(blobUrl);
      return blobUrl;
    } catch {
      return localPathToUrl(path);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    releaseObjectUrls();
    setOriginalFakeUrl("");
    setSanitizedFakeUrl("");
    setExpandedView(null);

    if (!result) return;

    const resolve = async () => {
      try {
        const [orig, sani] = await Promise.all([
          resolveLocalImage(result.originalFakePath),
          resolveLocalImage(result.sanitizedFakePath),
        ]);
        if (!cancelled) {
          setOriginalFakeUrl(orig);
          setSanitizedFakeUrl(sani);
        }
      } catch (err: unknown) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Could not preview attack outputs.");
        }
      }
    };

    void resolve();
    return () => {
      cancelled = true;
      releaseObjectUrls();
    };
  }, [releaseObjectUrls, resolveLocalImage, result]);

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

  const canRun = isLocalResult && Boolean(originalPath && sanitizedPath);

  const handleRun = useCallback(async () => {
    if (!originalPath || !sanitizedPath) {
      setError("Original/sanitized paths are missing. Load an image from disk and sanitize again.");
      return;
    }
    if (originalPath === sanitizedPath) {
      setError("Original and sanitized paths are identical. Re-run sanitization before testing.");
      return;
    }

    setRunning(true);
    setError(null);
    setResult(null);
    setOriginalFakeUrl("");
    setSanitizedFakeUrl("");

    try {
      const res = await runLocalDeepfakeTest(originalPath, sanitizedPath, attackType);
      setResult(res);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setRunning(false);
    }
  }, [attackType, originalPath, sanitizedPath]);

  const verdictStyle =
    result?.verdict === "blocked"
      ? "border-emerald-700 bg-emerald-900/30 text-emerald-300"
      : result?.verdict === "partial"
        ? "border-yellow-700 bg-yellow-900/30 text-yellow-300"
        : "border-red-700 bg-red-900/30 text-red-300";

  const verdictText =
    result?.verdict === "blocked"
      ? "Blocked"
      : result?.verdict === "partial"
        ? "Partial"
        : result?.verdict === "not_blocked"
          ? "Not Blocked"
          : "Unknown";

  return (
    <div className="rounded-2xl border border-slate-700/80 bg-slate-900/70 p-4">
      <div className="mb-4 flex items-start justify-between gap-4">
        <div>
          <h3 className="text-sm font-semibold text-slate-200">Deepfake Test (StarGAN)</h3>
          <p className="mt-1 text-xs text-slate-400">
            Run the same deepfake attack on original and sanitized images to compare outcomes.
          </p>
        </div>
      </div>

      {!canRun && (
        <div className="rounded-xl border border-amber-700 bg-amber-950/30 p-3 text-xs text-amber-300">
          Deepfake test requires Local Mode with a disk-backed input image and local sanitized output.
        </div>
      )}

      <div className="mt-3 flex flex-col gap-3 sm:flex-row sm:items-end">
        <label className="flex-1 text-xs text-slate-400">
          Attack Type
          <select
            value={attackType}
            onChange={(e) => setAttackType(e.target.value as DeepfakeAttackType)}
            disabled={!canRun || running}
            className="mt-1 block w-full rounded-lg border border-slate-600 bg-slate-800 px-3 py-2 text-sm text-slate-100 outline-none focus:border-indigo-500"
          >
            {DEEPFAKE_ATTACK_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </label>

        <button
          onClick={handleRun}
          disabled={!canRun || running}
          className="rounded-xl bg-indigo-600 px-4 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-indigo-500 disabled:opacity-50"
        >
          {running ? "Running Attack..." : "Run Deepfake Test"}
        </button>
      </div>

      <p className="mt-2 text-xs text-slate-500">
        {DEEPFAKE_ATTACK_OPTIONS.find((opt) => opt.value === attackType)?.description}
      </p>

      {error && <p className="mt-3 text-xs text-red-400">{error}</p>}

      {result && (
        <div className="mt-4 flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <div className="text-xs text-slate-400">
              Divergence: <span className="font-mono text-slate-200">{result.divergence.toFixed(4)}</span>
            </div>
            <div className={`rounded-full border px-3 py-1 text-xs font-semibold ${verdictStyle}`}>
              Verdict: {verdictText}
            </div>
          </div>

          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <div
              className={[
                "rounded-xl border border-red-700/60 bg-red-950/20 p-3 transition-colors",
                originalFakeUrl ? "cursor-zoom-in hover:border-red-600/80" : "cursor-default",
              ].join(" ")}
              onClick={() => {
                if (originalFakeUrl) setExpandedView("original");
              }}
              role={originalFakeUrl ? "button" : undefined}
              tabIndex={originalFakeUrl ? 0 : -1}
              onKeyDown={(event) => {
                if (!originalFakeUrl) return;
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  setExpandedView("original");
                }
              }}
              aria-label={originalFakeUrl ? "Enlarge deepfaked original image" : undefined}
            >
              <div className="mb-2 text-xs font-medium uppercase tracking-wider text-red-300">
                Original - Deepfaked
              </div>
              <div className="relative h-56 overflow-hidden rounded-lg bg-black">
                {originalFakeUrl ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={originalFakeUrl} alt="Deepfaked original" className="h-full w-full object-contain" />
                ) : (
                  <div className="flex h-full items-center justify-center text-xs text-slate-500">Loading...</div>
                )}
              </div>
            </div>

            <div
              className={[
                "rounded-xl border border-emerald-700/60 bg-emerald-950/20 p-3 transition-colors",
                sanitizedFakeUrl ? "cursor-zoom-in hover:border-emerald-600/80" : "cursor-default",
              ].join(" ")}
              onClick={() => {
                if (sanitizedFakeUrl) setExpandedView("sanitized");
              }}
              role={sanitizedFakeUrl ? "button" : undefined}
              tabIndex={sanitizedFakeUrl ? 0 : -1}
              onKeyDown={(event) => {
                if (!sanitizedFakeUrl) return;
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  setExpandedView("sanitized");
                }
              }}
              aria-label={sanitizedFakeUrl ? "Enlarge deepfake-attempt image" : undefined}
            >
              <div className="mb-2 text-xs font-medium uppercase tracking-wider text-emerald-300">
                Sanitized - Deepfake Attempt
              </div>
              <div className="relative h-56 overflow-hidden rounded-lg bg-black">
                {sanitizedFakeUrl ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={sanitizedFakeUrl}
                    alt="Deepfake attempt on sanitized"
                    className="h-full w-full object-contain"
                  />
                ) : (
                  <div className="flex h-full items-center justify-center text-xs text-slate-500">Loading...</div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

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
              <img src={originalFakeUrl} alt="Deepfaked original expanded" className="h-full w-full object-contain" />
            ) : (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={sanitizedFakeUrl} alt="Deepfake attempt expanded" className="h-full w-full object-contain" />
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
