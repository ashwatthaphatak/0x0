// src/components/compute-toggle.tsx
"use client";

import { useEffect, useState } from "react";
import { checkSidecarReady } from "@/lib/tauri-bridge";
import type { ComputeMode } from "@/types";

interface ComputeToggleProps {
  mode:     ComputeMode;
  onChange: (mode: ComputeMode) => void;
  disabled?: boolean;
}

export function ComputeToggle({ mode, onChange, disabled }: ComputeToggleProps) {
  const [sidecarReady, setSidecarReady] = useState<boolean | null>(null);
  const [checking,     setChecking]     = useState(false);

  // Ping the sidecar whenever the user switches to Local mode
  useEffect(() => {
    if (mode === "local") {
      setChecking(true);
      checkSidecarReady()
        .then((ready) => setSidecarReady(ready))
        .catch(()     => setSidecarReady(false))
        .finally(()   => setChecking(false));
    } else {
      setSidecarReady(null);
    }
  }, [mode]);

  return (
    <div className="flex flex-col gap-2">
      <span className="text-xs font-semibold uppercase tracking-widest text-slate-400">
        Processing Mode
      </span>

      <div className="flex items-center rounded-xl bg-slate-800 p-1 gap-1 shadow-inner">
        {(["local", "cloud"] as ComputeMode[]).map((m) => (
          <button
            key={m}
            disabled={disabled}
            onClick={() => onChange(m)}
            className={[
              "flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-lg",
              "text-sm font-medium transition-all duration-200",
              mode === m
                ? "bg-indigo-600 text-white shadow-lg shadow-indigo-900/50"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-700/60",
              disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer",
            ].join(" ")}
          >
            {m === "local" ? (
              <>
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                </svg>
                Local (Private)
              </>
            ) : (
              <>
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M3 15a4 4 0 004 4h9a5 5 0 10-.1-9.999 5.002 5.002 0 10-9.78 2.096A4.001 4.001 0 003 15z" />
                </svg>
                Cloud (Fast)
              </>
            )}
          </button>
        ))}
      </div>

      {/* Status indicators */}
      <div className="min-h-[20px] flex items-center gap-2 px-1">
        {mode === "local" && checking && (
          <span className="flex items-center gap-1.5 text-xs text-slate-400">
            <span className="w-2 h-2 rounded-full bg-yellow-400 animate-pulse" />
            Checking local engine…
          </span>
        )}
        {mode === "local" && !checking && sidecarReady === true && (
          <span className="flex items-center gap-1.5 text-xs text-emerald-400">
            <span className="w-2 h-2 rounded-full bg-emerald-400" />
            Local engine ready · runs fully offline
          </span>
        )}
        {mode === "local" && !checking && sidecarReady === false && (
          <span className="flex items-center gap-1.5 text-xs text-red-400">
            <span className="w-2 h-2 rounded-full bg-red-400" />
            Local engine unavailable · check Python deps/interpreter or use Cloud mode
          </span>
        )}
        {mode === "cloud" && (
          <span className="flex items-center gap-1.5 text-xs text-sky-400">
            <span className="w-2 h-2 rounded-full bg-sky-400" />
            Cloud mode · requires internet · faster on large images
          </span>
        )}
      </div>
    </div>
  );
}
