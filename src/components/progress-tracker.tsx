// src/components/progress-tracker.tsx
"use client";

import type { ProcessingState } from "@/types";

interface ProgressTrackerProps {
  state:      ProcessingState;
  percent:    number;
  statusMsg:  string;
  error?:     string | null;
  mode:       "local" | "cloud";
}

const STEPS_LOCAL = [
  "Loading model",
  "Generating attention map",
  "Extracting texture features",
  "Injecting perturbation",
  "Saving result",
];

const STEPS_CLOUD = [
  "Uploading image",
  "Job queued",
  "Worker processing",
  "Downloading result",
  "Complete",
];

function stepIndex(percent: number): number {
  return Math.min(Math.floor(percent / 20), 4);
}

export function ProgressTracker({
  state,
  percent,
  statusMsg,
  error,
  mode,
}: ProgressTrackerProps) {
  if (state === "idle") return null;

  const steps    = mode === "local" ? STEPS_LOCAL : STEPS_CLOUD;
  const curStep  = stepIndex(percent);
  const isError  = state === "error";
  const isDone   = state === "complete";

  return (
    <div className="flex flex-col gap-4 w-full">
      {/* Status message */}
      <div className="flex items-center gap-2.5">
        {isError ? (
          <div className="w-5 h-5 rounded-full bg-red-500 flex items-center justify-center flex-shrink-0">
            <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M6 18L18 6M6 6l12 12"/>
            </svg>
          </div>
        ) : isDone ? (
          <div className="w-5 h-5 rounded-full bg-emerald-500 flex items-center justify-center flex-shrink-0">
            <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7"/>
            </svg>
          </div>
        ) : (
          <svg className="w-5 h-5 text-indigo-400 animate-spin flex-shrink-0" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
            <path className="opacity-75" fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
          </svg>
        )}
        <span className={[
          "text-sm font-medium",
          isError ? "text-red-400" : isDone ? "text-emerald-400" : "text-slate-300",
        ].join(" ")}>
          {isError ? (error ?? "An error occurred") : (statusMsg || (isDone ? "Protection complete!" : "Processing…"))}
        </span>
      </div>

      {/* Progress bar */}
      <div className="w-full bg-slate-700 rounded-full h-2.5 overflow-hidden">
        <div
          className={[
            "h-full rounded-full transition-all duration-500 ease-out",
            isError ? "bg-red-500"
              : isDone ? "bg-emerald-500"
              : "bg-gradient-to-r from-indigo-500 to-violet-500",
          ].join(" ")}
          style={{ width: `${isError ? 100 : percent}%` }}
        />
      </div>

      {/* Step indicators */}
      <div className="flex justify-between">
        {steps.map((label, i) => (
          <div key={label} className="flex flex-col items-center gap-1 flex-1">
            <div
              className={[
                "w-2.5 h-2.5 rounded-full transition-colors duration-300",
                i < curStep  ? "bg-indigo-400"
                  : i === curStep && !isDone && !isError ? "bg-indigo-400 animate-pulse"
                  : isDone    ? "bg-emerald-400"
                  : "bg-slate-600",
              ].join(" ")}
            />
            <span className="text-[10px] text-slate-500 text-center leading-tight hidden sm:block">
              {label}
            </span>
          </div>
        ))}
      </div>

      {/* Percentage */}
      <div className="flex justify-between text-xs text-slate-500">
        <span>
          {mode === "local" ? "🖥 Running locally" : "☁️ Processing in cloud"}
        </span>
        <span className="font-mono">{percent}%</span>
      </div>
    </div>
  );
}
