// src/hooks/useProtection.ts
// ─────────────────────────────────────────────────────────────────────────────
// React hook that orchestrates the full protection pipeline for both
// Local (Tauri sidecar) and Cloud (Modal) modes.
// ─────────────────────────────────────────────────────────────────────────────

"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import type {
  ComputeMode,
  ProcessingState,
  ProtectionLevel,
  ProtectionResult,
  PROTECTION_LEVELS,
} from "@/types";
import { PROTECTION_LEVELS as LEVELS } from "@/types";
import {
  runLocalProtection,
  listenProtectionProgress,
  openFileDialog,
} from "@/lib/tauri-bridge";
import { processInCloud } from "@/lib/modal-client";

interface UseProtectionOptions {
  mode: ComputeMode;
  level: ProtectionLevel;
  idToken?: string;
}

interface UseProtectionReturn {
  state:       ProcessingState;
  progress:    number;
  statusMsg:   string;
  result:      ProtectionResult | null;
  error:       string | null;
  protect:     (imagePathOrBlob: string | Blob) => Promise<void>;
  reset:       () => void;
}

const LOCAL_PROTECTION_SIZE = 256;

export function useProtection({
  mode,
  level,
  idToken,
}: UseProtectionOptions): UseProtectionReturn {
  const [state,     setState]     = useState<ProcessingState>("idle");
  const [progress,  setProgress]  = useState(0);
  const [statusMsg, setStatusMsg] = useState("");
  const [result,    setResult]    = useState<ProtectionResult | null>(null);
  const [error,     setError]     = useState<string | null>(null);

  const unlistenRef = useRef<(() => void) | null>(null);

  // Clean up event listener on unmount
  useEffect(() => {
    return () => {
      unlistenRef.current?.();
    };
  }, []);

  const protect = useCallback(
    async (input: string | Blob) => {
      setState("processing");
      setProgress(0);
      setStatusMsg("Starting…");
      setError(null);
      setResult(null);

      const epsilon = LEVELS[level].epsilon;

      try {
        if (mode === "local") {
          if (typeof input !== "string") {
            throw new Error("Local mode requires a file path, not a Blob.");
          }

          // Subscribe to progress events from the Rust IPC
          const unlisten = await listenProtectionProgress((update) => {
            if (update.type === "status")   setStatusMsg(update.message ?? "");
            if (update.type === "progress") setProgress(update.percent ?? 0);
            if (update.type === "complete" && update.result) {
              setResult(update.result);
              setState("complete");
            }
            if (update.type === "error") {
              setError(update.message ?? "Unknown local error");
              setState("error");
            }
          });
          unlistenRef.current = unlisten;
          try {
            const res = await runLocalProtection(input, epsilon, undefined, LOCAL_PROTECTION_SIZE);
            // Keep direct invoke result as a fallback if complete event was missed.
            setResult(res);
            setState("complete");
          } finally {
            unlisten();
            unlistenRef.current = null;
          }
        } else {
          // Cloud mode
          if (!(input instanceof Blob)) {
            throw new Error("Cloud mode requires a Blob.");
          }

          const res = await processInCloud(
            input,
            epsilon,
            idToken,
            ({ percent, message }) => {
              setProgress(percent);
              setStatusMsg(message);
            }
          );

          setResult(res);
          setState("complete");
        }
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        setError(msg);
        setState("error");
        throw err instanceof Error ? err : new Error(msg);
      }
    },
    [mode, level, idToken]
  );

  const reset = useCallback(() => {
    unlistenRef.current?.();
    unlistenRef.current = null;
    setState("idle");
    setProgress(0);
    setStatusMsg("");
    setResult(null);
    setError(null);
  }, []);

  return { state, progress, statusMsg, result, error, protect, reset };
}
