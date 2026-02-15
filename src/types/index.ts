// src/types/index.ts

export type ComputeMode = "local" | "cloud";

export type ProtectionLevel = "low" | "medium" | "high";

export const PROTECTION_LEVELS: Record<ProtectionLevel, { label: string; epsilon: number; description: string }> = {
  low:    { label: "Low",    epsilon: 0.02, description: "Subtle – PSNR ≥ 42 dB" },
  medium: { label: "Medium", epsilon: 0.05, description: "Balanced – PSNR ≈ 38 dB" },
  high:   { label: "High",   epsilon: 0.08, description: "Maximum protection, slight noise" },
};

export type ProcessingState =
  | "idle"
  | "cropping"
  | "processing"
  | "complete"
  | "error";

export interface ProtectionResult {
  outputPath: string;   // local file path (local mode) or URL (cloud mode)
  score: number;        // 0–100
  isLocal: boolean;
}

export interface ProgressUpdate {
  type: "status" | "progress" | "complete" | "error";
  message?: string;
  percent?: number;
  result?: ProtectionResult;
}

// Cloud API types
export interface CloudJobStatus {
  jobId: string;
  status: "pending" | "running" | "complete" | "failed";
  percent?: number;
  resultUrl?: string;
  score?: number;
  message?: string;
}

export interface Metrics {
  PSNR: number;
  SSIM: number;
  L2: number;
}
