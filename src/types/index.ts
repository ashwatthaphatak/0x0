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

export type DeepfakeAttackType =
  | "black_hair_female_old"
  | "black_hair_female_young"
  | "black_hair_male_old"
  | "black_hair_male_young"
  | "blonde_hair_female_old"
  | "blonde_hair_female_young"
  | "blonde_hair_male_old"
  | "blonde_hair_male_young"
  | "brown_hair_female_old"
  | "brown_hair_female_young"
  | "brown_hair_male_old"
  | "brown_hair_male_young";

export interface DeepfakeAttackOption {
  value: DeepfakeAttackType;
  label: string;
  description: string;
}

export const DEEPFAKE_ATTACK_OPTIONS: DeepfakeAttackOption[] = [
  {
    value: "black_hair_female_old",
    label: "Black Hair + Female + Old",
    description: "Target vector: Black Hair, Female, Old",
  },
  {
    value: "black_hair_female_young",
    label: "Black Hair + Female + Young",
    description: "Target vector: Black Hair, Female, Young",
  },
  {
    value: "black_hair_male_old",
    label: "Black Hair + Male + Old",
    description: "Target vector: Black Hair, Male, Old",
  },
  {
    value: "black_hair_male_young",
    label: "Black Hair + Male + Young",
    description: "Target vector: Black Hair, Male, Young",
  },
  {
    value: "blonde_hair_female_old",
    label: "Blonde Hair + Female + Old",
    description: "Target vector: Blonde Hair, Female, Old",
  },
  {
    value: "blonde_hair_female_young",
    label: "Blonde Hair + Female + Young",
    description: "Target vector: Blonde Hair, Female, Young",
  },
  {
    value: "blonde_hair_male_old",
    label: "Blonde Hair + Male + Old",
    description: "Target vector: Blonde Hair, Male, Old",
  },
  {
    value: "blonde_hair_male_young",
    label: "Blonde Hair + Male + Young",
    description: "Target vector: Blonde Hair, Male, Young",
  },
  {
    value: "brown_hair_female_old",
    label: "Brown Hair + Female + Old",
    description: "Target vector: Brown Hair, Female, Old",
  },
  {
    value: "brown_hair_female_young",
    label: "Brown Hair + Female + Young",
    description: "Target vector: Brown Hair, Female, Young",
  },
  {
    value: "brown_hair_male_old",
    label: "Brown Hair + Male + Old",
    description: "Target vector: Brown Hair, Male, Old",
  },
  {
    value: "brown_hair_male_young",
    label: "Brown Hair + Male + Young",
    description: "Target vector: Brown Hair, Male, Young",
  },
];

export type DeepfakeVerdict = "blocked" | "partial" | "not_blocked" | "unknown";

export interface DeepfakeAttackResult {
  attackType: DeepfakeAttackType;
  attackLabel: string;
  originalFakePath: string;
  sanitizedFakePath: string;
  divergence: number;
  verdict: DeepfakeVerdict;
}
