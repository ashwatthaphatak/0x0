// src/lib/modal-client.ts
// ─────────────────────────────────────────────────────────────────────────────
// Client for the Modal.com cloud backend.
// Implements: upload → job polling → result download.
// ─────────────────────────────────────────────────────────────────────────────

import type { CloudJobStatus, ProtectionResult } from "@/types";

// ── Configuration ─────────────────────────────────────────────────────────────
// Set NEXT_PUBLIC_MODAL_BASE_URL in your .env.local to override.
const MODAL_BASE_URL =
  process.env.NEXT_PUBLIC_MODAL_BASE_URL ?? "https://your-modal-endpoint.modal.run";

const POLL_INTERVAL_MS = 2_000;
const MAX_POLL_ATTEMPTS = 300; // 10 minutes max

// ── Types ─────────────────────────────────────────────────────────────────────

interface IngestResponse {
  job_id: string;
  status: string;
}

interface StatusResponse {
  job_id: string;
  status: "pending" | "running" | "complete" | "failed";
  progress?: number;
  result_url?: string;
  score?: number;
  message?: string;
}

// ── API Calls ─────────────────────────────────────────────────────────────────

export async function uploadToModal(
  imageBlob: Blob,
  epsilon: number,
  idToken?: string
): Promise<string> {
  const formData = new FormData();
  formData.append("image",   imageBlob, "input.png");
  formData.append("epsilon", epsilon.toString());

  const headers: Record<string, string> = {};
  if (idToken) {
    headers["Authorization"] = `Bearer ${idToken}`;
  }

  const response = await fetch(`${MODAL_BASE_URL}/ingest`, {
    method:  "POST",
    headers,
    body:    formData,
  });

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(`Upload failed (${response.status}): ${text}`);
  }

  const data: IngestResponse = await response.json();
  if (!data.job_id) throw new Error("Modal did not return a job_id");
  return data.job_id;
}

export async function pollJobStatus(jobId: string): Promise<StatusResponse> {
  const response = await fetch(`${MODAL_BASE_URL}/status/${jobId}`);
  if (!response.ok) {
    throw new Error(`Status check failed (${response.status})`);
  }
  return response.json() as Promise<StatusResponse>;
}

export async function downloadResult(resultUrl: string): Promise<Blob> {
  const response = await fetch(resultUrl);
  if (!response.ok) throw new Error(`Download failed (${response.status})`);
  return response.blob();
}

// ── High-Level Flow ──────────────────────────────────────────────────────────

export async function processInCloud(
  imageBlob: Blob,
  epsilon: number,
  idToken: string | undefined,
  onProgress: (update: { percent: number; message: string }) => void
): Promise<ProtectionResult> {
  // 1. Upload
  onProgress({ percent: 5, message: "Uploading image to cloud…" });
  const jobId = await uploadToModal(imageBlob, epsilon, idToken);

  // 2. Poll
  onProgress({ percent: 15, message: "Job queued – waiting for worker…" });

  let attempts = 0;
  while (attempts < MAX_POLL_ATTEMPTS) {
    await delay(POLL_INTERVAL_MS);
    attempts++;

    const status = await pollJobStatus(jobId);

    const percent = Math.min(
      15 + Math.round(((status.progress ?? 0) / 100) * 75),
      90
    );
    onProgress({ percent, message: statusMessage(status.status) });

    if (status.status === "complete") {
      if (!status.result_url) throw new Error("Job complete but no result URL returned");

      onProgress({ percent: 95, message: "Downloading protected image…" });
      // Return the URL directly; saving is handled by the UI layer
      onProgress({ percent: 100, message: "Complete!" });

      return {
        outputPath: status.result_url,
        score:      status.score ?? 0,
        isLocal:    false,
      };
    }

    if (status.status === "failed") {
      throw new Error(status.message ?? "Cloud processing failed");
    }
  }

  throw new Error("Cloud job timed out after 10 minutes. Please try again.");
}

function statusMessage(status: string): string {
  switch (status) {
    case "pending": return "Job queued – waiting for worker…";
    case "running": return "Worker processing image…";
    case "complete": return "Processing complete!";
    case "failed":  return "Processing failed.";
    default:        return `Status: ${status}`;
  }
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ── Quota / Auth helpers ──────────────────────────────────────────────────────

export async function getAnonymousQuotaKey(): Promise<string> {
  // In production, request a signed anonymous token from your backend.
  // For now, generate a client-side fingerprint (not cryptographically secure).
  const arr = new Uint8Array(16);
  crypto.getRandomValues(arr);
  return Array.from(arr).map((b) => b.toString(16).padStart(2, "0")).join("");
}
