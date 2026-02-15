// src/api/modalClient.ts

/**
 * Modal Backend API Client
 * Handles cloud-based image protection via Modal serverless backend
 */

const MODAL_API_URL = "https://akshay-3046--deepfake-defense-web.modal.run";

export interface ProtectionJobStatus {
  job_id: string;
  status: "pending" | "running" | "complete" | "failed";
  progress: number;
  result_url?: string;
  score?: number;
  message: string;
}

export interface ProtectionResult {
  imageUrl: string;
  score: number;
  jobId: string;
}

/**
 * Check if Modal backend is healthy
 */
export async function checkModalHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${MODAL_API_URL}/health`, {
      method: "GET",
      signal: AbortSignal.timeout(5000),
    });
    const data = await response.json();
    return data.status === "ok";
  } catch (error) {
    console.error("Modal health check failed:", error);
    return false;
  }
}

/**
 * Submit an image for protection to Modal backend
 */
export async function submitImageForProtection(
  imageFile: File,
  epsilon: number = 0.05
): Promise<string> {
  const formData = new FormData();
  formData.append("image", imageFile);
  formData.append("epsilon", epsilon.toString());

  const response = await fetch(`${MODAL_API_URL}/ingest`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Failed to submit image: ${error}`);
  }

  const data = await response.json();
  return data.job_id;
}

/**
 * Poll job status until complete
 */
export async function pollJobStatus(
  jobId: string,
  onProgress?: (status: ProtectionJobStatus) => void
): Promise<ProtectionJobStatus> {
  const maxAttempts = 120; // 2 minutes max (120 * 1s)
  let attempts = 0;

  while (attempts < maxAttempts) {
    const response = await fetch(`${MODAL_API_URL}/status/${jobId}`);
    
    if (!response.ok) {
      throw new Error(`Failed to get job status: ${response.statusText}`);
    }

    const status: ProtectionJobStatus = await response.json();
    
    if (onProgress) {
      onProgress(status);
    }

    if (status.status === "complete" || status.status === "failed") {
      return status;
    }

    // Wait 1 second before next poll
    await new Promise((resolve) => setTimeout(resolve, 1000));
    attempts++;
  }

  throw new Error("Job timeout: exceeded maximum polling time");
}

/**
 * Complete protection workflow: submit → poll → return result
 */
export async function protectImageCloud(
  imageFile: File,
  epsilon: number = 0.05,
  onProgress?: (progress: number, message: string) => void
): Promise<ProtectionResult> {
  // Submit image
  if (onProgress) onProgress(5, "Submitting to cloud...");
  const jobId = await submitImageForProtection(imageFile, epsilon);

  // Poll for completion
  const finalStatus = await pollJobStatus(jobId, (status) => {
    if (onProgress) {
      onProgress(status.progress, status.message);
    }
  });

  if (finalStatus.status === "failed") {
    throw new Error(`Protection failed: ${finalStatus.message}`);
  }

  if (!finalStatus.result_url || !finalStatus.score) {
    throw new Error("Protection completed but no result returned");
  }

  return {
    imageUrl: finalStatus.result_url,
    score: finalStatus.score,
    jobId,
  };
}

/**
 * Download base64 data URL as a file
 */
export function downloadDataUrl(dataUrl: string, filename: string): void {
  const link = document.createElement("a");
  link.href = dataUrl;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}