// src/components/ProtectionForm.tsx

import { useState } from "react";
import { invoke } from "@tauri-apps/api/tauri";
import { open } from "@tauri-apps/api/dialog";
import {
  protectImageCloud,
  checkModalHealth,
  downloadDataUrl,
} from "../api/modalClient";

type ProcessingMode = "cloud" | "local";

interface ProtectionState {
  isProcessing: boolean;
  progress: number;
  message: string;
  resultImage?: string;
  score?: number;
  error?: string;
}

export default function ProtectionForm() {
  const [mode, setMode] = useState<ProcessingMode>("cloud");
  const [epsilon, setEpsilon] = useState(0.05);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [state, setState] = useState<ProtectionState>({
    isProcessing: false,
    progress: 0,
    message: "",
  });
  const [cloudAvailable, setCloudAvailable] = useState<boolean | null>(null);

  // Check cloud availability on mount
  useState(() => {
    checkModalHealth().then(setCloudAvailable);
  });

  const handleFileSelect = async () => {
    const selected = await open({
      multiple: false,
      filters: [
        {
          name: "Images",
          extensions: ["png", "jpg", "jpeg", "webp"],
        },
      ],
    });

    if (selected && typeof selected === "string") {
      // Convert file path to File object
      const response = await fetch(`file://${selected}`);
      const blob = await response.blob();
      const file = new File([blob], selected.split("/").pop() || "image.png", {
        type: blob.type,
      });
      setSelectedFile(file);
    }
  };

  const handleProtectCloud = async () => {
    if (!selectedFile) return;

    setState({
      isProcessing: true,
      progress: 0,
      message: "Starting cloud protection...",
    });

    try {
      const result = await protectImageCloud(
        selectedFile,
        epsilon,
        (progress, message) => {
          setState((prev) => ({
            ...prev,
            progress,
            message,
          }));
        }
      );

      setState({
        isProcessing: false,
        progress: 100,
        message: "Protection complete!",
        resultImage: result.imageUrl,
        score: result.score,
      });
    } catch (error) {
      setState({
        isProcessing: false,
        progress: 0,
        message: "",
        error: error instanceof Error ? error.message : "Unknown error",
      });
    }
  };

  const handleProtectLocal = async () => {
    if (!selectedFile) return;

    setState({
      isProcessing: true,
      progress: 0,
      message: "Starting local protection...",
    });

    try {
      // Call Rust backend which will invoke Python sidecar
      const result = await invoke<{
        path: string;
        score: number;
      }>("protect_image_local", {
        inputPath: selectedFile.name, // You'll need to handle file path properly
        epsilon,
      });

      // Convert local file to data URL for display
      const fileUrl = `file://${result.path}`;
      
      setState({
        isProcessing: false,
        progress: 100,
        message: "Protection complete!",
        resultImage: fileUrl,
        score: result.score,
      });
    } catch (error) {
      setState({
        isProcessing: false,
        progress: 0,
        message: "",
        error: error instanceof Error ? error.message : "Local processing failed",
      });
    }
  };

  const handleDownload = () => {
    if (state.resultImage) {
      downloadDataUrl(
        state.resultImage,
        `protected-${Date.now()}.png`
      );
    }
  };

  return (
    <div className="protection-form">
      <h2>DeepFake Defense</h2>

      {/* Mode Selector */}
      <div className="mode-selector">
        <label>
          <input
            type="radio"
            value="cloud"
            checked={mode === "cloud"}
            onChange={(e) => setMode(e.target.value as ProcessingMode)}
            disabled={cloudAvailable === false}
          />
          Cloud Mode (Modal)
          {cloudAvailable === false && " (Unavailable)"}
          {cloudAvailable === true && " ✅"}
        </label>
        <label>
          <input
            type="radio"
            value="local"
            checked={mode === "local"}
            onChange={(e) => setMode(e.target.value as ProcessingMode)}
          />
          Local Mode (Desktop)
        </label>
      </div>

      {/* Epsilon Control */}
      <div className="epsilon-control">
        <label>
          Protection Strength: {epsilon.toFixed(3)}
          <input
            type="range"
            min="0.001"
            max="0.2"
            step="0.001"
            value={epsilon}
            onChange={(e) => setEpsilon(parseFloat(e.target.value))}
            disabled={state.isProcessing}
          />
        </label>
      </div>

      {/* File Selection */}
      <div className="file-selector">
        <button onClick={handleFileSelect} disabled={state.isProcessing}>
          {selectedFile ? `Selected: ${selectedFile.name}` : "Choose Image"}
        </button>
      </div>

      {/* Process Button */}
      <button
        onClick={mode === "cloud" ? handleProtectCloud : handleProtectLocal}
        disabled={!selectedFile || state.isProcessing}
        className="process-btn"
      >
        {state.isProcessing ? "Processing..." : "Protect Image"}
      </button>

      {/* Progress */}
      {state.isProcessing && (
        <div className="progress">
          <div className="progress-bar" style={{ width: `${state.progress}%` }} />
          <p>{state.message}</p>
        </div>
      )}

      {/* Error */}
      {state.error && (
        <div className="error">
          <p>❌ {state.error}</p>
        </div>
      )}

      {/* Result */}
      {state.resultImage && (
        <div className="result">
          <h3>Protected Image</h3>
          <img src={state.resultImage} alt="Protected" />
          <p>Protection Score: {state.score?.toFixed(1)}/100</p>
          <button onClick={handleDownload}>Download</button>
        </div>
      )}
    </div>
  );
}