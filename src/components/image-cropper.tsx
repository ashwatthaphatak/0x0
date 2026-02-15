// src/components/image-cropper.tsx
"use client";

import { useState, useCallback } from "react";
import Cropper from "react-easy-crop";

type Point = { x: number; y: number };
type Area  = { x: number; y: number; width: number; height: number };

interface ImageCropperProps {
  imageSrc: string;
  onConfirm: (croppedBlob: Blob, croppedDataUrl: string) => void;
  onCancel:  () => void;
}

function loadImageElement(imageSrc: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Image element load failed"));
    image.src = imageSrc;
  });
}

async function loadImageForCanvas(imageSrc: string): Promise<CanvasImageSource> {
  try {
    const response = await fetch(imageSrc);
    if (!response.ok) {
      throw new Error(`Fetch failed with status ${response.status}`);
    }
    const blob = await response.blob();
    return await createImageBitmap(blob);
  } catch (err) {
    console.warn("Falling back to HTMLImageElement for crop source:", err);
    return await loadImageElement(imageSrc);
  }
}

async function getCroppedImg(
  imageSrc: string,
  pixelCrop: Area,
  outputSize = 1024
): Promise<{ blob: Blob; dataUrl: string }> {
  const image = await loadImageForCanvas(imageSrc);

  const canvas       = document.createElement("canvas");
  canvas.width       = outputSize;
  canvas.height      = outputSize;
  const ctx          = canvas.getContext("2d")!;

  ctx.drawImage(
    image,
    pixelCrop.x,
    pixelCrop.y,
    pixelCrop.width,
    pixelCrop.height,
    0,
    0,
    outputSize,
    outputSize
  );

  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (!blob) { reject(new Error("Canvas toBlob failed")); return; }
        resolve({ blob, dataUrl: canvas.toDataURL("image/png") });
      },
      "image/png",
      1.0
    );
  });
}

export function ImageCropper({ imageSrc, onConfirm, onCancel }: ImageCropperProps) {
  const [crop,       setCrop]       = useState<Point>({ x: 0, y: 0 });
  const [zoom,       setZoom]       = useState(1);
  const [croppedArea, setCroppedArea] = useState<Area | null>(null);
  const [loading,    setLoading]    = useState(false);
  const [cropError,  setCropError]  = useState<string | null>(null);

  const onCropComplete = useCallback((_: Area, croppedAreaPixels: Area) => {
    setCroppedArea(croppedAreaPixels);
  }, []);

  const handleConfirm = useCallback(async () => {
    if (!croppedArea) return;
    setLoading(true);
    setCropError(null);
    try {
      const { blob, dataUrl } = await getCroppedImg(imageSrc, croppedArea);
      onConfirm(blob, dataUrl);
    } catch (err) {
      console.error("Crop failed:", err);
      setCropError("Could not confirm this crop. Please try another image or restart the flow.");
    } finally {
      setLoading(false);
    }
  }, [croppedArea, imageSrc, onConfirm]);

  return (
    <div className="flex flex-col gap-4">
      <div className="relative w-full h-80 rounded-xl overflow-hidden bg-black">
        <Cropper
          image={imageSrc}
          crop={crop}
          zoom={zoom}
          aspect={1}
          onCropChange={setCrop}
          onZoomChange={setZoom}
          onCropComplete={onCropComplete}
          cropShape="rect"
          showGrid
          style={{
            containerStyle: { background: "#0f172a" },
            cropAreaStyle:  { border: "2px solid #6366f1" },
          }}
        />
      </div>

      {/* Zoom slider */}
      <div className="flex items-center gap-3">
        <svg className="w-4 h-4 text-slate-400 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
        </svg>
        <input
          type="range"
          min={1}
          max={3}
          step={0.05}
          value={zoom}
          onChange={(e) => setZoom(Number(e.target.value))}
          className="flex-1 accent-indigo-500 h-2"
        />
        <span className="text-xs text-slate-400 w-8 text-right">{zoom.toFixed(1)}×</span>
      </div>

      <p className="text-xs text-slate-500 text-center">
        Image will be cropped to 1:1 and resized to 1024 × 1024 px
      </p>
      {cropError && (
        <p className="text-xs text-red-400 text-center">{cropError}</p>
      )}

      {/* Actions */}
      <div className="flex gap-3">
        <button
          onClick={onCancel}
          disabled={loading}
          className="flex-1 px-4 py-2.5 rounded-xl border border-slate-600 text-slate-300
                     hover:bg-slate-700 transition-colors text-sm font-medium disabled:opacity-50"
        >
          Cancel
        </button>
        <button
          onClick={handleConfirm}
          disabled={loading || !croppedArea}
          className="flex-1 px-4 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500
                     text-white text-sm font-medium transition-colors disabled:opacity-50
                     flex items-center justify-center gap-2"
        >
          {loading ? (
            <>
              <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                <path className="opacity-75" fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
              </svg>
              Processing…
            </>
          ) : (
            "Confirm Crop"
          )}
        </button>
      </div>
    </div>
  );
}
