"""
modal_backend/app.py
────────────────────────────────────────────────────────────────────────────
Modal.com FastAPI app that mirrors the local Python engine over HTTP.

Endpoints:
  POST /ingest       → accepts multipart image + epsilon, enqueues job, returns job_id
  GET  /status/{id}  → returns job progress / result URL
  GET  /health       → liveness check

Deploy:
  modal deploy modal_backend/app.py

Environment variables required in Modal secrets:
  RESULT_BUCKET  – (optional) object storage bucket name for result URLs
"""

from __future__ import annotations

import io
import json
import os
import time
import uuid
from typing import Optional

import modal

# ── Modal app & image ─────────────────────────────────────────────────────────

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.2.2",
        "torchvision==0.17.2",
        "torchaudio==2.2.2",
        index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install(
        "opencv-python-headless>=4.8.0",
        "scikit-image>=0.22.0",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "fastapi[standard]>=0.111.0",
        "python-multipart>=0.0.9",
        "anyio>=4.0",
    )
    .copy_local_file("python_engine/defense_core.py", "/app/defense_core.py")
)

app = modal.App("deepfake-defense", image=image)

# ── In-memory job store (replace with Modal Dict for persistence) ─────────────

jobs: dict[str, dict] = {}   # job_id → {status, progress, result_url, score, message}

# ── GPU / CPU function ───────────────────────────────────────────────────────

@app.function(
    # Use GPU if available; falls back to CPU silently
    gpu=modal.gpu.T4(count=1),
    timeout=600,
    retries=1,
    memory=4096,
)
def run_protection_task(job_id: str, image_bytes: bytes, epsilon: float) -> dict:
    """
    Runs the full TFP pipeline on the given image bytes.
    Updates the job store and returns the result dict.
    """
    import sys
    sys.path.insert(0, "/app")

    import torch
    from defense_core import (
        DualAttentionModule,
        DeepfakeDefenseFramework,
        save_image,
        compute_protection_score,
    )
    from PIL import Image
    import torchvision.transforms as transforms
    import tempfile

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _update(progress: int, message: str = "") -> None:
        jobs[job_id] = {**jobs.get(job_id, {}), "progress": progress, "message": message}

    _update(5, "Loading model…")
    attention_module  = DualAttentionModule(device)
    defense_framework = DeepfakeDefenseFramework(epsilon=epsilon, device=device)

    _update(25, "Loading image…")
    img  = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tf   = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor()])
    tensor = tf(img).unsqueeze(0).to(device)

    _update(45, "Generating attention map…")
    attention_map = attention_module.get_attention_map(tensor)

    _update(65, "Injecting perturbation…")
    vaccinated, _ = defense_framework.vaccinate_image(tensor, attention_map)

    _update(85, "Computing score & saving…")
    score = compute_protection_score(tensor, vaccinated)

    # Save to temp file and return bytes
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf_file:
        save_image(vaccinated, tf_file.name)
        with open(tf_file.name, "rb") as f:
            result_bytes = f.read()
        os.unlink(tf_file.name)

    return {"score": score, "image_bytes": result_bytes}


# ── FastAPI web endpoints ─────────────────────────────────────────────────────

@app.function(
    keep_warm=1,       # keep one container warm to reduce cold starts
    allow_concurrent_inputs=20,
)
@modal.asgi_app()
def web() -> "fastapi.FastAPI":
    import base64
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, Response

    api = FastAPI(title="DeepFake Defense API", version="1.0.0")

    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @api.get("/health")
    async def health():
        return {"status": "ok", "timestamp": time.time()}

    @api.post("/ingest")
    async def ingest(
        image:   UploadFile = File(...),
        epsilon: float      = Form(default=0.05),
    ):
        # Validate
        if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise HTTPException(400, f"Unsupported content type: {image.content_type}")

        if not (0.001 <= epsilon <= 0.2):
            raise HTTPException(400, f"epsilon must be in [0.001, 0.2], got {epsilon}")

        image_bytes = await image.read()
        if len(image_bytes) > 50 * 1024 * 1024:
            raise HTTPException(413, "Image too large (max 50 MB)")

        job_id = str(uuid.uuid4())
        jobs[job_id] = {"status": "pending", "progress": 0, "message": "Queued"}

        # Spawn the GPU task (non-blocking)
        async def _run():
            jobs[job_id]["status"] = "running"
            try:
                result = await run_protection_task.remote.aio(job_id, image_bytes, epsilon)
                # Encode result image as base64 data URL (simple – for production use S3/R2)
                b64 = base64.b64encode(result["image_bytes"]).decode()
                jobs[job_id].update({
                    "status":     "complete",
                    "progress":   100,
                    "result_url": f"data:image/png;base64,{b64}",
                    "score":      result["score"],
                    "message":    "Done",
                })
            except Exception as exc:  # noqa: BLE001
                jobs[job_id].update({
                    "status":  "failed",
                    "message": str(exc),
                })

        import asyncio
        asyncio.create_task(_run())

        return JSONResponse({"job_id": job_id, "status": "pending"})

    @api.get("/status/{job_id}")
    async def status(job_id: str):
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, f"Job {job_id} not found")
        return JSONResponse({
            "job_id":     job_id,
            "status":     job.get("status", "pending"),
            "progress":   job.get("progress", 0),
            "result_url": job.get("result_url"),
            "score":      job.get("score"),
            "message":    job.get("message", ""),
        })

    return api
