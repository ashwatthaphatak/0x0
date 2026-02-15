"""
modal_backend/app.py
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
import time
import uuid

import modal
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Modal image ───────────────────────────────────────────────────────────────

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.2.2",
        "torchvision==0.17.2",
        "torchaudio==2.2.2",
        index_url="https://download.pytorch.org/whl/cu121",  # CUDA only
    )
    .pip_install(
    "opencv-python-headless>=4.8.0",
    "scikit-image>=0.22.0",
    "Pillow>=10.0.0",
    "numpy<2",           # ← pin to NumPy 1.x, compatible with torch 2.2.2
    "scipy>=1.11.0",
    "lpips>=0.1.4",
    "requests>=2.31.0",  # ← ADD THIS
    "fastapi[standard]>=0.111.0",
    "python-multipart>=0.0.9",
    "anyio>=4.0",
    )
    .add_local_file("python_engine/defense_core.py", "/app/defense_core.py")
)

app = modal.App("deepfake-defense", image=image)

# ── In-memory job store ───────────────────────────────────────────────────────

jobs: dict[str, dict] = {}

# ── GPU function ──────────────────────────────────────────────────────────────

@app.function(
    gpu="T4",
    timeout=600,
    retries=1,
    memory=4096,
)
def run_protection_task(job_id: str, image_b64: str, epsilon: float) -> dict:
    import sys
    import tempfile
    sys.path.insert(0, "/app")

    import torch
    import torchvision.transforms as transforms
    from PIL import Image as PILImage
    from defense_core import (
        DualAttentionModule,
        DeepfakeDefenseFramework,
        compute_protection_score,
        save_image,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    image_bytes = base64.b64decode(image_b64)

    def _update(progress: int, message: str = "") -> None:
        jobs[job_id] = {**jobs.get(job_id, {}), "progress": progress, "message": message}

    _update(5,  "Loading models...")
    attention_module  = DualAttentionModule(device)
    defense_framework = DeepfakeDefenseFramework(epsilon=epsilon, device=device).to(device)

    _update(25, "Loading image...")
    img    = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
    tf     = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
    tensor = tf(img).unsqueeze(0).to(device)

    _update(45, "Generating attention map...")
    attention_map = attention_module.get_attention_map(tensor)

    _update(65, "Injecting perturbation...")
    vaccinated, _ = defense_framework.vaccinate_image(tensor, attention_map)

    _update(85, "Computing score & saving...")
    score = compute_protection_score(tensor, vaccinated)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        save_image(vaccinated, tmp.name)
        with open(tmp.name, "rb") as f:
            result_b64 = base64.b64encode(f.read()).decode()
        os.unlink(tmp.name)

    return {"score": score, "image_b64": result_b64}


# ── FastAPI web endpoints ─────────────────────────────────────────────────────

@app.function(min_containers=1)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web() -> FastAPI:
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
        if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise HTTPException(400, f"Unsupported type: {image.content_type}")
        if not (0.001 <= epsilon <= 0.2):
            raise HTTPException(400, "epsilon must be in [0.001, 0.2]")

        image_bytes = await image.read()
        if len(image_bytes) > 50 * 1024 * 1024:
            raise HTTPException(413, "Image too large (max 50 MB)")

        job_id    = str(uuid.uuid4())
        image_b64 = base64.b64encode(image_bytes).decode()
        jobs[job_id] = {"status": "pending", "progress": 0, "message": "Queued"}

        async def _run():
            jobs[job_id]["status"] = "running"
            try:
                result = await run_protection_task.remote.aio(job_id, image_b64, epsilon)
                jobs[job_id].update({
                    "status":     "complete",
                    "progress":   100,
                    "result_url": f"data:image/png;base64,{result['image_b64']}",
                    "score":      result["score"],
                    "message":    "Done",
                })
            except Exception as exc:
                jobs[job_id].update({"status": "failed", "message": str(exc)})

        asyncio.create_task(_run())
        return JSONResponse({"job_id": job_id, "status": "pending"})

    @api.get("/status/{job_id}")
    async def status(job_id: str):
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, f"Job {job_id} not found")
        return JSONResponse({
            "job_id":     job_id,
            "status":     job.get("status",   "pending"),
            "progress":   job.get("progress", 0),
            "result_url": job.get("result_url"),
            "score":      job.get("score"),
            "message":    job.get("message",  ""),
        })

    return api


@app.local_entrypoint()
def main():
    print("App ready.")
    print("  Serve:  MODAL_CONFIG_PATH=../.modal.toml modal serve app.py")
    print("  Deploy: MODAL_CONFIG_PATH=../.modal.toml modal deploy app.py")
