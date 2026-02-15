# 🛡️ DeepFake Defense Desktop Platform

Proactively **vaccinate** images against deepfake manipulation using **Texture Feature Perturbation (TFP)** — based on Zhang et al., 2025.

## Team Members

- Ashwattha Phatak
- Akshay Dongare
- Anish Mulay

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Tauri 2.0 Desktop Shell (Rust)                     │
│  ┌─────────────────────┐  ┌───────────────────────┐ │
│  │  Next.js 14 (SPA)   │  │  Python Sidecar       │ │
│  │  Tailwind CSS       │◄─►  (PyInstaller binary) │ │
│  │  react-easy-crop    │  │  - GradCAM attention  │ │
│  └─────────────────────┘  │  - TextureExtractor   │ │
│           │                │  - PerturbationGen    │ │
│           │ (cloud mode)   └───────────────────────┘ │
│           ▼                                           │
│  Modal.com FastAPI (T4 GPU)                          │
└─────────────────────────────────────────────────────┘
```

## Quick Start

### Full Local Setup (for a new collaborator)

### 1. Install prerequisites

| Tool | Version |
|------|---------|
| Node.js | >= 18 |
| Rust | stable (>= 1.77) |
| Python | 3.11 |
| Tauri CLI | 2.x (installed from this repo's `devDependencies`) |

Install the OS-level dependencies required by Tauri for your platform:
https://tauri.app/start/prerequisites/

### 2. Clone and install JavaScript dependencies

```bash
git clone <repo-url>
cd 0x0
npm install
```

### 3. Install Python engine dependencies

Pick the exact Python interpreter you want this app to use, then install engine
dependencies into that interpreter.

Example:

```bash
/absolute/path/to/python -m pip install --upgrade pip
/absolute/path/to/python -m pip install -r python_engine/requirements.txt
```

### 4. Set `DEEPFAKE_DEFENSE_PYTHON`

In dev mode, Tauri runs `python_engine/main.py` directly and checks
`DEEPFAKE_DEFENSE_PYTHON` first.

```bash
export DEEPFAKE_DEFENSE_PYTHON=/absolute/path/to/python
```

To persist this on `zsh`:

```bash
echo 'export DEEPFAKE_DEFENSE_PYTHON=/absolute/path/to/python' >> ~/.zshrc
source ~/.zshrc
```

### 5. Optional: configure Cloud mode

Cloud mode uses `NEXT_PUBLIC_MODAL_BASE_URL` from `.env.local`.

```bash
cat > .env.local <<'EOF'
NEXT_PUBLIC_MODAL_BASE_URL=https://your-modal-endpoint.modal.run
EOF
```

If you only need local mode, skip this step.

### 6. Optional but recommended: preload StarGAN weights

```bash
"$DEEPFAKE_DEFENSE_PYTHON" python_engine/download_stargan_weights.py
```

Or import from a file you already have:

```bash
"$DEEPFAKE_DEFENSE_PYTHON" python_engine/download_stargan_weights.py --from-file /absolute/path/to/celeba-128x128-5attrs.zip
# or
"$DEEPFAKE_DEFENSE_PYTHON" python_engine/download_stargan_weights.py --from-file /absolute/path/to/200000-G.ckpt
```

### 7. Start the app in development

This is the exact flow used in this repo:

```bash
export DEEPFAKE_DEFENSE_PYTHON=/Users/anish/miniforge3/bin/python
npm run tauri dev
```

For your friend, keep the command shape identical and only change the Python
path.

### 8. Troubleshooting local engine readiness

If the app says local engine is unavailable, verify the interpreter and deps:

```bash
"$DEEPFAKE_DEFENSE_PYTHON" -c "import torch,torchvision,cv2,PIL,numpy,skimage; print('python deps ok')"
```

### 9. Build distributable desktop bundles

```bash
npm run tauri build
```

Outputs: `.msi` (Windows), `.dmg` (macOS), `.deb`/`.AppImage` (Linux) in `src-tauri/target/release/bundle/`.

### 10. Optional: build the packaged Python sidecar binary

This is mostly needed for packaging workflows; regular `tauri dev` can run the
Python script directly.

```bash
cd python_engine
chmod +x build_binary.sh
PYTHON_BIN="$DEEPFAKE_DEFENSE_PYTHON" ./build_binary.sh
cd ..
```

This places `defense-engine-<target-triple>` into `src-tauri/binaries/`.

---

## Deploy Cloud Backend (Modal)

```bash
pip install modal
modal setup   # authenticate
modal deploy modal_backend/app.py
```

Copy the printed web URL into `.env.local` as `NEXT_PUBLIC_MODAL_BASE_URL`.

---

## Project Structure

```
/
├── src-tauri/
│   ├── binaries/              ← compiled Python sidecar lives here
│   ├── src/
│   │   ├── main.rs            ← Tauri entry point, lifecycle, state
│   │   └── commands.rs        ← IPC commands (invoke handlers)
│   ├── capabilities/
│   │   └── default.json       ← FS / shell / dialog permissions
│   ├── tauri.conf.json        ← App config, sidecar registration
│   └── Cargo.toml
│
├── src/
│   ├── components/
│   │   ├── compute-toggle.tsx           ← Local / Cloud switch
│   │   ├── image-dropzone.tsx           ← Drag & drop + native picker
│   │   ├── image-cropper.tsx            ← react-easy-crop 1:1 cropper
│   │   ├── progress-tracker.tsx         ← Unified progress bar
│   │   ├── protection-level-selector.tsx
│   │   └── result-viewer.tsx            ← Side-by-side slider + save
│   ├── hooks/
│   │   └── useProtection.ts             ← Orchestration hook
│   ├── lib/
│   │   ├── tauri-bridge.ts              ← All invoke() / event calls
│   │   └── modal-client.ts             ← Cloud upload / polling
│   ├── types/
│   │   └── index.ts
│   ├── page.tsx                         ← Main app UI (App Router)
│   ├── layout.tsx
│   └── globals.css
│
├── python_engine/
│   ├── main.py                          ← CLI entry point (sidecar)
│   ├── defense_core.py                  ← GradCAM + TFP algorithm
│   ├── requirements.txt
│   └── build_binary.sh
│
├── modal_backend/
│   ├── app.py                           ← Modal FastAPI deployment
│   └── requirements.txt
│
├── next.config.js                       ← output: 'export' (mandatory)
├── package.json
├── tsconfig.json
├── tailwind.config.js
└── .env.local (optional, for cloud mode)
```

---

## Sidecar IPC Protocol

The Python engine communicates with Tauri via `stdout`. Each line is one of:

| Prefix | Example | Meaning |
|--------|---------|---------|
| `STATUS: ` | `STATUS: Loading model…` | Human-readable status update |
| `PROGRESS: ` | `PROGRESS: 55` | Integer 0–100 |
| `SUCCESS: ` | `SUCCESS: {"path":"/tmp/…","score":92.4}` | JSON result |
| `ERROR: ` | `ERROR: Out of memory` | Fatal error (exit ≠ 0) |

---

## Defense Algorithm (TFP)

1. **Dual Attention Map** — ResNet-50 + GradCAM identifies texture-critical regions.
2. **Texture Feature Extraction** — Sobel gradients + bilateral filtering + shallow CNN.
3. **Perturbation Generation** — Attention-fused encoder-decoder produces a 3-channel perturbation.
4. **Vaccination** — `vaccinated = clamp(original + ε × perturbation, 0, 1)`.

Typical quality metrics (ε = 0.05, 1024 × 1024):
- PSNR ≥ 38 dB
- SSIM ≥ 0.97
- Imperceptible to the naked eye

---

## Edge Cases Handled

| Scenario | Behaviour |
|----------|-----------|
| Locked file | Native OS alert: "File is locked by another process" |
| Sidecar crash | Rust catches non-zero exit; frontend shows "Try Cloud mode" |
| Cloud timeout (>10 min) | Polling aborts with user-facing error |
| Wrong file type | Rejected at drop / file-picker with friendly message |
| File > 50 MB | Rejected with size error |
| Out of GPU/RAM | Python emits ERROR line; UI suggests Cloud mode |
| App close | Temp dir deleted; sidecar process killed |

okay, run this app to app
