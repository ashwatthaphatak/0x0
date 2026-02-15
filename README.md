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

### Prerequisites

| Tool | Version |
|------|---------|
| Node.js | ≥ 18 |
| Rust | stable (≥ 1.77) |
| Python | 3.11 |
| `tauri-cli` | 2.x |

### 1. Install JS dependencies

```bash
npm install
```

### 2. Build the Python sidecar

```bash
cd python_engine
pip install -r requirements.txt
chmod +x build_binary.sh
./build_binary.sh
cd ..
```

This places `defense-engine-<target-triple>` into `src-tauri/binaries/`.

### 3. Configure environment

```bash
cp .env.example .env.local
# Edit .env.local with your Modal URL
```

### 4. Run in development

```bash
npm run tauri dev
```

### 5. Build for distribution

```bash
npm run tauri build
```

Outputs: `.msi` (Windows), `.dmg` (macOS), `.deb`/`.AppImage` (Linux) in `src-tauri/target/release/bundle/`.

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
└── .env.example
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
