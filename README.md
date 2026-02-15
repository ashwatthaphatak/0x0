# Vaxel

Protect your identity.

Vaxel is a desktop app that proactively sanitizes images to make deepfake manipulation less effective while preserving visual quality.

## Hackathon Submission

### Problem

People are often required to upload headshots into public or semi-public systems, and those images can be reused for identity theft.

High-risk upload scenarios include:

- Online verification flows
- Photo submissions for IDs (government IDs, school IDs, work IDs, and similar systems)

Most tools focus on detecting fake media after the damage is done. Users need prevention before sharing images.

### Solution

Vaxel applies texture-focused perturbations to an image before publication. These perturbations are visually subtle for humans but disruptive for deepfake pipelines.

This specifically helps defend against widely used face-attribute manipulation filters that can be abused for identity theft, including:

- Hair color modification
- Apparent age modification
- Identity misuse workflows based on manipulated headshots

### Why This Matters

- Prevention-first instead of detection-only
- Local, privacy-preserving workflow
- Simple UX for non-technical users
- First step toward broader fraud prevention

Vaxel is designed as a first practical step toward stopping fraud driven by manipulated identity photos.

## What Vaxel Does

- Upload and crop an image
- Apply one-click sanitization
- Show a `Protection Score` percentage (higher is better)
- Run optional local deepfake attack simulation for comparison

## Core Features

- Desktop app built with Tauri (macOS/Windows/Linux packaging)
- Local Python defense engine (no cloud required for core flow)
- Three protection levels (`Low`, `Medium`, `High`)
- Side-by-side result preview and export
- Optional webcam capture in upload flow

## Tech Stack

- Frontend: Next.js 14, React, Tailwind CSS
- Desktop Shell: Tauri 2 (Rust)
- ML Engine: Python (PyTorch, OpenCV, scikit-image)
- Optional Cloud Path: Modal backend (`modal_backend/`)

## Architecture

```text
┌─────────────────────────────────────────────────────┐
│  Tauri 2 Desktop Shell (Rust)                       │
│  ┌─────────────────────┐  ┌───────────────────────┐ │
│  │  Next.js 14 (UI)    │  │  Python Engine        │ │
│  │  Tailwind CSS       │◄─►  (dev: main.py)       │ │
│  │  react-easy-crop    │  │  (prod: sidecar bin)  │ │
│  └─────────────────────┘  └───────────────────────┘ │
│           │                                           │
│           └──────────────► Optional Modal Cloud API  │
└─────────────────────────────────────────────────────┘
```

## Demo Links

- Demo Video: `<add-link>`
- Presentation Slides: `<add-link>`
- Submission Page: `<add-link>`

## Quick Start (Local Development)

### 1. Prerequisites

| Tool    | Version                    |
| ------- | -------------------------- |
| Node.js | 18+ (20 LTS recommended)   |
| npm     | bundled with Node          |
| Rust    | stable (1.77+ recommended) |
| Python  | 3.11 recommended           |

Install Tauri platform prerequisites first:
https://tauri.app/start/prerequisites/

On macOS:

```bash
xcode-select --install
```

### 2. Install dependencies

```bash
git clone <repo-url>
cd 0x0
npm install
```

### 3. Create Python environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r python_engine/requirements.txt
```

### 4. Configure local engine interpreter

```bash
export DEEPFAKE_DEFENSE_PYTHON="$(pwd)/.venv/bin/python"
```

### 5. Run app

```bash
npm run tauri dev
```

## Build Desktop App (Submission/Distribution)

### 1. Build Python sidecar binary

```bash
cd python_engine
chmod +x build_binary.sh
PYTHON_BIN="$DEEPFAKE_DEFENSE_PYTHON" ./build_binary.sh
cd ..
```

### 2. Build installers

```bash
npm run tauri build
```

Artifacts are generated in:
`src-tauri/target/release/bundle/`

## Optional Cloud Mode

If you want cloud processing, configure:

```bash
cat > .env.local <<'EOF_ENV'
NEXT_PUBLIC_MODAL_BASE_URL=https://your-modal-endpoint.modal.run
EOF_ENV
```

Deploy backend from `modal_backend/` if needed.

## Judging-Relevant Highlights

- **Innovation:** prevention-focused image immunization workflow
- **Impact:** reduces identity-theft risk for publicly uploaded headshots
- **Technical Depth:** Rust desktop shell + Next.js UI + Python ML engine
- **Usability:** one primary action flow for non-technical users

## Current Scope / Limitations

- Core production path is local-first
- Cloud mode is optional and depends on backend deployment
- Quality and robustness vary by attack/model family

## Repository Layout

```text
.
├── src/                         # Next.js UI
├── src-tauri/                   # Rust/Tauri desktop shell
├── python_engine/               # Local defense engine (Python)
├── modal_backend/               # Optional cloud backend
├── deepfake_defense/            # Core defense/attack pipeline modules
├── package.json
└── README.md
```

## Team

**Project Name:** `Vaxel`

**Team Members**

- Ashwattha Phatak
- Anish Mulay
- Akshay Mulay

**Contact**

- aaphatak@ncsu.edu
- amulay2@ncsu.edu
- adongar@ncsu.edu
