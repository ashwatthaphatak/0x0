# DeepFake Defense Desktop Platform

Proactively protect images against deepfake manipulation using Texture Feature Perturbation (TFP), based on Zhang et al. (2025).

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

## Full Setup Tutorial (First-Time Clone)

These steps are written for a brand new machine/project clone.

### 1. Install prerequisites

| Tool | Required version | Why |
|---|---|---|
| Node.js | 18+ (20 LTS recommended) | Next.js + Tauri frontend |
| npm | Comes with Node.js | Package manager used by this repo |
| Rust | stable (1.77+ recommended) | Tauri desktop shell |
| Python | 3.11 recommended | Local defense engine |

Install Tauri OS prerequisites for your platform first:
https://tauri.app/start/prerequisites/

If you are on macOS, also install Xcode Command Line Tools:

```bash
xcode-select --install
```

### 2. Clone the repo and install JavaScript dependencies

```bash
git clone <repo-url>
cd 0x0
npm install
```

### 3. Create a Python environment for the local engine

Use a dedicated virtual environment so dependencies are isolated.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r python_engine/requirements.txt
```

### 4. Configure required environment variable

The local engine in dev mode uses `python_engine/main.py` and expects a Python interpreter that already has the ML dependencies installed.

```bash
export DEEPFAKE_DEFENSE_PYTHON="$(pwd)/.venv/bin/python"
```

Optional: persist it for future terminal sessions (`zsh`):

```bash
echo 'export DEEPFAKE_DEFENSE_PYTHON="/ABSOLUTE/PATH/TO/0x0/.venv/bin/python"' >> ~/.zshrc
source ~/.zshrc
```

### 5. Optional cloud mode configuration

If you want Cloud mode in the UI, set a Modal backend URL in `.env.local`.

```bash
cat > .env.local <<'EOF_ENV'
NEXT_PUBLIC_MODAL_BASE_URL=https://your-modal-endpoint.modal.run
EOF_ENV
```

If you only need local mode, skip this step.

### 6. Optional but recommended: preload StarGAN weights

This avoids first-run delays for local deepfake attack testing.

```bash
"$DEEPFAKE_DEFENSE_PYTHON" python_engine/download_stargan_weights.py
```

If you already have a weight file:

```bash
"$DEEPFAKE_DEFENSE_PYTHON" python_engine/download_stargan_weights.py --from-file /absolute/path/to/celeba-128x128-5attrs.zip
```

### 7. Run the desktop app (development)

From the repo root:

```bash
npm run tauri dev
```

This runs:
- Next.js frontend (`npm run dev`)
- Tauri desktop shell
- Python local engine via `DEEPFAKE_DEFENSE_PYTHON`

### 8. First run checks on macOS (camera feature)

This app includes webcam capture in the upload flow.

1. Launch with `npm run tauri dev`.
2. Click `Use Camera Instead`.
3. Allow the macOS camera prompt.
4. If previously denied, enable camera access in `System Settings > Privacy & Security > Camera`.

### 9. Verify local engine readiness

Run this command in the same terminal/session:

```bash
"$DEEPFAKE_DEFENSE_PYTHON" -c "import torch,torchvision,cv2,PIL,numpy,skimage; print('python deps ok')"
```

Expected output:

```text
python deps ok
```

## Optional: Deploy the Modal Cloud Backend

Use this only if you want Cloud mode.

### 1. Install Modal CLI and authenticate

```bash
python3 -m pip install --upgrade modal
modal setup
```

### 2. Install backend Python dependencies (recommended in a dedicated env)

```bash
python3 -m venv .venv-modal
source .venv-modal/bin/activate
python -m pip install --upgrade pip
python -m pip install -r modal_backend/requirements.txt
```

### 3. Deploy

```bash
modal deploy modal_backend/app.py
```

Copy the printed web endpoint and place it in `.env.local` as `NEXT_PUBLIC_MODAL_BASE_URL`.

## Build a Desktop Bundle (Installer/App)

For packaged builds, create the Python sidecar binary first, then build Tauri.

### 1. Build sidecar binary

```bash
cd python_engine
chmod +x build_binary.sh
PYTHON_BIN="$DEEPFAKE_DEFENSE_PYTHON" ./build_binary.sh
cd ..
```

This writes a platform-specific file into `src-tauri/binaries/` named like:
- `defense-engine-aarch64-apple-darwin`
- `defense-engine-x86_64-pc-windows-msvc.exe`

### 2. Build installers

```bash
npm run tauri build
```

Artifacts are placed under `src-tauri/target/release/bundle/`.

## Troubleshooting

### Error: "Local Python engine is not ready"

Cause: `DEEPFAKE_DEFENSE_PYTHON` is unset or points to an interpreter without required packages.

Fix:

```bash
export DEEPFAKE_DEFENSE_PYTHON="$(pwd)/.venv/bin/python"
"$DEEPFAKE_DEFENSE_PYTHON" -c "import torch,torchvision,cv2,PIL,numpy,skimage; print('ok')"
```

### Error: camera does not open in desktop app

Cause: macOS permission denied previously.

Fix: enable camera permission for the app in `System Settings > Privacy & Security > Camera`, then relaunch the app.

### Error: sidecar not found during packaged run

Cause: sidecar binary was not built before `tauri build`.

Fix:

```bash
cd python_engine
PYTHON_BIN="$DEEPFAKE_DEFENSE_PYTHON" ./build_binary.sh
cd ..
npm run tauri build
```

## Project Structure (high level)

```text
.
├── src/                         # Next.js UI
├── src-tauri/                   # Rust/Tauri desktop shell
├── python_engine/               # Local defense engine (Python)
├── modal_backend/               # Optional cloud backend
├── next.config.js               # output: "export" for Tauri
├── package.json
└── README.md
```
