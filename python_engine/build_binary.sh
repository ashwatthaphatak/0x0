#!/usr/bin/env bash
# ============================================================
# build_binary.sh – Compile the Python engine into a sidecar
# binary for Tauri bundling.
#
# Run this script once per target OS before running:
#   npm run tauri build
#
# Output lands in: ../src-tauri/binaries/
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BINARIES_DIR="$ROOT_DIR/src-tauri/binaries"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN="python3"
fi

echo "🐍  Checking Python version…"
"$PYTHON_BIN" --version

echo "🧭  Using interpreter: $(command -v "$PYTHON_BIN")"

if [ "${SKIP_PIP_INSTALL:-0}" = "1" ]; then
    echo "⏭️  Skipping pip install steps (SKIP_PIP_INSTALL=1)"
else
    echo "📦  Installing pip dependencies…"
    "$PYTHON_BIN" -m pip install --upgrade pip
    "$PYTHON_BIN" -m pip install -r "$SCRIPT_DIR/requirements.txt"
    "$PYTHON_BIN" -m pip install pyinstaller==6.3.0
fi

echo "🔨  Running PyInstaller…"
cd "$SCRIPT_DIR"

"$PYTHON_BIN" - "$BINARIES_DIR" "$SCRIPT_DIR" <<'PY'
import sys
from PyInstaller.__main__ import run

distpath = sys.argv[1]
script_dir = sys.argv[2]

# Large ML dependency graphs can exceed the default recursion depth during
# module analysis.
sys.setrecursionlimit(max(10000, sys.getrecursionlimit() * 10))

run(
    [
        "--onefile",
        "--clean",
        "--name",
        "defense-engine",
        "--distpath",
        distpath,
        "--workpath",
        f"{script_dir}/build",
        "--specpath",
        script_dir,
        "--hidden-import",
        "cv2",
        "--hidden-import",
        "skimage",
        "--hidden-import",
        "skimage.metrics",
        "--hidden-import",
        "PIL",
        "--hidden-import",
        "torchvision.models",
        "--exclude-module",
        "pkg_resources",
        "--exclude-module",
        "setuptools",
        "--exclude-module",
        "jaraco",
        "--exclude-module",
        "more_itertools",
        "--exclude-module",
        "autocommand",
        "--collect-all",
        "torchvision",
        "--collect-all",
        "timm",
        "--collect-all",
        "lpips",
        "main.py",
    ]
)
PY

# ── Rename the binary so Tauri can find it by triple ───────────────────────
# Tauri sidecar naming convention: <name>-<target-triple>[.exe]
# Detect the current Rust target triple.
# `rustup show active-toolchain` includes channel prefixes (e.g., stable-...),
# so use `rustc -vV` host instead.
if command -v rustc >/dev/null 2>&1; then
    TRIPLE=$(rustc -vV | awk '/^host: / { print $2 }' || true)
fi

if [ -z "${TRIPLE:-}" ]; then
    # Fallback detection
    OS=$(uname -s)
    ARCH=$(uname -m)
    case "$OS-$ARCH" in
        Linux-x86_64)   TRIPLE="x86_64-unknown-linux-gnu"  ;;
        Linux-aarch64)  TRIPLE="aarch64-unknown-linux-gnu" ;;
        Darwin-x86_64)  TRIPLE="x86_64-apple-darwin"       ;;
        Darwin-arm64)   TRIPLE="aarch64-apple-darwin"      ;;
        MINGW*-x86_64)  TRIPLE="x86_64-pc-windows-msvc"   ;;
        *)               TRIPLE="unknown-unknown-unknown"   ;;
    esac
fi

# Backward compatibility if a channel prefix somehow slips in.
TRIPLE="${TRIPLE#stable-}"
TRIPLE="${TRIPLE#beta-}"
TRIPLE="${TRIPLE#nightly-}"

EXT=""
if [[ "$TRIPLE" == *"windows"* ]]; then
    EXT=".exe"
fi

SRC="$BINARIES_DIR/defense-engine${EXT}"
DST="$BINARIES_DIR/defense-engine-${TRIPLE}${EXT}"

if [ -f "$SRC" ] && [ "$SRC" != "$DST" ]; then
    mv "$SRC" "$DST"
fi

echo ""
echo "✅  Binary ready: $DST"
echo "    Next: cd .. && npm run tauri build"
