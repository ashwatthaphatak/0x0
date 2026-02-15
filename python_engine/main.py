#!/usr/bin/env python3
"""
DeepFake Defense Engine – CLI Entry Point
==========================================
Compiled to a standalone binary via PyInstaller and bundled inside Tauri as a sidecar.

Protocol (stdout lines parsed by Tauri/Rust):
  STATUS: <message>
  PROGRESS: <0-100>
  SUCCESS: <json payload>
  ERROR: <message>

Usage:
  defense-engine --input /path/to/image.png \
                 --output /path/to/result.png \
                 --level 0.05 \
                 [--size 1024] \
                 [--verify]
"""

import argparse
import sys
import os
import json
import traceback


def _log(msg: str) -> None:
    """Flush a line to stdout for Tauri to capture."""
    print(msg, flush=True)


def _status(msg: str) -> None:
    _log(f"STATUS: {msg}")


def _progress(pct: int) -> None:
    _log(f"PROGRESS: {pct}")


def _success_payload(payload: dict) -> None:
    _log(f"SUCCESS: {json.dumps(payload)}")


def _error(msg: str) -> None:
    _log(f"ERROR: {msg}")
    sys.exit(1)


def _success_protection(path: str, score: float) -> None:
    _success_payload({"path": path, "score": score})


def _check_dependencies(mode: str) -> None:
    """Fail early with a clear message if a dependency is missing."""
    required = ["torch", "torchvision", "cv2", "PIL", "numpy", "skimage"]
    if mode == "attack":
        required.append("requests")

    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        _error(f"Missing packages: {', '.join(missing)}. "
               "Re-bundle the sidecar with: cd python_engine && ./build_binary.sh")


def run_protection(input_path: str, output_path: str,
                   epsilon: float, size: int) -> float:
    """
    Full pipeline: load → attention → vaccinate → save.
    Returns the protection score (0–100).
    """
    import torch
    from defense_core import (
        DualAttentionModule, DeepfakeDefenseFramework,
        load_image, save_image, compute_protection_score
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _status(f"Using device: {device}")
    _progress(5)

    _status("Loading model (ResNet-50 attention backbone)…")
    attention_module   = DualAttentionModule(device)
    defense_framework  = DeepfakeDefenseFramework(epsilon=epsilon, device=device)
    _progress(20)

    _status(f"Loading image from {input_path}…")
    if not os.path.isfile(input_path):
        _error(f"Input file not found: {input_path}")
    original_tensor = load_image(input_path, size=size, device=device)
    _progress(35)

    _status("Generating attention map…")
    attention_map = attention_module.get_attention_map(original_tensor)
    _progress(55)

    _status("Injecting adversarial perturbation…")
    vaccinated, _perturbation = defense_framework.vaccinate_image(
        original_tensor, attention_map
    )
    _progress(80)

    _status("Computing protection score…")
    score = compute_protection_score(original_tensor, vaccinated)
    _progress(90)

    _status(f"Saving result to {output_path}…")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    save_image(vaccinated, output_path)
    _progress(100)

    return score


def run_verify(input_path: str, protected_path: str) -> dict:
    """
    Quick verification: compare original vs. protected image metrics.
    """
    import torch
    from defense_core import load_image, calculate_metrics

    device = torch.device("cpu")
    orig   = load_image(input_path,    size=256, device=device)
    prot   = load_image(protected_path, size=256, device=device)
    return calculate_metrics(orig, prot)


def run_attack(
    input_path: str,
    protected_path: str,
    attack_type: str,
    output_dir: str,
    attack_size: int,
) -> dict:
    from stargan_attack import run_attack_comparison

    def _update(pct: int, message: str) -> None:
        _status(message)
        _progress(pct)

    return run_attack_comparison(
        original_path=input_path,
        sanitized_path=protected_path,
        attack_type=attack_type,
        output_dir=output_dir,
        attack_size=attack_size,
        progress=_update,
    )


def validate_input_image(path: str) -> None:
    if not os.path.isfile(path):
        _error(f"Input file does not exist: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
        _error(f"Unsupported file type: {ext}. Accepted: jpg, png, webp")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DeepFake Defense Engine – vaccinate images against deepfakes"
    )
    parser.add_argument(
        "--mode",
        choices=["protect", "attack"],
        default="protect",
        help="Execution mode. protect=sanitize image, attack=run StarGAN comparison.",
    )
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--output", help="Path to write protected image (protect mode)")
    parser.add_argument("--protected", help="Path to sanitized image (attack mode)")
    parser.add_argument(
        "--attack-type",
        default="blonde_hair",
        help="Attack type: blonde_hair, old_age, male (attack mode)",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to write attack outputs (attack mode)",
    )
    parser.add_argument(
        "--attack-size",
        type=int,
        default=256,
        help="StarGAN attack resolution (default: 256)",
    )
    parser.add_argument("--level", type=float, default=0.05,
                        help="Perturbation strength epsilon (default: 0.05)")
    parser.add_argument("--size", type=int, default=1024,
                        help="Processing resolution (default: 1024)")
    parser.add_argument("--verify", action="store_true",
                        help="Also run post-hoc verification pass")
    args = parser.parse_args()

    # ── Input validation ──────────────────────────────────────────────────────
    validate_input_image(args.input)

    if args.mode == "protect" and not args.output:
        _error("--output is required in protect mode")

    if args.mode == "attack":
        if not args.protected:
            _error("--protected is required in attack mode")
        validate_input_image(args.protected)

    if args.mode == "protect" and not (0.001 <= args.level <= 0.2):
        _error(f"Epsilon out of range: {args.level}. Must be between 0.001 and 0.2")

    # ── Dependency check ──────────────────────────────────────────────────────
    _status("Checking dependencies…")
    _check_dependencies(args.mode)
    _progress(2)

    # ── Main pipeline ─────────────────────────────────────────────────────────
    try:
        if args.mode == "protect":
            score = run_protection(
                input_path=args.input,
                output_path=args.output,
                epsilon=args.level,
                size=args.size,
            )

            if args.verify:
                _status("Running verification pass…")
                metrics = run_verify(args.input, args.output)
                _log(f"METRICS: {json.dumps(metrics)}")

            _success_protection(args.output, score)
        else:
            default_output_dir = os.path.join(
                os.path.dirname(os.path.abspath(args.protected)),
                "deepfake-tests",
            )
            attack_result = run_attack(
                input_path=args.input,
                protected_path=args.protected,
                attack_type=args.attack_type,
                output_dir=args.output_dir or default_output_dir,
                attack_size=args.attack_size,
            )
            _success_payload(attack_result)

    except MemoryError:
        _error("Out of memory. Try reducing --size or switching to Cloud mode.")

    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        _error(f"Unexpected error: {exc}\n{tb}")


if __name__ == "__main__":
    main()
