#!/usr/bin/env python3
import argparse

from deepfake_defense.gradio_app import build_gradio_app
from deepfake_defense.pipeline import DefensePipeline
from deepfake_defense.utils import get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Deepfake defense runner")
    parser.add_argument("--mode", choices=["demo", "gradio"], default="demo")
    parser.add_argument("--image", type=str, default="sample_images/sample_0.jpg")
    parser.add_argument("--attribute", type=str, default="Blonde Hair")
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share URL")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(42)

    device = get_device()
    print(f"Using device: {device}")
    if device.type != "cuda":
        print("Warning: CUDA not found. This project is intended for NVIDIA GPU + CUDA.")

    pipeline = DefensePipeline(device=device, epsilon=args.epsilon)

    if args.mode == "demo":
        metrics = pipeline.run_demo(
            image_path=args.image,
            target_attr=args.attribute,
            results_dir=args.results_dir,
        )
        print("Demo complete")
        print(f"Defense L2: {metrics['L2']:.4f}")
        return

    app = build_gradio_app(pipeline)
    app.launch(share=args.share, debug=True)


if __name__ == "__main__":
    main()
