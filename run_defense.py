#!/usr/bin/env python3
import argparse

from deepfake_defense.stargan import CELEBA_DOMAIN_ORDER, RAFD_DEFAULT_EXPRESSIONS, attribute_catalog
from deepfake_defense.utils import configure_runtime, get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Deepfake defense runner")
    parser.add_argument("--mode", choices=["demo", "gradio", "batch"], default="demo")
    parser.add_argument("--image", type=str, default="sample_images/sample_0.jpg")
    parser.add_argument("--input-dir", type=str, default="sample_images", help="Folder for batch mode")
    parser.add_argument("--output-dir", type=str, default="batch_results", help="Batch output folder")
    parser.add_argument("--domain", choices=["celeba", "rafd"], default="celeba")
    parser.add_argument(
        "--expression-checkpoint",
        type=str,
        default=None,
        help="Path to StarGAN RaFD generator checkpoint (required for --domain rafd)",
    )
    parser.add_argument(
        "--expression-labels",
        type=str,
        default=None,
        help="Comma-separated expression labels in checkpoint order for RaFD.",
    )
    parser.add_argument(
        "--attribute",
        type=str,
        default="Blonde Hair",
        help="Attack vector for demo/batch. Use exact name, comma list, or 'all' (batch mode).",
    )
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share URL")
    parser.add_argument("--enable-lpips", action="store_true", help="Initialize LPIPS (optional)")
    parser.add_argument("--list-attributes", action="store_true", help="Print supported attack vectors and exit")
    return parser.parse_args()


def main():
    args = parse_args()
    expressions = None
    if args.expression_labels:
        expressions = [x.strip() for x in args.expression_labels.split(",") if x.strip()]
    elif args.domain == "rafd":
        expressions = list(RAFD_DEFAULT_EXPRESSIONS)

    if args.list_attributes:
        print("StarGAN checkpoint domains:")
        print(f"- CelebA: {CELEBA_DOMAIN_ORDER}")
        print(f"- RaFD: {tuple(expressions or RAFD_DEFAULT_EXPRESSIONS)}")
        print("Supported attack vectors:")
        for name in attribute_catalog(domain=args.domain, expressions=expressions):
            print(f"- {name}")
        return
    if args.domain == "rafd" and not args.expression_checkpoint:
        raise SystemExit("--expression-checkpoint is required when --domain rafd.")
    if args.domain == "rafd" and args.attribute == "Blonde Hair":
        args.attribute = "Happy"

    configure_runtime()
    set_seed(42)

    device = get_device()
    print(f"Using device: {device}")
    if device.type == "mps":
        print("Backend: Apple Metal (MPS)")
    elif device.type == "cpu":
        print("Backend: CPU (slower; functional fallback)")

    from deepfake_defense.batch import process_folder
    from deepfake_defense.gradio_app import build_gradio_app
    from deepfake_defense.pipeline import DefensePipeline

    pipeline = DefensePipeline(
        device=device,
        epsilon=args.epsilon,
        enable_lpips=args.enable_lpips,
        domain=args.domain,
        expression_checkpoint=args.expression_checkpoint,
        expression_labels=expressions,
    )

    if args.mode == "demo":
        metrics = pipeline.run_demo(
            image_path=args.image,
            target_attr=args.attribute,
            results_dir=args.results_dir,
        )
        print("Demo complete")
        print(f"Defense L2: {metrics['L2']:.4f}")
        return

    if args.mode == "batch":
        aggregate = process_folder(
            pipeline=pipeline,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            attribute=args.attribute,
            domain=args.domain,
            expressions=expressions,
        )
        print("Batch complete")
        print(f"Images total: {aggregate['images_total']}")
        print(f"Images ok: {aggregate['images_ok']}")
        print(f"Images error: {aggregate['images_error']}")
        print(f"Batch total ms: {aggregate['batch_total_ms']:.3f}")
        return

    app = build_gradio_app(pipeline)
    app.launch(share=args.share, debug=True)


if __name__ == "__main__":
    main()
