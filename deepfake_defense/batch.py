import argparse
import csv
import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional

import numpy as np
import torch

try:
    from .stargan import (
        CELEBA_DOMAIN_ORDER,
        RAFD_DEFAULT_EXPRESSIONS,
        attribute_catalog,
        deepfake_attack,
        resolve_attribute_selection,
    )
    from .utils import calculate_metrics, configure_runtime, ensure_dirs, get_device, load_image, set_seed, to_pil
except ImportError:
    from deepfake_defense.stargan import (
        CELEBA_DOMAIN_ORDER,
        RAFD_DEFAULT_EXPRESSIONS,
        attribute_catalog,
        deepfake_attack,
        resolve_attribute_selection,
    )
    from deepfake_defense.utils import calculate_metrics, configure_runtime, ensure_dirs, get_device, load_image, set_seed, to_pil

if TYPE_CHECKING:
    try:
        from .pipeline import DefensePipeline
    except ImportError:
        from deepfake_defense.pipeline import DefensePipeline


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _elapsed(start: float, device: torch.device) -> float:
    _sync_device(device)
    return time.perf_counter() - start


def _write_timings(path: Path, timings: Dict[str, float]) -> None:
    lines = [f"{k}: {v * 1000:.3f} ms" for k, v in timings.items()]
    path.write_text("\n".join(lines) + "\n")


def _write_metrics(path: Path, metrics: Dict[str, float], defense_status: str, attribute: str) -> None:
    content = (
        f"attribute: {attribute}\n"
        f"defense_status: {defense_status}\n"
        f"PSNR: {metrics['PSNR']:.6f}\n"
        f"SSIM: {metrics['SSIM']:.6f}\n"
        f"L2: {metrics['L2']:.6f}\n"
    )
    path.write_text(content)


def _safe_dir_name(image_path: Path) -> str:
    return image_path.stem.replace(" ", "_")


def _safe_slug(text: str) -> str:
    slug = text.lower().replace("+", "plus").replace("|", "_").replace(" ", "_")
    return "".join(ch for ch in slug if ch.isalnum() or ch in {"_", "-"})


def _process_image_paths(
    pipeline: "DefensePipeline",
    image_paths: Iterable[Path],
    output_dir: str,
    attribute: str,
    domain: str,
    expressions: Optional[List[str]] = None,
) -> Dict[str, float]:
    ensure_dirs(output_dir)
    output_root = Path(output_dir)
    selected_attributes = resolve_attribute_selection(
        attribute,
        pipeline.attributes,
        domain=domain,
        expressions=expressions,
    )

    image_paths = list(image_paths)
    rows: List[Dict[str, str]] = []
    total_start = time.perf_counter()

    for image_path in image_paths:
        image_out = output_root / _safe_dir_name(image_path)
        ensure_dirs(str(image_out))
        for attribute_name in selected_attributes:
            image_total_start = time.perf_counter()
            target_attr_tensor = pipeline.attributes[attribute_name]
            attack_out = image_out / _safe_slug(attribute_name)
            ensure_dirs(str(attack_out))

            timings: Dict[str, float] = {}

            try:
                t0 = time.perf_counter()
                original = load_image(str(image_path), pipeline.transform, pipeline.device)
                timings["load_image"] = _elapsed(t0, pipeline.device)

                t0 = time.perf_counter()
                attention_map = pipeline.attention_module.get_attention_map(original)
                timings["attention_map"] = _elapsed(t0, pipeline.device)

                t0 = time.perf_counter()
                vaccinated, perturbation = pipeline.defense_framework.vaccinate_image(original, attention_map)
                timings["vaccinate_image"] = _elapsed(t0, pipeline.device)

                t0 = time.perf_counter()
                clean_deepfaked = deepfake_attack(pipeline.stargan_generator, original, target_attr_tensor)
                timings["deepfake_clean"] = _elapsed(t0, pipeline.device)

                t0 = time.perf_counter()
                vaccinated_deepfaked = deepfake_attack(pipeline.stargan_generator, vaccinated, target_attr_tensor)
                timings["deepfake_vaccinated"] = _elapsed(t0, pipeline.device)

                t0 = time.perf_counter()
                defense_metrics = calculate_metrics(clean_deepfaked, vaccinated_deepfaked)
                visual_metrics = calculate_metrics(original, vaccinated)
                timings["metrics"] = _elapsed(t0, pipeline.device)

                t0 = time.perf_counter()
                to_pil(original).save(attack_out / "original.png")
                to_pil(vaccinated).save(attack_out / "vaccinated.png")
                to_pil(clean_deepfaked).save(attack_out / "deepfake_clean.png")
                to_pil(vaccinated_deepfaked).save(attack_out / "deepfake_vaccinated.png")
                to_pil((perturbation * 10).clamp(0, 1)).save(attack_out / "perturbation_x10.png")
                timings["save_outputs"] = _elapsed(t0, pipeline.device)

                timings["total"] = _elapsed(image_total_start, pipeline.device)
                defense_status = "SUCCESS" if defense_metrics["L2"] > 0.05 else "FAILED"

                _write_timings(attack_out / "timings.txt", timings)
                _write_metrics(attack_out / "defense_metrics.txt", defense_metrics, defense_status, attribute_name)
                _write_metrics(attack_out / "visual_metrics.txt", visual_metrics, "N/A", attribute_name)

                row = {
                    "image": image_path.name,
                    "attribute": attribute_name,
                    "status": "ok",
                    "defense_status": defense_status,
                    "defense_L2": f"{defense_metrics['L2']:.6f}",
                    "visual_PSNR": f"{visual_metrics['PSNR']:.6f}",
                    "visual_SSIM": f"{visual_metrics['SSIM']:.6f}",
                }
                for key, val in timings.items():
                    row[f"time_{key}_ms"] = f"{val * 1000:.3f}"
                rows.append(row)
            except Exception as exc:
                timings["total"] = _elapsed(image_total_start, pipeline.device)
                _write_timings(attack_out / "timings.txt", timings)
                (attack_out / "error.txt").write_text(str(exc) + "\n")
                rows.append(
                    {
                        "image": image_path.name,
                        "attribute": attribute_name,
                        "status": "error",
                        "defense_status": "N/A",
                        "defense_L2": "N/A",
                        "visual_PSNR": "N/A",
                        "visual_SSIM": "N/A",
                        "time_total_ms": f"{timings['total'] * 1000:.3f}",
                    }
                )

    elapsed_total = _elapsed(total_start, pipeline.device)
    summary_path = output_root / "timings_summary.csv"

    # Keep stable columns even if some rows are error rows.
    base_columns = [
        "image",
        "attribute",
        "status",
        "defense_status",
        "defense_L2",
        "visual_PSNR",
        "visual_SSIM",
        "time_load_image_ms",
        "time_attention_map_ms",
        "time_vaccinate_image_ms",
        "time_deepfake_clean_ms",
        "time_deepfake_vaccinated_ms",
        "time_metrics_ms",
        "time_save_outputs_ms",
        "time_total_ms",
    ]

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=base_columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    totals = [float(r["time_total_ms"]) for r in ok_rows if r.get("time_total_ms")]
    aggregate = {
        "inputs_total": len(image_paths),
        "attack_vectors": len(selected_attributes),
        "images_total": len(rows),
        "images_ok": len(ok_rows),
        "images_error": len(rows) - len(ok_rows),
        "batch_total_ms": elapsed_total * 1000.0,
        "per_image_total_ms_mean": float(np.mean(totals)) if totals else float("nan"),
        "per_image_total_ms_std": float(np.std(totals)) if totals else float("nan"),
    }
    aggregate_path = output_root / "batch_aggregate.txt"
    aggregate_path.write_text(
        "\n".join(
            [
                f"inputs_total: {aggregate['inputs_total']}",
                f"attack_vectors: {aggregate['attack_vectors']}",
                f"images_total: {aggregate['images_total']}",
                f"images_ok: {aggregate['images_ok']}",
                f"images_error: {aggregate['images_error']}",
                f"batch_total_ms: {aggregate['batch_total_ms']:.3f}",
                f"per_image_total_ms_mean: {aggregate['per_image_total_ms_mean']:.3f}",
                f"per_image_total_ms_std: {aggregate['per_image_total_ms_std']:.3f}",
            ]
        )
        + "\n"
    )
    return aggregate


def process_folder(
    pipeline: "DefensePipeline",
    input_dir: str,
    output_dir: str = "batch_results",
    attribute: str = "Blonde Hair",
    domain: str = "celeba",
    expressions: Optional[List[str]] = None,
) -> Dict[str, float]:
    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        raise ValueError(f"Input folder not found: {input_dir}")

    image_paths = sorted(
        [p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]
    )
    if not image_paths:
        raise ValueError(f"No supported images found in: {input_dir}")

    return _process_image_paths(pipeline, image_paths, output_dir, attribute, domain, expressions)


def process_single_image(
    pipeline: "DefensePipeline",
    image_path: str,
    output_dir: str = "batch_results",
    attribute: str = "Blonde Hair",
    domain: str = "celeba",
    expressions: Optional[List[str]] = None,
) -> Dict[str, float]:
    path = Path(image_path)
    if not path.exists() or not path.is_file():
        raise ValueError(f"Image not found: {image_path}")
    if path.suffix.lower() not in VALID_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {path.suffix}")
    return _process_image_paths(pipeline, [path], output_dir, attribute, domain, expressions)


def parse_args():
    parser = argparse.ArgumentParser(description="Batch/single-image deepfake defense runner")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--input-dir", type=str, help="Folder of images to process")
    group.add_argument("--image", type=str, help="Single image to process")
    parser.add_argument("--output-dir", type=str, default="batch_results")
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
        help="Comma-separated expression labels in checkpoint order for RaFD (default: 8-label RaFD order).",
    )
    parser.add_argument(
        "--attribute",
        type=str,
        default="Blonde Hair",
        help="Attack vector: exact name, comma-separated names, or 'all'",
    )
    parser.add_argument("--list-attributes", action="store_true", help="Print supported attack vectors and exit")
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--enable-lpips", action="store_true", help="Initialize LPIPS (optional)")
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
    if not args.input_dir and not args.image:
        raise SystemExit("One of --input-dir or --image is required.")
    if args.domain == "rafd" and not args.expression_checkpoint:
        raise SystemExit("--expression-checkpoint is required when --domain rafd.")
    if args.domain == "rafd" and args.attribute == "Blonde Hair":
        args.attribute = "Happy"

    try:
        from .pipeline import DefensePipeline
    except ImportError:
        from deepfake_defense.pipeline import DefensePipeline

    configure_runtime()
    set_seed(42)
    device = get_device()
    pipeline = DefensePipeline(
        device=device,
        epsilon=args.epsilon,
        enable_lpips=args.enable_lpips,
        domain=args.domain,
        expression_checkpoint=args.expression_checkpoint,
        expression_labels=expressions,
    )

    if args.image:
        aggregate = process_single_image(
            pipeline=pipeline,
            image_path=args.image,
            output_dir=args.output_dir,
            attribute=args.attribute,
            domain=args.domain,
            expressions=expressions,
        )
    else:
        aggregate = process_folder(
            pipeline=pipeline,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            attribute=args.attribute,
            domain=args.domain,
            expressions=expressions,
        )

    print("Processing complete")
    print(f"Images total: {aggregate['images_total']}")
    print(f"Images ok: {aggregate['images_ok']}")
    print(f"Images error: {aggregate['images_error']}")
    print(f"Batch total ms: {aggregate['batch_total_ms']:.3f}")


if __name__ == "__main__":
    main()
