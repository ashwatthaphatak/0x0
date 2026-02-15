# 0x0

Repository for the 2026 Edition of HackNCState

Team Memebers

Ashwattha Phatak

Akshay Dongare

Anish Mulay

## Deepfake Defense (Script Version)

The notebook has been split into Python modules under `deepfake_defense/` with a single entrypoint:

```bash
python3 run_defense.py --mode demo --image sample_images/sample_0.jpg --attribute "Blonde Hair"
```

Outputs are written to `results/`:

- `results/step1_attention.png`
- `results/step2_perturbation.png`
- `results/final_comparison.png`

Optional Gradio app:

```bash
python3 run_defense.py --mode gradio --share
```

Batch processing (one subfolder per image + per-stage timings):

```bash
python3 run_defense.py --mode batch --input-dir sample_images --output-dir batch_results --attribute "Blonde Hair"
```

Run the same logic directly via `batch.py`:

```bash
python3 -m deepfake_defense.batch --input-dir sample_images --output-dir batch_results --attribute "Blonde Hair"
python3 -m deepfake_defense.batch --image sample_images/sample_0.jpg --output-dir batch_results_single --attribute "Blonde Hair"
```

Extensive attack coverage:

```bash
# Run every supported StarGAN attack vector on each image.
python3 -m deepfake_defense.batch --input-dir sample_images --output-dir batch_results_all --attribute all

# Run a custom subset.
python3 -m deepfake_defense.batch --input-dir sample_images --output-dir batch_results_subset --attribute "Black Hair + Male + Young,Brown Hair + Female + Old"

# List available attack vectors.
python3 -m deepfake_defense.batch --list-attributes
```

Real expression model path (additional, keeps CelebA path unchanged):

```bash
# 1) Bootstrap official StarGAN v1 repo.
./.venv_deepfake/bin/python scripts/rafd_checkpoint_helper.py bootstrap --repo-dir external/stargan_v1

# 2) Print train/test commands for RaFD (you must have RaFD dataset access).
./.venv_deepfake/bin/python scripts/rafd_checkpoint_helper.py train-cmd --repo-dir external/stargan_v1 --rafd-image-dir /absolute/path/to/RaFD/train

# 3) Validate that your trained checkpoint is truly RaFD-compatible (c_dim=8).
./.venv_deepfake/bin/python scripts/rafd_checkpoint_helper.py validate --checkpoint /absolute/path/to/stargan_rafd/models/200000-G.ckpt

# 4) Run real expression attacks with that checkpoint.
python3 -m deepfake_defense.batch --domain rafd --expression-checkpoint /absolute/path/to/stargan_rafd/models/200000-G.ckpt --input-dir sample_images --output-dir batch_results_rafd --attribute all

# Optional: provide explicit label order if your checkpoint used a different class order.
python3 -m deepfake_defense.batch --domain rafd --expression-checkpoint /absolute/path/to/stargan_rafd/models/200000-G.ckpt --expression-labels \"Angry,Contemptuous,Disgusted,Fearful,Happy,Neutral,Sad,Surprised\" --input-dir sample_images --output-dir batch_results_rafd --attribute \"Happy,Sad,Angry\"
```

Batch outputs:

- `batch_results/<image_name>/original.png`
- `batch_results/<image_name>/deepfake_clean.png`
- `batch_results/<image_name>/deepfake_vaccinated.png`
- `batch_results/<image_name>/timings.txt`
- `batch_results/timings_summary.csv`
- `batch_results/batch_aggregate.txt`

GPU dependant -> Mac version (MPS)

1. Replaced CUDA-only device logic with a portable backend chooser: cuda (if present) → mps (Apple Metal on Mac) → cpu, and enabled PYTORCH_ENABLE_MPS_FALLBACK=1 so unsupported MPS ops automatically fall back to CPU.
2. Removed hardcoded device resets in later StarGAN cells, so the notebook no longer accidentally switches back to CUDA/CPU-only paths; checkpoint loading is done with map_location="cpu" for portability.
3. Made LPIPS non-blocking (and skipped by default), because it is not used in the defense math path; this removes the runtime failure/download path and avoids requiring CUDA-linked behavior for that component.
4. Reworked setup into a managed .venv_deepfake flow (create/reuse + kernel registration) with notebook-safe installs, so the same Mac-compatible environment is reused every run without CUDA-specific install assumptions.

./.venv_deepfake/bin/python -m deepfake_defense.batch \
 --domain rafd \
 --expression-checkpoint "./stargan_celeba_128/models/200000-G.ckpt" \
 --input-dir sample_images \
 --output-dir batch_results_rafd \
 --attribute all
