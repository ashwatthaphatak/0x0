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
