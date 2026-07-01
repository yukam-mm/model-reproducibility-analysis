# DenseNet-121 Fungal Identification — Reproducibility Analysis

## Overview

Reproducibility is a fundamental requirement in machine learning research, emphasized by major venues such as NeurIPS and initiatives like the ML Reproducibility Challenge. In practice, however, downstream users — a graduate student adapting a model to a new dataset, or a reviewer replicating an experiment — pay the cost of missing or ambiguous documentation.

This project takes one model from a published deep learning paper and documents the full path from checkpoint to working inference pipeline. It uses the authors' released DenseNet-121 checkpoint from Mansourvar et al. (2025), applied to a separate, publicly available fungal image dataset (OpenFungi).

## Common Reproducibility Barriers

1. Closed training dataset, or repository not published.
2. Repository and dataset are public, but the inference workflow is incomplete.
3. Released code does not reproduce the reported behavior due to environment mismatches.

## What This Repository Demonstrates

- Verifying the checkpoint file.
- Verifying that the architecture matches.
- Verifying that weights load cleanly.
- Sanity-checking preprocessing on a single image.
- Running inference.
- Writing results to CSV.

## Repository Structure

```
densenet/
├── README.md                     # This file
├── densenet.py                   # Faithful copy of the paper's DenseNet implementation
├── inspect_checkpoint.py         # Step 1: verify checkpoint shape/keys without loading model
├── check_arch.py                 # Step 2: instantiate architecture and confirm shape match
├── load_weights.py               # Step 3: load state_dict with strict=True
├── preprocess.py                 # Step 4: PIL → normalized (1, 3, 224, 224) tensor
├── run_inference.py              # Step 5: iterate over images, print top-5
├── save_results.py               # Step 6: dump predictions to CSV + per-folder summary
├── predictions.csv               # Output: 385 rows, top-5 per image
├── fungi_densenet121_Full_B32_E1000_lr1.000e-04_Adam_KLDIV.pth   # Checkpoint (not in repo)
└── macro/                        # Inference dataset (not in repo)
```

## Pipeline Overview

```
JPG image
  → PIL.Image.open, convert to RGB
  → Resize((224, 224), antialias=True)
  → ToImage → ToDtype(float32, scale=True)
  → Normalize(mean=[0.7186, 0.6085, 0.5382], std=[0.1204, 0.1915, 0.1695])
  → unsqueeze batch dim → (1, 3, 224, 224)
  → DenseNet-121 (in_channels=3, num_classes=110), eval mode
  → softmax(dim=1)
  → topk(k=5) → per-image top-5 (class_index, probability) pairs
```

Normalization constants are those defined for the paper's validation split, which is the closest analog to unseen inference data.

## Reproducing This Reproduction

Environment setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision pillow
```
Sequence:
 
```bash
python3 inspect_checkpoint.py    
python3 check_arch.py            
python3 load_weights.py          
python3 preprocess.py            
python3 run_inference.py         
python3 save_results.py          
```
## References

- Mansourvar, M., Charylo, K. R., Frandsen, R. J. N., Brewer, S. S., & Hoof, J. B. (2025). Automated Fungal Identification with Deep Learning on Time-Lapse Images. Information, 16(2), 109. https://doi.org/10.3390/info16020109
- Cighir, A., Bolboacă, R., & Lenard, T. (2025). OpenFungi: A Machine Learning Dataset for Fungal Image Recognition Tasks. Life, 15(7), 1132. https://doi.org/10.3390/life15071132
