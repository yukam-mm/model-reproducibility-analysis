# Running the loaded model on the dataset

import time
from pathlib import Path

import torch
from densenet import DenseNet, model_parameters
from preprocess import load_image

CKPT = Path("fungi_densenet121_Full_B32_E1000_lr1.000e-04_Adam_KLDIV.pth")
MACRO = Path("macro")

# Build model+load weights
model = DenseNet(
    densenet_variant = model_parameters["densenet121"],
    in_channels = 3,
    num_classes = 110,
)

model.load_state_dict(
    torch.load(CKPT, map_location= "cpu", weights_only = True),
    strict = True,
)
model.eval()

# Dataset
image_paths = sorted(MACRO.rglob("*.jpg"))
print(f"Found {len(image_paths)} images under {MACRO}/")
print()

# Inference loop
start = time.perf_counter()

with torch.inference_mode():
    for i, path in enumerate(image_paths):
        x = load_image(path)
        logits = model(x)
        # softmax- probability distribution
        probs = torch.softmax(logits, dim=1).squeeze(0)

        top5_probs, top5_idx = torch.topk(probs, k=5)
        top1_idx = top5_idx[0].item()
        top1_prob = top5_probs[0].item()

        # print result
        folder = path.parent.name
        top5_str = " , ".join(
            f"{idx.item()}({p.item():.2f})"
            for idx, p in zip(top5_idx, top5_probs)
        )
        print(
            f"[{i+1:4}/{len(image_paths)}] {folder:32} "
            f"{path.name:35} -> class {top1_idx:3} "
            f"(p={top1_prob:.3f})   top5: {top5_str}"
        )

# Timing summary
elapsed = time.perf_counter() - start
print()
