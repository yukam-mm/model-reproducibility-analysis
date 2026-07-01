import csv
import time
from collections import Counter
from pathlib import Path
from statistics import mean

import torch
from densenet import DenseNet, model_parameters
from preprocess import load_image

CKPT = Path("fungi_densenet121_Full_B32_E1000_lr1.000e-04_Adam_KLDIV.pth")
MACRO=Path("macro")
OUT_CSV = Path("predictions.csv")

# Loading the model
model = DenseNet(
    densenet_variant = model_parameters["densenet121"],
    in_channels = 3,
    num_classes = 110,
)

model.load_state_dict(
    torch.load(CKPT, map_location = "cpu", weights_only = True),
    strict = True
)
model.eval()

image_paths = sorted(MACRO.rglob("*.jpg"))
print(f"Found {len(image_paths)} images. And running inference.....")

# CSV writer
csv_file = open(OUT_CSV, "w", newline = "")
writer = csv.writer(csv_file)

header = ["folder", "filename", "top1_class", "top1_prob"]
for k in range(2,6):
    header.extend([f"top{k}_class", f"top{k}_prob"])
writer.writerow(header)

# in memory data for the summary
rows_by_folder: dict[str, list[tuple[int, float]]] = {}

# writing to csv
start = time.perf_counter()
with torch.inference_mode():
    for i, path in enumerate(image_paths):
        x = load_image(path)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        top5_probs, top5_idx = torch.topk(probs, k=5)

        folder = path.parent.name

        # Build CSV
        row = [folder, path.name,
               top5_idx[0].item(), round(top5_probs[0].item(), 4)]
        for k in range(1,5):
            row.extend([
                top5_idx[k].item(),
                round(top5_probs[k].item(), 4),
            ])
        writer.writerow(row)

        rows_by_folder.setdefault(folder, []).append(
            (top5_idx[0].item(), top5_probs[0].item())
        )

csv_file.close()

elapsed = time.perf_counter() - start
print(f"\nDone in {elapsed:.1f}s. Results saved to {OUT_CSV}\n")



