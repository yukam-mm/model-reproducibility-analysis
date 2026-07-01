from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import v2

# from the repo's validation images (because it is the closest match)
VAL_MEAN = [0.7186, 0.6085, 0.5382]
VAL_STD  = [0.1204, 0.1915, 0.1695]

transform = v2.Compose([
    v2.Resize((224, 224) , antialias = True),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean = VAL_MEAN, std = VAL_STD),
])

def load_image(path: str | Path)-> torch.Tensor:
    img = Image.open(path).convert("RGB")
    tensor = transform(img)
    return tensor.unsqueeze(0)

if __name__ == "__main__":
    macro_dir = Path("macro")
    sample = next(macro_dir.rglob("*.jpg"), None)
    if sample is None:
        raise SystemExit("No files!!!")
    print(f"Sample file {sample}")

    # to show what the raw imag looks like before preprocessing
    raw = Image.open(sample)
    print(f"Raw image: size = {raw.size}, mode = {raw.mode}")

    # run through the full pipeline
    x = load_image(sample)
    print(f"Tensor : shape = {tuple(x.shape)}, dtype = {x.dtype}")

    print(f"Value stats: min = {x.min().item():+.3f},"
          f"max = {x.max().item():+.3f}, "
          f"mean = {x.mean().item():+.3f}, "
          f"std = {x.std().item():+.3f}")