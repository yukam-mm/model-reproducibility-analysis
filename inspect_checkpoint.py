import torch
from pathlib import Path

CKPT = Path("fungi_densenet121_Full_B32_E1000_lr1.000e-04_Adam_KLDIV.pth")

state = torch.load(CKPT, map_location = "cpu", weights_only = True)

# Check 1
print(f"Loaded object type: {type(state).__name__}")
print(f"Top-level key count: {len(state)}")

first_key = next(iter(state.keys()))

print(f"First key: '{first_key}'")
print(f"Frist value type: {type(state[first_key]).__name__}")
print()

# Check 2 input channels via conv1.weight
conv1_shape = tuple(state["conv1.weight"].shape)
print(f"conv1.weight shape: {conv1_shape}")

print(f" in channels = {conv1_shape[1]}")
print()

# Check 3 num_classes via fc1.weight
fc1_shape = tuple(state["fc1.weight"].shape)
print(f"fc1.weight shape: {fc1_shape}")
print(f" num_classes = {fc1_shape[0]}")

# check 4 how many tensors total, and what do the key names look like?
keys = list(state.keys())
print(f"Total tesnor keys: {len(keys)}")

print("First 5 keys:")
for k in keys[:5]:
    print(f" {k}")
print("Last 5 keys:")
for k in keys[-5:]:
    print(f" {k}")

