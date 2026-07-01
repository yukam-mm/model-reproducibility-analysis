# Sanity-check the architecture

import torch
from densenet import DenseNet, model_parameters

model = DenseNet(
    densenet_variant = model_parameters["densenet121"],
    in_channels = 3,
    num_classes = 110,
)

n_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {n_params:,}")

print(f"conv1.weight: {tuple(model.conv1.weight.shape)}") # (64,3,7,7) from the original paper
print(f"fc1.weight: {tuple(model.fc1.weight.shape)}") # (110, 1024) from the original paper

sd_keys = list(model.state_dict().keys())
print(f"state_dict keys: {len(sd_keys)}") # 727

# Forward-pass smoke test with a random 224*224 RGB image
model.eval()
dummy = torch.randn(1,3,224,224)
with torch.no_grad():
    out = model(dummy)
print(f"Forward pass OK. Output shape : {tuple(out.shape)}") # (1,110)

