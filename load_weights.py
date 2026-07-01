import torch
from pathlib import Path
from densenet import DenseNet, model_parameters

CKPT = Path("fungi_densenet121_Full_B32_E1000_lr1.000e-04_Adam_KLDIV.pth")

model = DenseNet(
    densenet_variant = model_parameters["densenet121"],
    in_channels = 3,
    num_classes = 110,
)

state_dict = torch.load(CKPT, map_location="cpu", weights_only = True)

result = model.load_state_dict(state_dict, strict = True)
print(f"Missing keys: {result.missing_keys}")
print(f"Unexpected keys: {result.unexpected_keys}")

model.eval()

loaded_conv1 = model.conv1.weight.detach()
file_conv1 = state_dict["conv1.weight"]
print(f"conv1.weight loaded correctly: {torch.equal(loaded_conv1, file_conv1)}")

print(f"conv1.weight[0,0,0,:] = {model.conv1.weight[0,0,0, :].tolist()}")
print("Everything is OK. Model ready for interence yey")