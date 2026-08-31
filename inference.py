import torch

checkpoint = torch.load(
    "arrhenius_runs/slurm_test/checkpoints/latest.pt",
    map_location="cpu",
    weights_only=False,
)

print(checkpoint["args"])
print("Epoch:", checkpoint["epoch"])