import pathlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from fastmri.pl_modules import UnetModule

base_unet = UnetModule(chans=256)

checkpoint_path = "knee_sc_leaderboard_state_dict.pt"
state_dict = torch.load(checkpoint_path, map_location="cpu")

# Load only into the unet submodule
base_unet.unet.load_state_dict(state_dict)
unet_model = base_unet.unet

from fastmri.data import SliceDataset
from fastmri.data.subsample import RandomMaskFunc
from torch.utils.data import DataLoader
from fastmri.data.transforms import UnetDataTransform


# Example mask function and data path
mask_func = RandomMaskFunc(center_fractions=[0.08], accelerations=[4])
# data_path = "refiner_test/reconstructions"
data_path = "singlecoil_refiner_test"

transform = UnetDataTransform(mask_func=mask_func, use_seed=False, which_challenge="singlecoil")

knee_dataset = SliceDataset(
    root=data_path,
    transform=transform,
    challenge="singlecoil",
    sample_rate=1.0,
)

# Wrap in DataLoader
knee_dataloader = DataLoader(knee_dataset, batch_size=1, num_workers=4)

class RefinerNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.refine(x)


def train_refiner(unet_model, refiner_model, dataloader, device="cuda"):
    unet_model.eval()
    refiner_model.train()
    loss_fn = nn.L1Loss()
    optimizer = optim.Adam(refiner_model.parameters(), lr=1e-4)

    for sample in dataloader:
        input_img = sample.image
        target_img = sample.target

        input_img = input_img.to(device)
        target_img = target_img.to(device)

        with torch.no_grad():
            unet_output = unet_model(input_img)

        refined_output = refiner_model(unet_output)
        loss = loss_fn(refined_output, target_img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item():.4f}")


knee_refiner = RefinerNet()
train_refiner(unet_model, knee_refiner, knee_dataloader, device="cpu")

# brain_refiner = RefinerNet()
# train_refiner(unet_model, brain_refiner, brain_dataloader, device="cuda")


