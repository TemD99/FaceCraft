from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from config import *

# Setup the dataset and DataLoader
transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
dataset = datasets.ImageFolder(root=DATASET, transform=transform)
loader = DataLoader(dataset, batch_size=56, num_workers=4, shuffle=False)

# Function to calculate mean and std
def get_mean_std(loader):
    # Variances and means
    mean = 0.
    std = 0.
    total_images_count = 0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    # Final mean and std
    mean /= total_images_count
    std /= total_images_count

    return mean, std

mean, std = get_mean_std(loader)
print(f"Calculated Mean: {mean}")
print(f"Calculated Std: {std}")