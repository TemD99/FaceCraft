import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import logging
from config import *
from math import log2
import matplotlib.pyplot as plt
import pandas as pd
from torchvision.transforms import ToTensor, Normalize, Compose
import zipfile


def get_loader():
    # Apply some transformation to the images
    transform = transforms.Compose(
        [
            # resize the images to the resolution that we want
            transforms.Resize((2 ** LOG_RESOLUTION, 2 ** LOG_RESOLUTION)),
            # convert them to tensors
            transforms.ToTensor(),
            #  apply some augmentation
            transforms.RandomHorizontalFlip(p=0.5),
            # normalize them to be all the pixels ranging from -1 to 1
            transforms.Normalize(
                mean=[0.5211, 0.4260, 0.3812], # for 128 dataset
                std=[0.2469, 0.2210, 0.2182]
            ),
        ]
    )
    # Prepare the dataset by using ImageFolder
    dataset = datasets.ImageFolder(root=DATASET, transform=transform)
    # Create mini-batch sizes
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    # return the loader
    return loader

def check_loader():
    loader = get_loader()  # Corrected to not expect two outputs
    # Iterate through the first batch from the loader to display images
    for cloth, _ in loader:
        _, ax = plt.subplots(3, 3, figsize=(8, 8))
        plt.suptitle('Some real samples')
        ind = 0
        for k in range(3):
            for kk in range(3):
                # Ensure that we do not go out of index bounds
                if ind >= len(cloth):
                    break
                ax[k][kk].imshow((cloth[ind].permute(1, 2, 0) + 1) / 2)
                ind += 1
        plt.show()
        break  # Only show the first batch

class CelebDataset(Dataset):
    def __init__(self, image_dir, attribute_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform if transform else Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.attributes = pd.read_csv(attribute_file, delim_whitespace=True, header=1)
        self.image_ids = self.attributes.index.values

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = str(self.image_ids[idx]).zfill(6)
        image_path = os.path.join(self.image_dir, f"{image_id}")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        attributes = torch.tensor(self.attributes.iloc[idx].values, dtype=torch.float32)

        return image, attributes

# Dataset class for loading images from a zip file
class ZipDataset(Dataset):
    def __init__(self, zip_path, transform=None):
        self.zip_path = zip_path
        self.transform = transform
        self.images = self.load_images()

    def load_images(self):
        # Function to load images from the zip file
        images = []
        with zipfile.ZipFile(self.zip_path, 'r') as z:
            for filename in z.namelist():
                if filename.endswith('.png') or filename.endswith('.jpg'):
                    with z.open(filename) as f:
                        img = Image.open(f)
                        img = img.convert('RGB')
                        if self.transform:
                            img = self.transform(img)
                        images.append(img)
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]

