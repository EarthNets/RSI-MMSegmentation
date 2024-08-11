import os
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image

class GamusDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.image_dir = os.path.join(root_dir, 'images', split)
        self.class_dir = os.path.join(root_dir, 'classes', split)
        self.height_dir = os.path.join(root_dir, 'heights', split)

        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('IMG.h5')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        base_name = img_file[:-6]  # Remove 'IMG.h5'

        # Load image
        img_path = os.path.join(self.image_dir, img_file)
        with h5py.File(img_path, 'r') as f:
            image = f['data'][()]  # Assuming the image is stored as a numpy array

        # Load class
        cls_file = f"{base_name}CLS.h5"
        cls_path = os.path.join(self.class_dir, cls_file)
        with h5py.File(cls_path, 'r') as f:
            class_label = f['data'][()]

        # Load height
        agl_file = f"{base_name}AGL.h5"
        agl_path = os.path.join(self.height_dir, agl_file)
        with h5py.File(agl_path, 'r') as f:
            height = f['data'][()]

        # Convert image to PIL Image for transforms
        image = Image.fromarray(image.astype(np.uint8))

        if self.transform:
            image = self.transform(image)

        return image, class_label, height


if __name__=="__main__":

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = GamusDataset(root_dir='./', split='train', transform=transform)
    val_dataset = GamusDataset(root_dir='./', split='val', transform=transform)
    test_dataset = GamusDataset(root_dir='./', split='test', transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

