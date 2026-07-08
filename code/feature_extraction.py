# feature_extraction.py

import torch
import numpy as np
import os
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.efficientnet_b0(
    weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
)
model.classifier = torch.nn.Identity()
model.eval()
model.to(device)


class PlantImageDataset(Dataset):
    def __init__(self, data_dir, image_size=(224, 224)):
        self.image_paths = []
        self.labels = []
        self.label_map = {}

        class_names = sorted([
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ])

        for idx, label_name in enumerate(class_names):
            class_dir = os.path.join(data_dir, label_name)
            self.label_map[label_name] = idx

            for img_file in sorted(os.listdir(class_dir)):
                path = os.path.join(class_dir, img_file)
                if os.path.isfile(path) and img_file.lower().endswith(
                    ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
                ):
                    self.image_paths.append(path)
                    self.labels.append(idx)

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


def extract_features(data_dir, out_file, batch=32):

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    torch.manual_seed(42)
    np.random.seed(42)

    dataset = PlantImageDataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=0)

    X_chunks = []
    y_chunks = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            feats = model(images)

            X_chunks.append(feats.cpu().numpy())
            y_chunks.append(np.array(labels))

    X = np.concatenate(X_chunks, axis=0) if X_chunks else np.zeros((0, 1280))
    y = np.concatenate(y_chunks, axis=0) if y_chunks else np.array([])

    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    # ✅ FIX: reconstruct label_map correctly from dataset
    label_map = dataset.label_map

    np.savez(out_file, X=X, y=y, label_map=label_map)

    print(f"Saved {out_file} (features={X.shape}, labels={y.shape})")
    return out_file

def extract_features_from_image(image):

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model(img_tensor)

    return features.cpu().numpy().flatten()