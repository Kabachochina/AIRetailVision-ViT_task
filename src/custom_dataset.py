import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import ViTImageProcessor

class CustomDataset(Dataset):
    def __init__(
            self,
            root_dir: str,
            processor: ViTImageProcessor
    ) -> None:
        self.root_dir = root_dir
        self.processor = processor
        self.samples = []

        licensed = ['BABY_PRODUCTS', 'BEAUTY_HEALTH', 'ELECTRONICS', 'GROCERY', 'PET_SUPPLIES']
        unlicensed = ['CLOTHING_ACCESSORIES_JEWELLERY', 'HOBBY_ARTS_STATIONARY', 'HOME_KITCHEN_TOOLS', 'SPORTS_OUTDOOR']

        for category in os.listdir(root_dir):
            category_path = os.path.join(self.root_dir, category)
            if not os.path.isdir(category_path):
                continue

            if category in licensed:
                label = 1
            elif category in unlicensed:
                label = 0
            else:
                continue

            for filename in os.listdir(category_path):
                if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(category_path, filename)
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> (torch.Tensor, torch.Tensor):
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert('RGB')
        inputs = self.processor(
            images=image,
            return_tensors='pt'
        )
        pixel_values = inputs['pixel_values'].squeeze(0) # убираем измерение батча
        return {'pixel_values': pixel_values, 'labels': torch.tensor(label)}







