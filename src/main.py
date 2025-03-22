from torch.utils.data import DataLoader
from torchvision import transforms
from custom_dataset import CustomDataset
import json


with open('config.json', 'r') as f:
    config = json.load(f)

transform = transforms.Compose([
    transforms.Resize(config['data']['image_size']),
    transforms.ToTensor()
])

check_dataset = CustomDataset(
    root_dir=config['data']['path_check'],
    transform=transform
)

check_dataloader = DataLoader(
    dataset=check_dataset,
    batch_size=config['data']['batch_size'],
    shuffle=True
)






