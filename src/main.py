import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from custom_dataset import CustomDataset
import json

from src.train import train_model, test_model
from src.utils import save_model
from src.vit_model import ViTClassifier

with open('config.json', 'r') as f:
    config = json.load(f)

transform = transforms.Compose([
    transforms.Resize(config['data']['image_size']),
    transforms.ToTensor()
])

train_transform = transforms.Compose([
    transforms.RandAugment(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = CustomDataset(
    root_dir=config['data']['path_train'],
    transform=train_transform
)

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=config['data']['batch_size'],
    shuffle=True
)

valid_dataset = CustomDataset(
    root_dir=config['data']['path_valid'],
    transform=transform
)

valid_dataloader = DataLoader(
    dataset=valid_dataset,
    batch_size=config['data']['batch_size'],
    shuffle=True
)

test_dataset = CustomDataset(
    root_dir=config['data']['path_test'],
    transform=transform
)

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=config['data']['batch_size'],
    shuffle=True
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

model = ViTClassifier(
    num_transformer_layers=6,
    embed_dim=128,
    num_heads=4,
    patch_size=16,
    num_patches=((config['data']['image_size'][0] // 16) ** 2), # делим на patch size
    mlp_dim=128*2,
    num_classes=1,
    device=device
).to(device)

optimizer = torch.optim.AdamW(
    params=model.parameters(),
    lr = 5e-5
)

loss_fn = torch.nn.BCEWithLogitsLoss()

train_model(
    model=model,
    loss_fn=loss_fn,
    epochs=10,
    device=device,
    optimizer=optimizer,
    train_dataloader=train_dataloader,
    valid_dataloader=valid_dataloader
)

test_model(
    model=model,
    loss_fn=loss_fn,
    device=device,
    test_dataloader=test_dataloader
)

save_model(
    model=model,
    path='../saved_models/test_model_1.ckpt'
)



