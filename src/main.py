import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from transformers import ViTFeatureExtractor, ViTForImageClassification, ViTImageProcessor, TrainingArguments, Trainer

from evaluate import load

from custom_dataset import CustomDataset
import json

from src.attack import PGDAttack
from src.test import test_model, pgd_test_model
from src.utils import show_image, show_adversarial_pgd_image, attack_pgd_and_save_images

with open('config.json', 'r') as f:
    config = json.load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# device = torch.device('cpu')

# model_name_or_path = 'google/vit-base-patch16-224-in21k'
model_name_or_path = '../saved_model'
processor = ViTImageProcessor.from_pretrained(model_name_or_path)

test_dataset = CustomDataset(
    root_dir=config['testing_data']['path_orig_test']
)

adversarial_test_dataset = CustomDataset(
    root_dir=config['testing_data']['path_pgd_test']
)

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=config['data']['batch_size']
)

adversarial_test_dataloader = DataLoader(
    dataset=adversarial_test_dataset,
    batch_size=config['data']['batch_size']
)

model = ViTForImageClassification.from_pretrained(
    pretrained_model_name_or_path=model_name_or_path,
    num_labels=2
).to(device)

model.eval()

print('Original test:')

test_model(
    model=model,
    loss_fn=CrossEntropyLoss(),
    device=device,
    test_dataloader=DataLoader(
        dataset=test_dataset,
        batch_size=config['data']['batch_size']
    )
)

attack = PGDAttack(
    model=model,
    epsilon=0.01,
    steps=10,
    step_size=0.001
)

attack_pgd_and_save_images(
    attacked_dir='D:\\pythonProjects\\AIRetailVision\\data_for_testing\\original_test',
    output_dir='D:\\pythonProjects\\AIRetailVision\\data_for_testing\\pgd_test_eps_001_step_size_0001_steps_10',
    batch_size=config['data']['batch_size'],
    attack=attack,
    device=device
)

# для просмотра изображений
# for batch in test_dataloader:
#     # print(batch)
#     show_image(
#         image_tensor=batch['pixel_values'][0],
#         save_name='orig_image'
#     )
#     show_adversarial_pgd_image(
#         image_tensor=batch['pixel_values'].to(device),
#         label=batch['labels'].to(device),
#         attack=attack
#     )
#

print('PGD test:')

test_model(
    model=model,
    loss_fn=CrossEntropyLoss(),
    device=device,
    test_dataloader=adversarial_test_dataloader
)

print('\nPGD test on-the-fly:')

pgd_test_model(
    model=model,
    loss_fn=CrossEntropyLoss(),
    device=device,
    test_dataloader=DataLoader(
        dataset=test_dataset,
        batch_size=config['data']['batch_size']
    ),
    attack=attack
)
