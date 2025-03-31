import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from transformers import ViTFeatureExtractor, ViTForImageClassification, ViTImageProcessor, TrainingArguments, Trainer

from evaluate import load

from custom_dataset import CustomDataset
import json

from src.train import test_model

with open('config.json', 'r') as f:
    config = json.load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name_or_path = 'google/vit-base-patch16-224-in21k'
processor = ViTImageProcessor.from_pretrained(model_name_or_path)

train_dataset = CustomDataset(
    root_dir=config['data']['path_train'],
    processor=processor
)

valid_dataset = CustomDataset(
    root_dir=config['data']['path_valid'],
    processor=processor
)

test_dataset = CustomDataset(
    root_dir=config['data']['path_test'],
    processor=processor
)

model = ViTForImageClassification.from_pretrained(
    pretrained_model_name_or_path=model_name_or_path,
    num_labels=2
).to(device)

training_args = TrainingArguments(
    output_dir='../vit_tuned',
    eval_strategy='epoch',
    per_device_train_batch_size=config['data']['batch_size'],
    per_device_eval_batch_size=config['data']['batch_size'],
    num_train_epochs=3,
    learning_rate=5e-5,
    weight_decay=0.01
)


metric = load("accuracy")

def compute_metrics(p):
    global metric
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics
)

print('Test before training')

test_model(
    model=model,
    loss_fn=torch.nn.CrossEntropyLoss(),
    device=device,
    test_dataloader=DataLoader(
        dataset=test_dataset,
        batch_size=config['data']['batch_size']
    )
)

trainer.train()

trainer.save_model(
    output_dir='../saved_model'
)

print('Test after training')

test_model(
    model=model,
    loss_fn=CrossEntropyLoss(),
    device=device,
    test_dataloader=DataLoader(dataset=test_dataset)
)



