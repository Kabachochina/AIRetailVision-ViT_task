from typing import Callable

import torch
from torch.utils.data import DataLoader


def train_loop(
        model: torch.nn.Module,
        loss_fn: Callable,
        optimizer: torch.optim,
        device: torch.device,
        train_loader: DataLoader
):
    model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        prediction = model(images)
        loss = loss_fn(prediction, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"loss: {loss.item():>7f}  [{batch_idx * len(images):>5d}/{len(train_loader.dataset):>5d}]")


def test_model(
        model: torch.nn.Module,
        loss_fn: Callable,
        device: torch.device,
        test_dataloader: DataLoader
) -> float:
    model.eval()
    test_loss = 0
    correct = 0
    count_processed_images = 0

    with torch.no_grad():
        for (images, labels) in test_dataloader:
            images, labels = images.to(device), labels.to(device)

            prediction = model(images)
            test_loss += loss_fn(prediction, labels).item() # can be output for check loss
            correct += (prediction.round() == labels).type(torch.float).sum().item()
            count_processed_images += len(images)
    accuracy = correct / count_processed_images
    print(f"Test Accuracy: {(100 * accuracy):.2f}%")
    return accuracy


def train_model(
        model: torch.nn.Module,
        loss_fn: Callable,
        epochs: int,
        device: torch.device,
        optimizer: torch.optim,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader
):
    print("Using device:", device)

    # train model
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        train_loop(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            train_loader=train_dataloader
        )
        test_model(
            model=model,
            loss_fn=loss_fn,
            device=device,
            test_dataloader=valid_dataloader
        )



