from typing import Callable
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader


def train_loop(
        model: torch.nn.Module,
        loss_fn: Callable,
        optimizer: torch.optim,
        device: torch.device,
        train_loader: DataLoader,
        epoch_num: int
):
    model.train()
    progress_bar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Epoch {epoch_num + 1}",
        leave=False
    )
    for batch_idx, (images, labels) in progress_bar:
        images, labels = images.to(device), labels.to(device)

        prediction = model(images)
        loss = loss_fn(prediction, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar.set_postfix({
            "Loss": f"{loss.item():.4f}"
        })

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

    progress_bar = tqdm(
        test_dataloader,
        desc="Testing",
        leave=False
    )

    with torch.no_grad():
        for (images, labels) in progress_bar:
            images, labels = images.to(device), labels.to(device)

            prediction = model(images)
            test_loss += loss_fn(prediction, labels).item() # can be output for check loss
            correct += (prediction.round() == labels).type(torch.float).sum().item()
            count_processed_images += len(images)

            progress_bar.set_postfix({
                "Current Accuracy": f"{(correct / count_processed_images) * 100:.2f}%"
            })

    accuracy = correct / count_processed_images
    print(f"\nTest Accuracy: {(100 * accuracy):.2f}%")
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

    epoch_progress = tqdm(
        range(epochs),
        desc="Total Progress"
    )

    # train model
    for epoch in epoch_progress:
        print(f"Epoch {epoch + 1}")
        train_loop(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            train_loader=train_dataloader,
            epoch_num=epoch
        )
        test_model(
            model=model,
            loss_fn=loss_fn,
            device=device,
            test_dataloader=valid_dataloader
        )



