from typing import Callable
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

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
        for batch in progress_bar:
            images = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=images)
            logits = outputs.logits

            test_loss += loss_fn(logits, labels).item() # can be output for check loss
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            count_processed_images += images.size(0)

            progress_bar.set_postfix({
                "Current Accuracy": f"{(correct / count_processed_images) * 100:.2f}%"
            })

    accuracy = correct / count_processed_images
    print(f"\nTest Accuracy: {(100 * accuracy):.2f}%")
    return accuracy



