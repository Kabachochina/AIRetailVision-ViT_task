import torch
import numpy as np
import matplotlib.pyplot as plt


def show_image(image_tensor: torch.Tensor):
    """
    Принимает тензор изображения с форматом (C, H, W) и отображает его.
    Если значения не в диапазоне [0, 1], функция пытается их нормализовать.
    """
    image_array = image_tensor.numpy()

    # Если значения не в диапазоне [0, 1], нормализуем их
    if image_array.max() > 1:
        image_array = image_array / 255.0

    # Транспонируем массив с (C, H, W) в (H, W, C)
    image_array = np.transpose(image_array, (1, 2, 0))

    plt.imshow(image_array)
    plt.axis('off')
    plt.show()

