from typing import Optional

import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import torchvision
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification
from tqdm import tqdm

from src.attack import PGDAttack


def show_image(
        image_tensor: torch.Tensor,
        save_name: Optional[str] = None
):
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
    if save_name:
        pil_image = Image.fromarray(image_array)
        pil_image.save(save_name, format='PNG')

def show_adversarial_pgd_image(
        image_tensor: torch.Tensor,
        label: torch.Tensor,
        attack: PGDAttack
):
    """
    Проводит атаку на изображение и применяет метод show_image
    """
    adv_image = attack.perturb(
        images=image_tensor,
        labels=label
    )
    adv_image = adv_image.to(torch.device('cpu'))
    show_image(
        image_tensor=adv_image[0],
        save_name='adv_image'
    )

def attack_pgd_and_save_images(
        attacked_dir: str,
        output_dir: str,
        batch_size: int,
        attack: PGDAttack,
        device: torch.device,
        processor: ViTImageProcessor = None
) -> None:
    """
    :param attacked_dir: path to directory with original images
    :param output_dir: full path needed for save
    :param batch_size: size of batch for attack
    :param attack: PGDAttack object
    :param device: device for attack images (cpu or gpu)
    :param processor: optional
    :return: nothing
    """

    default_transform = torchvision.transforms.ToTensor()

    licensed = ['BABY_PRODUCTS', 'BEAUTY_HEALTH', 'ELECTRONICS', 'GROCERY', 'PET_SUPPLIES']
    unlicensed = ['CLOTHING_ACCESSORIES_JEWELLERY', 'HOBBY_ARTS_STATIONARY', 'HOME_KITCHEN_TOOLS', 'SPORTS_OUTDOOR']

    for category in os.listdir(attacked_dir):
        category_path = os.path.join(attacked_dir, category)
        if not os.path.isdir(category_path):
            continue

        if category in licensed:
            label = 1
        elif category in unlicensed:
            label = 0
        else:
            continue

        images_batch = []
        filenames_batch = []

        label =  torch.tensor([label], dtype=torch.long)

        filenames = [f for f in os.listdir(category_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        for filename in tqdm(filenames, desc=f"Processing {category}", unit="image"):
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(category_path, filename)
                image = Image.open(img_path).convert('RGB')
                if processor:
                    inputs = processor(
                        images=image,
                        return_tensors='pt'
                    )
                    pixel_values = inputs['pixel_values'].squeeze(0)  # убираем измерение батча
                else:
                    pixel_values = default_transform(image)
                images_batch.append(pixel_values)
                filenames_batch.append(filename)

                if len(images_batch) == batch_size or filename == filenames[-1]:

                    batch_tensor = torch.stack(images_batch).to(device) # объединяем картинки в один тензор
                    batch_labels = label.repeat(len(images_batch)).to(device) # у этих картинок одинаковые label т.к. они из одной директории

                    adv_imgs = attack.perturb(batch_tensor, batch_labels).cpu()

                    for i in range(len(images_batch)):
                        adv_img = adv_imgs[i]

                        image_array = adv_img.numpy()

                        # Транспонируем массив с (C, H, W) в (H, W, C)
                        image_array = np.transpose(image_array, (1, 2, 0))
                        image_array = np.clip(image_array, 0, 1)

                        image_array = (image_array * 255).astype(np.uint8)

                        os.makedirs(os.path.join(output_dir, category), exist_ok=True)
                        base_name = os.path.splitext(filenames_batch[i])[0]
                        save_path = os.path.join(output_dir, category, f"adversarial_{base_name}.png")
                        pil_image = Image.fromarray(image_array)
                        pil_image.save(save_path, format='PNG')

                    # Очищаем списки для следующего батча
                    images_batch = []
                    filenames_batch = []