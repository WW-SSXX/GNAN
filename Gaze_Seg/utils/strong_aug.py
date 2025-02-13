import random

import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np


def RandomBrightnessContrast(img, brightness_limit=0.5, contrast_limit=0.5, p=1.0):
    output = torch.zeros_like(img)
    threshold = 0.5
    brightness_min = 0.2
    contrast_min = 0.1

    for i in range(output.shape[0]):
        img_min, img_max = torch.min(img[i]), torch.max(img[i])

        output[i] = (img[i] - img_min) / (img_max - img_min) * 255.0
        if random.random() < p:
            brightness = 1.0 + random.uniform(-brightness_limit, brightness_limit)
            if brightness > 1.0:
                brightness = max(1.0 + brightness_min, brightness)
            else:
                brightness = min(1.0 - brightness_min, brightness)
            output[i] = torch.clamp(output[i] * brightness, 0., 255.)

            contrast = 0.0 + random.uniform(-contrast_limit, contrast_limit)
            if contrast > 0.0:
                contrast = max(0.0 + contrast_min, contrast)
            else:
                contrast = min(0.0 - contrast_min, contrast)
            output[i] = torch.clamp(output[i] + (output[i] - threshold * 255.0) * contrast, 0., 255.)

        output[i] = output[i] / 255.0 * (img_max - img_min) + img_min
    return output


def add_gaussian_noise(tensor, device, mean=0.0, std=0.1):
    noise = torch.normal(mean=mean, std=std, size=tensor.size()).to(device)
    noisy_tensor = tensor + noise
    return noisy_tensor


def apply_strong_augmentations(tensor, device):
    tensor = RandomBrightnessContrast(tensor)
    tensor = add_gaussian_noise(tensor, device)
    return tensor

if __name__ == '__main__':
    from PIL import Image
    # 示例用法：
    # 假设 tensor 维度是 BxCx224x224
    img = Image.open('img.jpg')
    img = torch.tensor(np.array(img) / 255, dtype=torch.float32)
    print(img.shape)  # 输出的 tensor 维度应为 CxHxWx1
    img = img.permute([2, 0, 1]).unsqueeze(0)
    print(img.shape)  # 输出的 tensor 维度应为 CxHxWx1
    # tensor = torch.rand(4, 3, 224, 224)  # B=4, C=3, H=224, W=224
    augmented_tensor = apply_strong_augmentations(img) * 255
    print(augmented_tensor.shape)  # 输出的 tensor 维度应为 BxCx224x224
    augmented_img = np.array(augmented_tensor.squeeze(0).permute(1,2,0)).astype(np.uint8)
    augmented_img = Image.fromarray(augmented_img)
    augmented_img.save('augmented_img.jpg')

    # from PIL import Image
    # gaze = Image.open('gaze.png')
    # gaze = torch.tensor(np.array(gaze) / 255)
    # gaze = gaze.unsqueeze(0).unsqueeze(0)
    # gaze = torch.nn.functional.interpolate(gaze, size=(7, 7), mode='bilinear')
    # print(gaze)
    # t1, t2 = 0.2, 0.6
    # gaze = torch.where(gaze < t1, 0, gaze)
    # gaze = torch.where(gaze > t2, 2, gaze)
    # gaze = torch.where((gaze >= t1) & (gaze <= t2), 1, gaze) * 127
    # print(gaze)
    #
    #
    # gaze = np.array(gaze[0,0,:]).astype(np.uint8)
    # gaze = Image.fromarray(gaze)
    # gaze.save('gaze_resize.png')