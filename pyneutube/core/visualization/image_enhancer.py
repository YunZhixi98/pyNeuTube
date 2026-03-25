##########################################################
import numpy as np

def soft_standardize(img, k_sigma, epsilon=1e-8):
    img_min = max(img.mean() - img.std() * k_sigma, 0)
    img_max = min(img.mean() + img.std() * k_sigma, img.max())
    img = (img - img_min) / (img_max - img_min + epsilon) * 255
    img = np.clip(img, 0, 255)
    return img


