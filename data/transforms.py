import numpy as np
import random


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, label):
        for t in self.transforms:
            img, label = t(img, label)
        return img, label


class RandomFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, label):
        if random.random() < self.prob:
            axis = random.choice([1, 2])
            img = np.flip(img, axis=axis)
            label = np.flip(label, axis=axis)
        return img, label


class RandomRotate:
    def __call__(self, img, label):
        k = random.choice([0, 1, 2, 3])
        if k:
            img = np.rot90(img, k, axes=(1, 2))
            label = np.rot90(label, k, axes=(1, 2))
        return img, label


class RandomScale:
    def __init__(self, scale_range=(0.9, 1.1)):
        self.scale_range = scale_range

    def __call__(self, img, label):
        factor = random.uniform(*self.scale_range)
        h, w = img.shape[1], img.shape[2]
        new_h, new_w = int(h * factor), int(w * factor)

        from scipy.ndimage import zoom
        img = zoom(img, (1, new_h / h, new_w / w), order=1)
        label = zoom(label, (new_h / h, new_w / w), order=0)
        return img, label


def build_transform(train=True):
    if train:
        return Compose([RandomFlip(), RandomRotate()])
    return None
