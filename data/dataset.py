import os
import torch
import numpy as np
from torch.utils.data import Dataset


class Brats2DDataset(Dataset):
    def __init__(self, data_dir, split="train", fold=0, num_folds=5, transform=None):
        self.transform = transform
        self.samples = []

        slices = np.load(os.path.join(data_dir, "slices.npy"))
        labels = np.load(os.path.join(data_dir, "labels.npy"))

        n = len(slices)
        idx = np.arange(n)
        np.random.RandomState(42).shuffle(idx)

        fold_size = n // num_folds
        start = fold * fold_size
        end = start + fold_size if fold < num_folds - 1 else n

        if split == "train":
            mask = np.ones(n, dtype=bool)
            mask[idx[start:end]] = False
            select = idx[mask]
        else:
            select = idx[start:end]

        self.samples = list(zip(slices[select], labels[select]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img, label = self.samples[index]
        img = img.astype(np.float32)
        label = label.astype(np.int64)

        if self.transform:
            img, label = self.transform(img, label)

        img = torch.from_numpy(img).unsqueeze(0)
        label = torch.from_numpy(label)
        return img, label


class Brats3DDataset(Dataset):
    def __init__(self, data_dir, split="train", fold=0, num_folds=5):
        import pickle
        self.samples = []

        files = sorted([f for f in os.listdir(data_dir) if f.endswith(".pkl")])
        n = len(files)
        idx = np.arange(n)
        np.random.RandomState(42).shuffle(idx)

        fold_size = n // num_folds
        start = fold * fold_size
        end = start + fold_size if fold < num_folds - 1 else n

        if split == "train":
            mask = np.ones(n, dtype=bool)
            mask[idx[start:end]] = False
            select = idx[mask]
        else:
            select = idx[start:end]

        self.files = [files[i] for i in select]
        self.data_dir = data_dir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        import pickle
        with open(os.path.join(self.data_dir, self.files[index]), "rb") as f:
            sample = pickle.load(f)
        img = torch.from_numpy(sample["image"].astype(np.float32))
        label = torch.from_numpy(sample["seg"].astype(np.int64))
        return img, label
