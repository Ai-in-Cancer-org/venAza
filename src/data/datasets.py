"""Bag datasets for multi-instance learning (one bag = one patient)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ResizeKeepAspectPad:
    """Resize to fit the target box, pad the remainder (white by default)."""

    def __init__(self, size, fill=255):
        self.h, self.w = size
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        s = min(self.w / w, self.h / h)
        nw, nh = int(w * s), int(h * s)
        img = TF.resize(img, (nh, nw))
        pw, ph = self.w - nw, self.h - nh
        return TF.pad(img, [pw // 2, ph // 2, pw - pw // 2, ph - ph // 2], fill=self.fill)


def resnet_transforms(input_size, train: bool):
    ops = [T.Lambda(lambda im: im.convert("RGB")), ResizeKeepAspectPad(tuple(input_size))]
    if train:
        ops += [T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(5),
                T.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05, hue=0.02)]
    ops += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(ops)


class TileBagDataset(Dataset):
    """Image-tile bags for the end-to-end CNN branch.

    ``tiles`` must contain ``patient_id`` and ``patch_path`` (relative to ``root``)
    for QC-passed patches only.
    """

    def __init__(self, patients: pd.DataFrame, tiles: pd.DataFrame, root, transform,
                 max_tiles: int | None = None):
        self.root = Path(root)
        self.transform = transform
        self.max_tiles = max_tiles
        groups = dict(tuple(tiles.groupby("patient_id")))
        keep = [p for p in patients["patient_id"] if p in groups]
        missing = set(patients["patient_id"]) - set(keep)
        if missing:
            print(f"[TileBagDataset] {len(missing)} patient(s) without QC-passed tiles skipped")
        self.patients = patients.set_index("patient_id").loc[keep].reset_index()
        self.paths = {p: groups[p]["patch_path"].tolist() for p in keep}

    def __len__(self):
        return len(self.patients)

    def labels(self):
        return self.patients["label"].astype(int).tolist()

    def __getitem__(self, i):
        pid = self.patients.loc[i, "patient_id"]
        label = float(self.patients.loc[i, "label"])
        paths = self.paths[pid]
        if self.max_tiles and len(paths) > self.max_tiles:
            sel = np.random.choice(len(paths), self.max_tiles, replace=False)
            paths = [paths[j] for j in sorted(sel)]
        bag = torch.stack([self.transform(Image.open(self.root / p)) for p in paths])
        return bag, label, pid, paths


class FeatureBagDataset(Dataset):
    """Pre-computed tile embeddings ``<feature_dir>/<patient_id>.pt``."""

    def __init__(self, patients: pd.DataFrame, feature_dir, augment: bool = False,
                 noise_std: float = 0.01):
        self.feature_dir = Path(feature_dir)
        ok = [p for p in patients["patient_id"] if (self.feature_dir / f"{p}.pt").exists()]
        missing = set(patients["patient_id"]) - set(ok)
        if missing:
            print(f"[FeatureBagDataset] {len(missing)} patient(s) without features skipped")
        self.patients = patients[patients["patient_id"].isin(ok)].reset_index(drop=True)
        self.augment = augment
        self.noise_std = noise_std

    def __len__(self):
        return len(self.patients)

    def labels(self):
        return self.patients["label"].astype(int).tolist()

    def __getitem__(self, i):
        pid = self.patients.loc[i, "patient_id"]
        feats = torch.load(self.feature_dir / f"{pid}.pt", weights_only=False)["features"].float()
        if self.augment:  # bag-level augmentation: tile order + small feature noise
            feats = feats[torch.randperm(feats.shape[0])]
            feats = feats + torch.randn_like(feats) * self.noise_std
        return feats, float(self.patients.loc[i, "label"]), pid


def single_bag_collate(batch):
    """Batches of one patient (bags differ in size)."""
    assert len(batch) == 1
    return batch[0]


def balanced_sampler(labels) -> WeightedRandomSampler:
    """Patient-level class-balanced sampling with replacement."""
    labels = np.asarray(labels, dtype=int)
    counts = np.bincount(labels, minlength=2).astype(float)
    w = 1.0 / counts[labels]
    return WeightedRandomSampler(w.tolist(), num_samples=len(labels), replacement=True)
