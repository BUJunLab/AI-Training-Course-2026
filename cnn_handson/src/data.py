"""
Dataset utilities for the three CNN tasks in this hands-on.

Datasets used
-------------
1. Brain Tumor MRI Dataset (Masoud Nickparvar, Kaggle)
   - 4-class classification: glioma / meningioma / notumor / pituitary
   - ImageFolder layout: Training/<class>/*.jpg, Testing/<class>/*.jpg

2. LGG MRI Segmentation Dataset (Mateusz Buda, Kaggle)
   - Binary tumor masks on FLAIR brain MRI from TCGA-LGG patients
   - One folder per patient, each containing slice image (.tif) and mask (.tif)
   - Used for BOTH the segmentation example and the detection example
     (bounding boxes are derived from the masks on the fly).

All datasets are downloaded via kagglehub. Set up Kaggle credentials with
`~/.kaggle/kaggle.json` or via the kagglehub login prompt.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


# ---------------------------------------------------------------------
# Common constants
# ---------------------------------------------------------------------

DEFAULT_IMG_SIZE = 128

# Class labels for the 4-class brain tumor classification dataset
TUMOR_LABELS: List[str] = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES: int = len(TUMOR_LABELS)

# Approximate single-channel normalization stats for brain MRI
MRI_MEAN: Tuple[float] = (0.1738,)
MRI_STD: Tuple[float] = (0.1894,)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------
# Dataset downloaders (kagglehub)
# ---------------------------------------------------------------------

def _download_kaggle(slug: str, target: Path) -> Path:
    """Download a Kaggle dataset via kagglehub and symlink it to a stable path."""
    if target.exists():
        return target

    try:
        import kagglehub
    except ImportError as e:
        raise ImportError(
            "kagglehub is required for automatic download.\n"
            f"  pip install kagglehub\n"
            f"or manually place the {slug!r} dataset at {target}."
        ) from e

    print(f"[data] downloading {slug} via kagglehub ...")
    src = Path(kagglehub.dataset_download(slug))
    print(f"[data] kagglehub stored it at: {src}")

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        target.unlink()
    target.symlink_to(src, target_is_directory=True)
    print(f"[data] linked -> {target}")
    return target


def download_brain_tumor_classification(data_dir: str | os.PathLike = "./data") -> Path:
    """Download the 4-class brain tumor MRI dataset (Masoud Nickparvar)."""
    target = Path(data_dir) / "brain-tumor-mri"
    if (target / "Training").exists() and (target / "Testing").exists():
        return target
    return _download_kaggle("masoudnickparvar/brain-tumor-mri-dataset", target)


def download_lgg_segmentation(data_dir: str | os.PathLike = "./data") -> Path:
    """Download the LGG MRI segmentation dataset (Mateusz Buda)."""
    target = Path(data_dir) / "lgg-mri-segmentation"
    # The dataset extracts to a folder containing 'kaggle_3m/' inside.
    if (target / "kaggle_3m").exists() or any(
        target.glob("kaggle_3m/*")
    ):
        return target
    return _download_kaggle("mateuszbuda/lgg-mri-segmentation", target)


# ---------------------------------------------------------------------
# 1. Classification dataloaders
# ---------------------------------------------------------------------

def get_classification_dataloaders(
    data_dir: str | os.PathLike = "./data",
    batch_size: int = 64,
    num_workers: int = 0,
    img_size: int = DEFAULT_IMG_SIZE,
    val_split: float = 0.1,
    augment: bool = True,
    seed: int = 42,
):
    """Return train / val / test loaders for the 4-class classification dataset."""
    root = download_brain_tumor_classification(data_dir)
    train_dir, test_dir = root / "Training", root / "Testing"

    train_tf = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
    ]
    if augment:
        train_tf += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
        ]
    train_tf += [transforms.ToTensor(), transforms.Normalize(MRI_MEAN, MRI_STD)]
    train_tf = transforms.Compose(train_tf)

    test_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MRI_MEAN, MRI_STD),
    ])

    full_train = datasets.ImageFolder(str(train_dir), transform=train_tf)
    test_ds = datasets.ImageFolder(str(test_dir), transform=test_tf)
    assert full_train.classes == TUMOR_LABELS, f"Unexpected classes: {full_train.classes}"

    n_val = int(len(full_train) * val_split)
    n_train = len(full_train) - n_val
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_train, [n_train, n_val], generator=gen)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        (train_ds, val_ds, test_ds),
    )


# ---------------------------------------------------------------------
# 2. Segmentation dataset (LGG MRI)
# ---------------------------------------------------------------------

def _list_lgg_pairs(root: Path) -> List[Tuple[Path, Path]]:
    """Walk the LGG folder and return (image_path, mask_path) pairs.

    The LGG dataset stores each patient's slices like:
        TCGA_xx_xxxx_<slice>.tif       # image (3-channel RGB)
        TCGA_xx_xxxx_<slice>_mask.tif  # mask  (single channel, white=tumor)
    """
    # The dataset structure: root/kaggle_3m/<patient>/...
    base = root / "kaggle_3m"
    if not base.exists():
        # Fallback: maybe the dataset was placed directly at root
        base = root
    pairs = []
    for patient_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        for mask in sorted(patient_dir.glob("*_mask.tif")):
            img = mask.with_name(mask.name.replace("_mask.tif", ".tif"))
            if img.exists():
                pairs.append((img, mask))
    return pairs


def _split_by_patient(pairs, val_frac=0.15, test_frac=0.15, seed=42):
    """Split (image, mask) pairs by *patient* to prevent leakage.

    Important in medical imaging: slices from the same subject must not appear
    in both train and test, otherwise accuracy is wildly over-estimated.
    """
    patients = sorted({p[0].parent.name for p in pairs})
    rng = np.random.RandomState(seed)
    rng.shuffle(patients)

    n = len(patients)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    test_pat = set(patients[:n_test])
    val_pat = set(patients[n_test:n_test + n_val])

    train, val, test = [], [], []
    for img, mask in pairs:
        subj = img.parent.name
        if subj in test_pat:
            test.append((img, mask))
        elif subj in val_pat:
            val.append((img, mask))
        else:
            train.append((img, mask))
    return train, val, test


class LGGSegmentationDataset(Dataset):
    """LGG MRI segmentation dataset (FLAIR slice + binary tumor mask)."""

    def __init__(
        self,
        pairs: List[Tuple[Path, Path]],
        img_size: int = DEFAULT_IMG_SIZE,
        augment: bool = False,
    ):
        self.pairs = pairs
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def _load(self, idx: int):
        img_path, mask_path = self.pairs[idx]
        img = Image.open(img_path).convert("L")     # grayscale
        mask = Image.open(mask_path).convert("L")

        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        img = np.asarray(img, dtype=np.float32) / 255.0
        mask = (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)
        return img, mask

    def __getitem__(self, idx: int):
        img, mask = self._load(idx)

        # Apply matched (image+mask) augmentations
        if self.augment:
            if np.random.rand() < 0.5:
                img = img[:, ::-1].copy()
                mask = mask[:, ::-1].copy()
            if np.random.rand() < 0.5:
                img = img[::-1, :].copy()
                mask = mask[::-1, :].copy()

        # Normalize image (mask stays in {0, 1})
        img = (img - MRI_MEAN[0]) / MRI_STD[0]

        img_t = torch.from_numpy(img).unsqueeze(0).float()    # [1, H, W]
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()  # [1, H, W]
        return img_t, mask_t


def get_segmentation_dataloaders(
    data_dir: str | os.PathLike = "./data",
    batch_size: int = 16,
    num_workers: int = 0,
    img_size: int = DEFAULT_IMG_SIZE,
    augment: bool = True,
    seed: int = 42,
):
    """Return train / val / test loaders for LGG segmentation, patient-level split."""
    root = download_lgg_segmentation(data_dir)
    pairs = _list_lgg_pairs(root)
    if not pairs:
        raise RuntimeError(f"No (image, mask) pairs found under {root}.")

    train_pairs, val_pairs, test_pairs = _split_by_patient(pairs, seed=seed)
    print(
        f"[seg] patients -> "
        f"train: {len({p[0].parent.name for p in train_pairs})}, "
        f"val: {len({p[0].parent.name for p in val_pairs})}, "
        f"test: {len({p[0].parent.name for p in test_pairs})}"
    )
    print(
        f"[seg] slices   -> train: {len(train_pairs)}, "
        f"val: {len(val_pairs)}, test: {len(test_pairs)}"
    )

    train_ds = LGGSegmentationDataset(train_pairs, img_size, augment=augment)
    val_ds = LGGSegmentationDataset(val_pairs, img_size, augment=False)
    test_ds = LGGSegmentationDataset(test_pairs, img_size, augment=False)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        (train_ds, val_ds, test_ds),
    )


# ---------------------------------------------------------------------
# 3. Detection dataset (LGG bounding boxes derived from masks)
# ---------------------------------------------------------------------

def _mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    """Compute bbox (cx, cy, w, h) normalized to [0, 1] from a binary mask.

    Returns None if the mask has no foreground pixels.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    H, W = mask.shape
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    cx = (x_min + x_max) / 2.0 / W
    cy = (y_min + y_max) / 2.0 / H
    w = (x_max - x_min + 1) / W
    h = (y_max - y_min + 1) / H
    return float(cx), float(cy), float(w), float(h)


class LGGDetectionDataset(Dataset):
    """Single-object detection dataset built on top of LGG segmentation masks.

    Each item returns:
        image:    Tensor [1, H, W]
        presence: 0 or 1 (does this slice contain a tumor?)
        bbox:     Tensor [4] = (cx, cy, w, h), all in [0, 1]
                  (filled with zeros when presence = 0)
    """

    def __init__(
        self,
        pairs: List[Tuple[Path, Path]],
        img_size: int = DEFAULT_IMG_SIZE,
        augment: bool = False,
    ):
        self.seg = LGGSegmentationDataset(pairs, img_size=img_size, augment=augment)

    def __len__(self):
        return len(self.seg)

    def __getitem__(self, idx: int):
        img_t, mask_t = self.seg[idx]
        mask = mask_t.squeeze(0).numpy()
        bbox = _mask_to_bbox(mask)

        if bbox is None:
            presence = torch.tensor(0.0)
            bbox_t = torch.zeros(4)
        else:
            presence = torch.tensor(1.0)
            bbox_t = torch.tensor(bbox, dtype=torch.float32)

        return img_t, presence, bbox_t


def get_detection_dataloaders(
    data_dir: str | os.PathLike = "./data",
    batch_size: int = 32,
    num_workers: int = 0,
    img_size: int = DEFAULT_IMG_SIZE,
    augment: bool = True,
    seed: int = 42,
):
    """Return train / val / test loaders for single-object tumor detection."""
    root = download_lgg_segmentation(data_dir)
    pairs = _list_lgg_pairs(root)
    if not pairs:
        raise RuntimeError(f"No (image, mask) pairs found under {root}.")

    train_pairs, val_pairs, test_pairs = _split_by_patient(pairs, seed=seed)

    train_ds = LGGDetectionDataset(train_pairs, img_size, augment=augment)
    val_ds = LGGDetectionDataset(val_pairs, img_size, augment=False)
    test_ds = LGGDetectionDataset(test_pairs, img_size, augment=False)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        (train_ds, val_ds, test_ds),
    )


def denormalize_for_display(img: torch.Tensor) -> np.ndarray:
    """Undo MRI mean/std normalization for plotting."""
    arr = img.detach().cpu().squeeze().numpy()
    arr = arr * MRI_STD[0] + MRI_MEAN[0]
    return np.clip(arr, 0, 1)
