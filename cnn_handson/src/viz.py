"""
Visualization helpers for the three tasks.

Includes:
    - sample-grid plots
    - class distribution
    - training history curves
    - learned conv filters
    - intermediate feature maps
    - confusion matrix
    - prediction grid
    - Grad-CAM heatmap overlay (interpretability)
    - segmentation mask overlay (pred vs. ground truth)
    - detection bbox overlay
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import data as _data
from .data import MRI_MEAN, MRI_STD, TUMOR_LABELS, NUM_CLASSES


# ---------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------

def _denorm(img: torch.Tensor) -> np.ndarray:
    arr = img.detach().cpu().squeeze().numpy()
    arr = arr * MRI_STD[0] + MRI_MEAN[0]
    return np.clip(arr, 0, 1)


# ---------------------------------------------------------------------
# Classification visualizations
# ---------------------------------------------------------------------

def plot_sample_grid(dataset, n: int = 16, ncols: int = 8, title: str = "MRI samples"):
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.6, nrows * 1.8))
    axes = np.array(axes).reshape(-1)
    idxs = np.random.choice(len(dataset), size=n, replace=False)
    for ax_i, idx in enumerate(idxs):
        img, label = dataset[idx]
        axes[ax_i].imshow(_denorm(img), cmap="gray")
        axes[ax_i].set_title(TUMOR_LABELS[label], fontsize=8)
        axes[ax_i].axis("off")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_class_distribution(dataset, title: str = "Class distribution"):
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        targets = [dataset.dataset.targets[i] for i in dataset.indices]
    else:
        targets = list(dataset.targets)
    counts = np.bincount(targets, minlength=NUM_CLASSES)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(TUMOR_LABELS, counts,
                  color=["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"])
    ax.set_ylabel("# samples")
    ax.set_title(title)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c, str(c),
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return fig


def plot_history(history: Dict[str, List[float]],
                 left=("train_loss", "val_loss"),
                 right=("train_acc", "val_acc"),
                 right_scale: float = 100.0,
                 right_label: str = "Accuracy (%)"):
    """Generic two-panel curve plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    epochs = range(1, len(history[left[0]]) + 1)

    for key, marker in zip(left, ["o-", "s-"]):
        if key in history:
            ax1.plot(epochs, history[key], marker, label=key)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Loss")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    for key, marker in zip(right, ["o-", "s-"]):
        if key in history:
            ax2.plot(epochs, [v * right_scale for v in history[key]], marker, label=key)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel(right_label); ax2.set_title(right_label)
    ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_conv_filters(model: nn.Module, layer_name: str = "conv1"):
    """Visualize learned conv filters (lecture p.88)."""
    layer = dict(model.features.named_children())[layer_name]
    w = layer.weight.detach().cpu()
    n_filters = w.shape[0]
    ncols = 8
    nrows = (n_filters + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.1, nrows * 1.2))
    axes = np.array(axes).reshape(-1)
    for i in range(n_filters):
        f = w[i, 0]
        f = (f - f.min()) / (f.max() - f.min() + 1e-8)
        axes[i].imshow(f.numpy(), cmap="viridis")
        axes[i].set_title(f"f{i}", fontsize=7)
        axes[i].axis("off")
    for j in range(n_filters, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Learned filters: {layer_name}")
    fig.tight_layout()
    return fig


def plot_feature_maps(model: nn.Module, image: torch.Tensor,
                      device: torch.device, max_channels: int = 8):
    """Visualize feature maps per stage for a SimpleCNN-like model."""
    model.eval().to(device)
    if image.dim() == 3:
        image = image.unsqueeze(0)
    acts = model.feature_maps(image.to(device))

    fig, axes = plt.subplots(len(acts), max_channels,
                             figsize=(max_channels * 1.2, len(acts) * 1.4))
    if len(acts) == 1:
        axes = np.array([axes])
    for row, (name, fmap) in enumerate(acts.items()):
        fmap = fmap[0].detach().cpu()
        c = min(max_channels, fmap.shape[0])
        for col in range(max_channels):
            ax = axes[row, col]
            if col < c:
                ax.imshow(fmap[col].numpy(), cmap="viridis")
                ax.set_title(f"{name}[{col}]", fontsize=7)
            ax.axis("off")
    fig.suptitle("Feature maps at each stage")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm: np.ndarray, class_names: Optional[List[str]] = None):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    if class_names is None:
        class_names = TUMOR_LABELS
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion Matrix")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)
    fig.tight_layout()
    return fig


@torch.no_grad()
def plot_predictions(model, dataset, n: int = 12, ncols: int = 6,
                     device: Optional[torch.device] = None):
    """Show predictions vs. ground truth for n random images."""
    device = device or torch.device("cpu")
    model.eval().to(device)
    idxs = np.random.choice(len(dataset), n, replace=False)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.8, nrows * 2.0))
    axes = np.array(axes).reshape(-1)
    for i, idx in enumerate(idxs):
        img, label = dataset[idx]
        logits = model(img.unsqueeze(0).to(device))
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(probs.argmax())
        axes[i].imshow(_denorm(img), cmap="gray")
        color = "green" if pred == label else "red"
        axes[i].set_title(
            f"T: {TUMOR_LABELS[label]}\nP: {TUMOR_LABELS[pred]} ({probs[pred]*100:.0f}%)",
            fontsize=8, color=color,
        )
        axes[i].axis("off")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Predictions (green=correct, red=wrong)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------
# Grad-CAM (interpretability for classification)
# ---------------------------------------------------------------------

class GradCAM:
    """Grad-CAM heatmaps for a target conv layer.

    Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep
    Networks via Gradient-based Localization", ICCV 2017.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._activations = None
        self._gradients = None
        target_layer.register_forward_hook(self._save_act)
        target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, _m, _i, output):
        self._activations = output.detach()

    def _save_grad(self, _m, _gi, grad_out):
        self._gradients = grad_out[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: Optional[int] = None):
        self.model.eval()
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())
        confidence = float(probs[0, class_idx].item())

        self.model.zero_grad()
        logits[0, class_idx].backward()

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self._activations).sum(dim=1).squeeze(0)
        cam = F.relu(cam).cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, class_idx, confidence


def plot_gradcam(model, dataset, n: int = 6,
                 device: Optional[torch.device] = None,
                 target_layer_name: str = "conv3"):
    """Grad-CAM heatmaps overlaid on MRI slices."""
    device = device or torch.device("cpu")
    model.eval().to(device)
    target_layer = dict(model.features.named_children())[target_layer_name]
    cam_tool = GradCAM(model, target_layer)

    idxs = np.random.choice(len(dataset), n, replace=False)
    fig, axes = plt.subplots(2, n, figsize=(n * 2.0, 4.2))
    for col, idx in enumerate(idxs):
        img, label = dataset[idx]
        x = img.unsqueeze(0).to(device).requires_grad_(True)
        cam, pred, conf = cam_tool(x)

        disp = _denorm(img)
        cam_t = torch.from_numpy(cam)[None, None].float()
        cam_resized = F.interpolate(cam_t, size=disp.shape,
                                    mode="bilinear", align_corners=False)[0, 0].numpy()

        axes[0, col].imshow(disp, cmap="gray")
        axes[0, col].set_title(f"T: {TUMOR_LABELS[label]}", fontsize=8)
        axes[0, col].axis("off")

        axes[1, col].imshow(disp, cmap="gray")
        axes[1, col].imshow(cam_resized, cmap="jet", alpha=0.45)
        color = "green" if pred == label else "red"
        axes[1, col].set_title(f"P: {TUMOR_LABELS[pred]} ({conf*100:.0f}%)",
                               fontsize=8, color=color)
        axes[1, col].axis("off")
    fig.suptitle(f"Grad-CAM @ {target_layer_name} — where the model looks")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------
# Segmentation visualizations
# ---------------------------------------------------------------------

def plot_segmentation_samples(dataset, n: int = 4, title: str = "LGG samples"):
    """Show image + ground-truth mask + overlay."""
    fig, axes = plt.subplots(n, 3, figsize=(8, n * 2.5))
    idxs = np.random.choice(len(dataset), n, replace=False)
    for r, idx in enumerate(idxs):
        img, mask = dataset[idx]
        disp = _denorm(img)
        m = mask.squeeze().numpy()

        axes[r, 0].imshow(disp, cmap="gray");          axes[r, 0].set_title("FLAIR slice", fontsize=8); axes[r, 0].axis("off")
        axes[r, 1].imshow(m, cmap="gray");             axes[r, 1].set_title("Tumor mask", fontsize=8); axes[r, 1].axis("off")
        axes[r, 2].imshow(disp, cmap="gray")
        axes[r, 2].imshow(np.ma.masked_where(m == 0, m), cmap="autumn", alpha=0.5)
        axes[r, 2].set_title("Overlay", fontsize=8);   axes[r, 2].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


@torch.no_grad()
def plot_segmentation_predictions(model, dataset, n: int = 5,
                                  device: Optional[torch.device] = None,
                                  threshold: float = 0.5):
    """For each image: input | GT mask | predicted mask | overlay (pred + GT)."""
    from .train import dice_score, iou_score

    device = device or torch.device("cpu")
    model.eval().to(device)

    # Prefer to show slices that actually have a tumor (more informative)
    pos_idxs = []
    for i in range(min(len(dataset), 200)):
        _, m = dataset[i]
        if m.sum() > 0:
            pos_idxs.append(i)
        if len(pos_idxs) >= n * 4:
            break
    chosen = list(np.random.choice(pos_idxs, n, replace=False)) if pos_idxs else list(range(n))

    fig, axes = plt.subplots(n, 4, figsize=(11, n * 2.5))
    if n == 1:
        axes = axes[None, :]
    for r, idx in enumerate(chosen):
        img, gt = dataset[idx]
        logits = model(img.unsqueeze(0).to(device))
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
        pred = (prob > threshold).astype(np.float32)

        dsc = (
            2.0 * (pred * gt.squeeze().numpy()).sum()
            / (pred.sum() + gt.squeeze().numpy().sum() + 1e-6)
        )

        disp = _denorm(img)
        m = gt.squeeze().numpy()

        axes[r, 0].imshow(disp, cmap="gray");          axes[r, 0].set_title("Input", fontsize=8); axes[r, 0].axis("off")
        axes[r, 1].imshow(m, cmap="gray");             axes[r, 1].set_title("GT mask", fontsize=8); axes[r, 1].axis("off")
        axes[r, 2].imshow(pred, cmap="gray");          axes[r, 2].set_title(f"Pred (Dice={dsc:.2f})", fontsize=8); axes[r, 2].axis("off")
        axes[r, 3].imshow(disp, cmap="gray")
        axes[r, 3].imshow(np.ma.masked_where(m == 0, m), cmap="winter", alpha=0.5)
        axes[r, 3].imshow(np.ma.masked_where(pred == 0, pred), cmap="autumn", alpha=0.5)
        axes[r, 3].set_title("Overlay (GT blue / Pred red)", fontsize=8); axes[r, 3].axis("off")
    fig.suptitle("Segmentation predictions")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------
# Detection visualizations
# ---------------------------------------------------------------------

def _draw_bbox(ax, bbox_norm, H, W, color, label: Optional[str] = None):
    """Draw a normalized (cx, cy, w, h) bbox on an axes."""
    cx, cy, w, h = bbox_norm
    x = (cx - w / 2) * W
    y = (cy - h / 2) * H
    rect = patches.Rectangle((x, y), w * W, h * H,
                             linewidth=2, edgecolor=color, facecolor="none")
    ax.add_patch(rect)
    if label:
        ax.text(x, max(0, y - 4), label, color=color, fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6, edgecolor="none"))


def plot_detection_samples(dataset, n: int = 6, title: str = "Detection labels"):
    """Show n samples with ground-truth bbox drawn over the slice."""
    ncols = min(n, 6)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.4))
    axes = np.array(axes).reshape(-1)
    idxs = np.random.choice(len(dataset), n, replace=False)
    for i, idx in enumerate(idxs):
        img, pres, bbox = dataset[idx]
        disp = _denorm(img)
        H, W = disp.shape
        axes[i].imshow(disp, cmap="gray")
        if pres.item() > 0.5:
            _draw_bbox(axes[i], bbox.tolist(), H, W, color="lime", label="tumor")
            axes[i].set_title("tumor", fontsize=8)
        else:
            axes[i].set_title("no tumor", fontsize=8)
        axes[i].axis("off")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


@torch.no_grad()
def plot_detection_predictions(model, dataset, n: int = 6,
                               device: Optional[torch.device] = None,
                               threshold: float = 0.5):
    """Show ground-truth bbox (green) vs predicted bbox (red) for n samples."""
    from .train import _bbox_iou
    device = device or torch.device("cpu")
    model.eval().to(device)

    # Prefer to draw on positive slices
    pos_idxs = [i for i in range(min(len(dataset), 200))
                if dataset[i][1].item() > 0.5]
    chosen = list(np.random.choice(pos_idxs, min(n, len(pos_idxs)), replace=False)) \
             if pos_idxs else list(range(n))
    if len(chosen) < n:
        extra = list(np.random.choice(len(dataset), n - len(chosen), replace=False))
        chosen = chosen + extra

    ncols = min(n, 6)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.4, nrows * 2.6))
    axes = np.array(axes).reshape(-1)
    for i, idx in enumerate(chosen):
        img, pres, bbox = dataset[idx]
        pres_logits, bbox_pred = model(img.unsqueeze(0).to(device))
        p = float(torch.sigmoid(pres_logits).item())
        pred_box = bbox_pred[0].cpu().tolist()

        disp = _denorm(img)
        H, W = disp.shape
        axes[i].imshow(disp, cmap="gray")

        # ground truth
        if pres.item() > 0.5:
            _draw_bbox(axes[i], bbox.tolist(), H, W, color="lime",
                       label="GT")
        # prediction
        if p > threshold:
            iou = float(_bbox_iou(torch.tensor([pred_box]), torch.tensor([bbox.tolist()]))[0])\
                if pres.item() > 0.5 else 0.0
            _draw_bbox(axes[i], pred_box, H, W, color="red",
                       label=f"P={p:.2f} IoU={iou:.2f}")
        axes[i].set_title(f"GT pres={int(pres.item())}, Pred pres={p:.2f}", fontsize=7)
        axes[i].axis("off")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Detection predictions (green=GT, red=Pred)")
    fig.tight_layout()
    return fig
