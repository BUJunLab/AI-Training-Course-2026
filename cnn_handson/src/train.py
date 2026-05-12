"""
Training and evaluation loops for the three task types.

The classification loop mirrors the lecture pipeline (Training_Course.pdf
p.70-76): zero_grad -> forward -> loss -> backward -> step. The segmentation
and detection loops follow the same structure but with different losses and
metrics.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# =====================================================================
# 1. Classification
# =====================================================================

def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device | None = None,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """CrossEntropy + Adam training loop. Returns history dict."""
    device = device or torch.device("cpu")
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    for epoch in range(1, epochs + 1):
        model.train()
        tot_loss, tot_correct, tot = 0.0, 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optim.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            optim.step()
            tot_loss += loss.item() * X.size(0)
            tot_correct += (logits.argmax(1) == y).sum().item()
            tot += X.size(0)
        tr_loss, tr_acc = tot_loss / tot, tot_correct / tot

        vl_loss, vl_acc = evaluate_classifier(model, val_loader, loss_fn, device)

        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(vl_loss)
        hist["train_acc"].append(tr_acc)
        hist["val_acc"].append(vl_acc)

        if verbose:
            print(
                f"Epoch [{epoch:2d}/{epochs}] "
                f"train_loss={tr_loss:.4f} train_acc={tr_acc*100:5.2f}% | "
                f"val_loss={vl_loss:.4f} val_acc={vl_acc*100:5.2f}%"
            )
    return hist


@torch.no_grad()
def evaluate_classifier(model, loader, loss_fn, device) -> Tuple[float, float]:
    model.eval()
    tot_loss, tot_correct, tot = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = loss_fn(logits, y)
        tot_loss += loss.item() * X.size(0)
        tot_correct += (logits.argmax(1) == y).sum().item()
        tot += X.size(0)
    return tot_loss / tot, tot_correct / tot


@torch.no_grad()
def confusion_matrix(model, loader, num_classes, device) -> np.ndarray:
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        preds = model(X).argmax(1)
        for t, p in zip(y.cpu().numpy(), preds.cpu().numpy()):
            cm[t, p] += 1
    return cm


def classification_report(cm: np.ndarray, class_names: List[str]) -> Dict[str, Dict[str, float]]:
    """Per-class precision / sensitivity / specificity / F1 (clinical metrics)."""
    report = {}
    total = cm.sum()
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        report[name] = {
            "precision": precision,
            "sensitivity": recall,
            "specificity": specificity,
            "f1": f1,
            "support": int(cm[i, :].sum()),
        }
    return report


# =====================================================================
# 2. Segmentation
# =====================================================================

def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss for binary segmentation. logits: [B, 1, H, W], target same."""
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    target = target.flatten(1)
    inter = (probs * target).sum(dim=1)
    denom = probs.sum(dim=1) + target.sum(dim=1)
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def bce_dice_loss(logits, target, bce_weight: float = 0.5):
    """Combined BCE + Dice loss; common for medical segmentation."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dl = dice_loss(logits, target)
    return bce_weight * bce + (1.0 - bce_weight) * dl


@torch.no_grad()
def dice_score(logits, target, threshold: float = 0.5, eps: float = 1e-6) -> float:
    """Hard Dice (a.k.a. F1 for foreground) over a batch."""
    pred = (torch.sigmoid(logits) > threshold).float()
    pred = pred.flatten(1)
    target = target.flatten(1)
    inter = (pred * target).sum(dim=1)
    denom = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2 * inter + eps) / (denom + eps)
    return dice.mean().item()


@torch.no_grad()
def iou_score(logits, target, threshold: float = 0.5, eps: float = 1e-6) -> float:
    pred = (torch.sigmoid(logits) > threshold).float()
    pred = pred.flatten(1)
    target = target.flatten(1)
    inter = (pred * target).sum(dim=1)
    union = ((pred + target) >= 1.0).float().sum(dim=1)
    iou = (inter + eps) / (union + eps)
    return iou.mean().item()


def train_segmenter(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: torch.device | None = None,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """Train a segmentation model with BCE+Dice loss."""
    device = device or torch.device("cpu")
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    hist = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": [], "val_iou": []}
    for epoch in range(1, epochs + 1):
        model.train()
        tot_loss, tot_dice, tot = 0.0, 0.0, 0
        for X, M in train_loader:
            X, M = X.to(device), M.to(device)
            optim.zero_grad()
            logits = model(X)
            loss = bce_dice_loss(logits, M)
            loss.backward()
            optim.step()
            bs = X.size(0)
            tot_loss += loss.item() * bs
            tot_dice += dice_score(logits, M) * bs
            tot += bs
        tr_loss = tot_loss / tot
        tr_dice = tot_dice / tot

        vl_loss, vl_dice, vl_iou = evaluate_segmenter(model, val_loader, device)

        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(vl_loss)
        hist["train_dice"].append(tr_dice)
        hist["val_dice"].append(vl_dice)
        hist["val_iou"].append(vl_iou)

        if verbose:
            print(
                f"Epoch [{epoch:2d}/{epochs}] "
                f"train_loss={tr_loss:.4f} train_dice={tr_dice:.4f} | "
                f"val_loss={vl_loss:.4f} val_dice={vl_dice:.4f} val_iou={vl_iou:.4f}"
            )
    return hist


@torch.no_grad()
def evaluate_segmenter(model, loader, device) -> Tuple[float, float, float]:
    model.eval()
    tot_loss, tot_dice, tot_iou, tot = 0.0, 0.0, 0.0, 0
    for X, M in loader:
        X, M = X.to(device), M.to(device)
        logits = model(X)
        loss = bce_dice_loss(logits, M)
        bs = X.size(0)
        tot_loss += loss.item() * bs
        tot_dice += dice_score(logits, M) * bs
        tot_iou += iou_score(logits, M) * bs
        tot += bs
    return tot_loss / tot, tot_dice / tot, tot_iou / tot


# =====================================================================
# 3. Detection (single-object: presence + bbox regression)
# =====================================================================

def _bbox_iou(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    """IoU between two batches of (cx, cy, w, h) boxes."""
    def to_xyxy(b):
        cx, cy, w, h = b.unbind(-1)
        return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)

    a, b = to_xyxy(box_a), to_xyxy(box_b)
    inter_l = torch.maximum(a[..., 0], b[..., 0])
    inter_t = torch.maximum(a[..., 1], b[..., 1])
    inter_r = torch.minimum(a[..., 2], b[..., 2])
    inter_b = torch.minimum(a[..., 3], b[..., 3])
    inter = (inter_r - inter_l).clamp(min=0) * (inter_b - inter_t).clamp(min=0)

    area_a = (a[..., 2] - a[..., 0]).clamp(min=0) * (a[..., 3] - a[..., 1]).clamp(min=0)
    area_b = (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)
    union = area_a + area_b - inter
    return inter / union.clamp(min=1e-8)


def detection_loss(
    presence_logits: torch.Tensor,
    bbox_pred: torch.Tensor,
    presence_target: torch.Tensor,
    bbox_target: torch.Tensor,
    bbox_weight: float = 5.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """BCE presence loss + Smooth L1 bbox loss (only on positive samples)."""
    pres_loss = F.binary_cross_entropy_with_logits(presence_logits, presence_target)

    pos_mask = presence_target > 0.5
    if pos_mask.sum() > 0:
        bbox_l = F.smooth_l1_loss(bbox_pred[pos_mask], bbox_target[pos_mask])
    else:
        bbox_l = torch.tensor(0.0, device=presence_logits.device)

    total = pres_loss + bbox_weight * bbox_l
    return total, pres_loss.detach(), bbox_l.detach()


def train_detector(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    bbox_weight: float = 5.0,
    device: torch.device | None = None,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """Train the single-object detector."""
    device = device or torch.device("cpu")
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    hist = {
        "train_loss": [], "val_loss": [],
        "val_presence_acc": [], "val_mean_iou_pos": [],
    }
    for epoch in range(1, epochs + 1):
        model.train()
        tot_loss, tot = 0.0, 0
        for X, pres, bbox in train_loader:
            X = X.to(device); pres = pres.to(device); bbox = bbox.to(device)
            optim.zero_grad()
            pres_logits, bbox_pred = model(X)
            loss, _, _ = detection_loss(pres_logits, bbox_pred, pres, bbox, bbox_weight)
            loss.backward()
            optim.step()
            bs = X.size(0)
            tot_loss += loss.item() * bs
            tot += bs
        tr_loss = tot_loss / tot

        vl_loss, vl_pres_acc, vl_miou = evaluate_detector(model, val_loader, device, bbox_weight)

        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(vl_loss)
        hist["val_presence_acc"].append(vl_pres_acc)
        hist["val_mean_iou_pos"].append(vl_miou)

        if verbose:
            print(
                f"Epoch [{epoch:2d}/{epochs}] "
                f"train_loss={tr_loss:.4f} | "
                f"val_loss={vl_loss:.4f} "
                f"val_presence_acc={vl_pres_acc*100:5.2f}% "
                f"val_mean_iou(+)={vl_miou:.4f}"
            )
    return hist


@torch.no_grad()
def evaluate_detector(model, loader, device, bbox_weight: float = 5.0):
    model.eval()
    tot_loss, tot = 0.0, 0
    correct, total_p = 0, 0
    iou_sum, iou_n = 0.0, 0
    for X, pres, bbox in loader:
        X = X.to(device); pres = pres.to(device); bbox = bbox.to(device)
        pres_logits, bbox_pred = model(X)
        loss, _, _ = detection_loss(pres_logits, bbox_pred, pres, bbox, bbox_weight)
        bs = X.size(0)
        tot_loss += loss.item() * bs
        tot += bs

        pres_hat = (torch.sigmoid(pres_logits) > 0.5).float()
        correct += (pres_hat == pres).sum().item()
        total_p += bs

        pos = pres > 0.5
        if pos.sum() > 0:
            iou = _bbox_iou(bbox_pred[pos], bbox[pos])
            iou_sum += iou.sum().item()
            iou_n += int(pos.sum().item())

    return (
        tot_loss / tot,
        correct / total_p,
        (iou_sum / iou_n) if iou_n > 0 else 0.0,
    )
