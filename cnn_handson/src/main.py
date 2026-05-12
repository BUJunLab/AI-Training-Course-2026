"""
Command-line entry points for the three tasks.

Examples:
    python -m src.main classify  --model simplecnn --epochs 10
    python -m src.main classify  --model deepcnn   --epochs 10
    python -m src.main classify  --model mlp       --epochs 10
    python -m src.main segment   --epochs 10
    python -m src.main detect    --epochs 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .data import (
    NUM_CLASSES,
    get_classification_dataloaders,
    get_detection_dataloaders,
    get_segmentation_dataloaders,
    get_device,
)
from .models import (
    DeeperCNN,
    SimpleCNN,
    SimpleDetector,
    SimpleMLP,
    UNet,
    count_parameters,
)
from .train import (
    classification_report,
    confusion_matrix,
    evaluate_classifier,
    evaluate_detector,
    evaluate_segmenter,
    train_classifier,
    train_detector,
    train_segmenter,
)


def _build_classifier(name: str, img_size: int) -> torch.nn.Module:
    if name == "simplecnn":
        return SimpleCNN(in_channels=1, num_classes=NUM_CLASSES, img_size=img_size)
    if name == "deepcnn":
        return DeeperCNN(in_channels=1, num_classes=NUM_CLASSES)
    if name == "mlp":
        return SimpleMLP(input_shape=(1, img_size, img_size), num_classes=NUM_CLASSES)
    raise ValueError(f"Unknown classifier: {name}")


def _run_classify(args, device):
    train_loader, val_loader, test_loader, _ = get_classification_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size,
        img_size=args.img_size, augment=not args.no_augment, seed=args.seed,
    )
    model = _build_classifier(args.model, args.img_size)
    print(f"Model: {args.model}  params: {count_parameters(model):,}")

    history = train_classifier(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, device=device,
    )

    test_loss, test_acc = evaluate_classifier(
        model, test_loader, torch.nn.CrossEntropyLoss(), device
    )
    cm = confusion_matrix(model, test_loader, NUM_CLASSES, device)
    print(f"\n[TEST] loss={test_loss:.4f} acc={test_acc*100:.2f}%")
    print("Confusion matrix:\n", cm)
    return model, {"history": history, "cm": cm, "test_acc": test_acc}


def _run_segment(args, device):
    train_loader, val_loader, test_loader, _ = get_segmentation_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size,
        img_size=args.img_size, augment=not args.no_augment, seed=args.seed,
    )
    model = UNet(in_channels=1, out_channels=1, base=32)
    print(f"Model: UNet  params: {count_parameters(model):,}")

    history = train_segmenter(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, device=device,
    )

    test_loss, test_dice, test_iou = evaluate_segmenter(model, test_loader, device)
    print(f"\n[TEST] loss={test_loss:.4f} dice={test_dice:.4f} iou={test_iou:.4f}")
    return model, {"history": history, "test_dice": test_dice, "test_iou": test_iou}


def _run_detect(args, device):
    train_loader, val_loader, test_loader, _ = get_detection_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size,
        img_size=args.img_size, augment=not args.no_augment, seed=args.seed,
    )
    model = SimpleDetector(in_channels=1)
    print(f"Model: SimpleDetector  params: {count_parameters(model):,}")

    history = train_detector(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, device=device,
    )

    test_loss, test_pres_acc, test_miou = evaluate_detector(model, test_loader, device)
    print(
        f"\n[TEST] loss={test_loss:.4f} "
        f"presence_acc={test_pres_acc*100:.2f}% mean_iou(+)={test_miou:.4f}"
    )
    return model, {"history": history, "test_pres_acc": test_pres_acc, "test_miou": test_miou}


def parse_args():
    p = argparse.ArgumentParser(description="Brain MRI CNN hands-on (3 tasks)")
    sub = p.add_subparsers(dest="task", required=True)

    def common(sp):
        sp.add_argument("--epochs", type=int, default=10)
        sp.add_argument("--batch-size", type=int, default=64)
        sp.add_argument("--lr", type=float, default=1e-3)
        sp.add_argument("--weight-decay", type=float, default=1e-4)
        sp.add_argument("--img-size", type=int, default=128)
        sp.add_argument("--data-dir", type=str, default="./data")
        sp.add_argument("--out-dir", type=str, default="./outputs")
        sp.add_argument("--seed", type=int, default=42)
        sp.add_argument("--no-augment", action="store_true")

    sp_c = sub.add_parser("classify", help="4-class brain tumor MRI classification")
    sp_c.add_argument("--model", choices=["simplecnn", "deepcnn", "mlp"], default="simplecnn")
    common(sp_c)

    sp_s = sub.add_parser("segment", help="LGG MRI tumor segmentation (U-Net)")
    common(sp_s)

    sp_d = sub.add_parser("detect", help="LGG MRI single-object tumor detection")
    common(sp_d)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    if args.task == "classify":
        model, info = _run_classify(args, device)
    elif args.task == "segment":
        model, info = _run_segment(args, device)
    else:
        model, info = _run_detect(args, device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{args.task}.pt"
    torch.save(
        {"model_state": model.state_dict(), **info, "args": vars(args)},
        ckpt_path,
    )
    print(f"Saved -> {ckpt_path}")


if __name__ == "__main__":
    main()
