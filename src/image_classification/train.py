import os
import glob
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import ResNet18_Weights, ResNet50_Weights
from torchvision.models import EfficientNet_B0_Weights, ConvNeXt_Tiny_Weights

from .config import IMAGENET_MEAN, IMAGENET_STD, TrainingConfig

MODEL_REGISTRY = {
    "resnet18": (models.resnet18, ResNet18_Weights.IMAGENET1K_V1, 512),
    "resnet50": (models.resnet50, ResNet50_Weights.IMAGENET1K_V1, 2048),
    "efficientnet_b0": (models.efficientnet_b0, EfficientNet_B0_Weights.IMAGENET1K_V1, 1280),
    "convnext_tiny": (models.convnext_tiny, ConvNeXt_Tiny_Weights.IMAGENET1K_V1, 768),
}


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transforms(
    mean: list[float],
    std: list[float],
    augment: bool = True,
) -> transforms.Compose:
    if augment:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_dataloaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    val_split: float,
    seed: int,
    mean: list[float],
    std: list[float],
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    train_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(data_dir, "train_dataset"),
        transform=get_transforms(mean, std, augment=True),
    )

    val_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(data_dir, "train_dataset"),
        transform=get_transforms(mean, std, augment=False),
    )

    test_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(data_dir, "test_dataset") if os.path.isdir(os.path.join(data_dir, "test_dataset")) else os.path.join(data_dir, "train_dataset"),
        transform=get_transforms(mean, std, augment=False),
    )

    train_size = int((1.0 - val_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    if os.path.isdir(os.path.join(data_dir, "test_dataset")):
        test_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(data_dir, "test_dataset"),
            transform=get_transforms(mean, std, augment=False),
        )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    classes = train_dataset.classes
    return train_loader, val_loader, test_loader, classes


def create_model(
    model_name: str,
    num_classes: int,
    compile_model: bool = True,
) -> nn.Module:
    if model_name not in MODEL_REGISTRY:
        msg = f"Unknown model {model_name}. Choose from {list(MODEL_REGISTRY)}"
        raise ValueError(msg)

    builder, weights, feat_dim = MODEL_REGISTRY[model_name]
    model = builder(weights=weights)

    if hasattr(model, "fc"):
        model.fc = nn.Linear(feat_dim, num_classes)
    elif hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Sequential):
            in_feats = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_feats, num_classes)
        else:
            model.classifier = nn.Linear(feat_dim, num_classes)

    if compile_model and torch.cuda.is_available():
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception:
            pass

    return model


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: float | None = None
        self.early_stop = False

    def step(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return True
        if score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        self.best_score = score
        self.counter = 0
        return True


class CheckpointSaver:
    def __init__(self, save_dir: str, max_to_keep: int = 3) -> None:
        self.save_dir = Path(save_dir)
        self.max_to_keep = max_to_keep
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        val_acc: float,
        is_best: bool = False,
    ) -> None:
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
        }
        epoch_path = self.save_dir / f"epoch_{epoch:03d}_acc_{val_acc:.2f}.pt"
        torch.save(state, epoch_path)

        if is_best:
            best_path = self.save_dir / "best_model.pt"
            torch.save(state, best_path)

        checkpoints = sorted(
            self.save_dir.glob("epoch_*.pt"),
            key=lambda p: float(p.stem.split("_acc_")[1]),
            reverse=True,
        )
        for ckpt in checkpoints[self.max_to_keep :]:
            ckpt.unlink(missing_ok=True)

    def load_best(self, model: nn.Module, optimizer: optim.Optimizer | None = None) -> int:
        best_path = self.save_dir / "best_model.pt"
        if not best_path.exists():
            return 0
        state = torch.load(best_path, weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        return state["epoch"]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    clip_max_norm: float = 1.0,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        total += labels.size(0)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
            optimizer.step()

        _, predicted = torch.max(outputs, 1)
        running_loss += loss.item()
        running_correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * running_correct / total
    return epoch_loss, epoch_acc


@torch.inference_mode()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        total += labels.size(0)

        outputs = model(images)
        loss = criterion(outputs, labels)

        _, predicted = torch.max(outputs, 1)
        running_loss += loss.item()
        running_correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * running_correct / total
    return epoch_loss, epoch_acc


def train(config: TrainingConfig) -> float:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(config.log_dir)

    train_loader, val_loader, test_loader, classes = get_dataloaders(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        val_split=config.val_split,
        seed=config.seed,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    )

    num_classes = len(classes)
    model = create_model(config.model_name, num_classes, compile_model=config.compile_model)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=1e-6,
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" and config.amp else None

    early_stopping = EarlyStopping(patience=config.patience)
    saver = CheckpointSaver(save_dir=config.save_dir, max_to_keep=config.max_keep_checkpoints)

    start_epoch = 0
    if config.resume is not None:
        checkpoint = torch.load(config.resume, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

    best_val_acc = 0.0

    for epoch in range(start_epoch, config.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Acc/val", val_acc, epoch)
        writer.add_scalar("LR", current_lr, epoch)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc

        saver.save(model, optimizer, epoch, val_acc, is_best=is_best)

        log = (
            f"Epoch {epoch+1:03d}/{config.epochs:03d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
            f"LR: {current_lr:.2e}"
        )
        if is_best:
            log += " *"
        print(log)

        early_stopping.step(val_acc)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f} Acc: {test_acc:.2f}%")
    print(f"Best Val Acc: {best_val_acc:.2f}%")

    writer.add_hparams(
        {
            "lr": config.lr,
            "batch_size": config.batch_size,
            "model": config.model_name,
        },
        {"hparam/test_acc": test_acc, "hparam/best_val_acc": best_val_acc},
    )
    writer.close()
    return best_val_acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train image classifier")
    parser.add_argument("--data-dir", default="Data", help="Dataset root directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--model", default="resnet18", choices=list(MODEL_REGISTRY), help="Model architecture")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP")
    parser.add_argument("--log-dir", default="runs/cats_vs_dogs", help="TensorBoard log dir")
    parser.add_argument("--save-dir", default="checkpoints", help="Checkpoint save dir")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from")
    args = parser.parse_args()

    config = TrainingConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        seed=args.seed,
        num_workers=args.num_workers,
        val_split=args.val_split,
        patience=args.patience,
        compile_model=not args.no_compile,
        amp=not args.no_amp,
        log_dir=args.log_dir,
        save_dir=args.save_dir,
        resume=args.resume,
        model_name=args.model,
    )
    train(config)


if __name__ == "__main__":
    main()
