"""Training utilities for CIFAR-100 strong baseline experiments."""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


class TransformSubset(Dataset):
    """Dataset subset wrapper that applies a transform on-the-fly."""

    def __init__(self, base_dataset: datasets.ImageFolder, indices: Sequence[int], transform):
        self.dataset = base_dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:  # pragma: no cover - simple delegation
        return len(self.indices)

    def __getitem__(self, idx: int):  # pragma: no cover - simple delegation
        sample_idx = self.indices[idx]
        path, target = self.dataset.samples[sample_idx]
        image = self.dataset.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    @property
    def classes(self) -> List[str]:  # pragma: no cover - data container
        return self.dataset.classes


def load_transforms(train: bool = False) -> transforms.Compose:
    """Create torchvision transforms for CIFAR inputs."""

    normalize = transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)

    if train:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
                transforms.ToTensor(),
                normalize,
                transforms.RandomErasing(
                    p=0.25, value="random", scale=(0.02, 0.33), ratio=(0.3, 3.3)
                ),
            ]
        )

    return transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )


def load_data(data_dir: str, batch_size: int, val_split: float = 0.1, seed: int = 42):
    """Load CIFAR data from disk and create train/validation loaders."""

    base_dataset = datasets.ImageFolder(root=data_dir)
    num_samples = len(base_dataset)
    train_size = int((1.0 - val_split) * num_samples)
    val_size = num_samples - train_size

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_samples, generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = TransformSubset(
        base_dataset, train_indices, transform=load_transforms(train=True)
    )
    val_dataset = TransformSubset(
        base_dataset, val_indices, transform=load_transforms(train=False)
    )

    num_workers = min(8, os.cpu_count() or 2)
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    print(f"Dataset loaded from: {data_dir}")
    print(f"Total images: {num_samples}")
    print(f"Number of classes: {len(base_dataset.classes)}")
    print(f"Class names: {base_dataset.classes}")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    return train_loader, val_loader


@dataclass
class WarmupCosineScheduler:
    """Cosine schedule with linear warm-up that matches ReduceLROnPlateau API."""

    optimizer: optim.Optimizer
    warmup_epochs: int = 20
    total_epochs: int = 300
    max_lr: float = 0.1
    min_lr: float = 1e-6
    current_epoch: int = 0

    def __post_init__(self) -> None:
        self._set_lr(self._compute_lr(0))

    def _set_lr(self, lr: float) -> None:
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self.last_lr = lr

    def _compute_lr(self, epoch: int) -> float:
        if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            return self.max_lr * float(epoch + 1) / float(self.warmup_epochs)

        progress = (epoch - self.warmup_epochs) / float(
            max(1, self.total_epochs - self.warmup_epochs)
        )
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.max_lr - self.min_lr) * cosine

    def step(self, *_args, **_kwargs) -> float:  # matches ReduceLROnPlateau signature
        self.current_epoch += 1
        lr = self._compute_lr(self.current_epoch)
        self._set_lr(lr)
        return lr

    def state_dict(self) -> dict:
        return {
            "warmup_epochs": self.warmup_epochs,
            "total_epochs": self.total_epochs,
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
            "current_epoch": self.current_epoch,
            "last_lr": self.last_lr,
        }

    def load_state_dict(self, state: dict) -> None:
        self.warmup_epochs = state.get("warmup_epochs", self.warmup_epochs)
        self.total_epochs = state.get("total_epochs", self.total_epochs)
        self.max_lr = state.get("max_lr", self.max_lr)
        self.min_lr = state.get("min_lr", self.min_lr)
        self.current_epoch = state.get("current_epoch", self.current_epoch)
        last_lr = state.get("last_lr", self._compute_lr(self.current_epoch))
        self._set_lr(last_lr)


def define_loss_and_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    total_epochs: int = 300,
    warmup_epochs: int = 20,
    min_lr: float = 1e-6,
):
    """Create loss, optimizer and scheduler following the strong baseline recipe."""

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay,
        nesterov=True,
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=total_epochs,
        max_lr=lr,
        min_lr=min_lr,
    )

    return criterion, optimizer, scheduler


def _rand_bbox(size, lam: float, device: torch.device):
    """Generate CutMix bounding box."""

    _, _, height, width = size
    cut_ratio = math.sqrt(1.0 - lam)
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)

    cx = torch.randint(0, width, (1,), device=device).item()
    cy = torch.randint(0, height, (1,), device=device).item()

    x1 = int(np.clip(cx - cut_w // 2, 0, width))
    x2 = int(np.clip(cx + cut_w // 2, 0, width))
    y1 = int(np.clip(cy - cut_h // 2, 0, height))
    y2 = int(np.clip(cy + cut_h // 2, 0, height))

    return x1, y1, x2, y2


def _apply_mixup_cutmix(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mixup_alpha: float = 1.0,
    cutmix_alpha: float = 1.0,
    switch_prob: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply Mixup or CutMix augmentation to a batch."""

    if mixup_alpha <= 0 and cutmix_alpha <= 0:
        return inputs, targets, targets, 1.0

    use_cutmix = cutmix_alpha > 0 and random.random() < switch_prob
    if use_cutmix:
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        indices = torch.randperm(inputs.size(0), device=inputs.device)
        shuffled_inputs = inputs[indices]
        shuffled_targets = targets[indices]

        x1, y1, x2, y2 = _rand_bbox(inputs.size(), lam, device=inputs.device)
        inputs[:, :, y1:y2, x1:x2] = shuffled_inputs[:, :, y1:y2, x1:x2]

        lam_adjusted = 1.0 - ((x2 - x1) * (y2 - y1) / (inputs.size(-1) * inputs.size(-2)))
        return inputs, targets, shuffled_targets, lam_adjusted

    lam = np.random.beta(mixup_alpha, mixup_alpha)
    indices = torch.randperm(inputs.size(0), device=inputs.device)
    shuffled_inputs = inputs[indices]
    shuffled_targets = targets[indices]
    mixed_inputs = lam * inputs + (1.0 - lam) * shuffled_inputs
    return mixed_inputs, targets, shuffled_targets, lam


def _mixup_criterion(
    criterion: nn.Module,
    preds: torch.Tensor,
    targets_a: torch.Tensor,
    targets_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    return lam * criterion(preds, targets_a) + (1.0 - lam) * criterion(preds, targets_b)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    mixup_alpha: float = 1.0,
    cutmix_alpha: float = 1.0,
    mix_prob: float = 1.0,
    switch_prob: float = 0.5,
):
    """Train the model for one epoch with Mixup/CutMix regularisation."""

    model.train()
    running_loss = 0.0
    correct = 0.0
    total = 0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for inputs, labels in progress_bar:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if random.random() < mix_prob:
            mixed_inputs, targets_a, targets_b, lam = _apply_mixup_cutmix(
                inputs, labels, mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha, switch_prob=switch_prob
            )
        else:
            mixed_inputs, targets_a, targets_b, lam = inputs, labels, labels, 1.0

        optimizer.zero_grad(set_to_none=True)
        outputs = model(mixed_inputs)
        loss = _mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += lam * predicted.eq(targets_a).sum().item()
        correct += (1.0 - lam) * predicted.eq(targets_b).sum().item()

        progress_bar.set_postfix(
            {
                "Loss": f"{loss.item():.4f}",
                "Acc": f"{100.0 * correct / total:.2f}%",
                "LR": f"{optimizer.param_groups[0]['lr']:.5f}",
            }
        )

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    """Validate the model without stochastic regularisation."""

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation", leave=False)

        for inputs, labels in progress_bar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "Acc": f"{100.0 * correct / total:.2f}%",
                }
            )

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def save_checkpoint(state, filename):
    """Save model checkpoint."""

    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer=None, scheduler=None):
    """Load model checkpoint and restore state."""

    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Checkpoint file {filename} not found")

    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["state_dict"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint


def save_metrics(metrics: str, filename: str = "training_metrics.txt"):
    """Persist textual metrics to disk."""

    with open(filename, "w", encoding="utf-8") as f:
        f.write(metrics)
