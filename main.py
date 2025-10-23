#!/usr/bin/env python3

import argparse
import logging
import os
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets

from scripts.data_download import (
    download_and_extract_cifar10_data,
    download_and_extract_cifar100_data,
)
from scripts.data_augmentation import augment_dataset  # 统一入口
from scripts.model_architectures import create_model
from scripts.train_utils import (
    train_epoch,
    validate_epoch,
    save_checkpoint,
    define_optimizer_and_scheduler,
    compute_f1_scores
)
from scripts.evaluation_metrics import evaluate_model

# ===== Logging =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("cifar_pipeline.log")],
)
logger = logging.getLogger(__name__)


def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR-10/100 Strong Baseline")

    # Dataset / IO
    parser.add_argument("--dataset", type=str, choices=["cifar10", "cifar100"], default="cifar100")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results")

    # Hyperparams
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=300)
    parser.add_argument("--warmup_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    # Hardware
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=4)

    # Misc
    parser.add_argument("--seed", type=int, default=42)

    # Offline augmentation switch（保持原结构）
    parser.add_argument("--offline_aug", action="store_true", help="Run disk-based augmentation before training")
    parser.add_argument("--offline_aug_times", type=int, default=1, help="Augmentations per image when offline")
    return parser.parse_args()


# --------------------------
# 1) Collect data
# --------------------------
def collect_data(args):
    os.makedirs(os.path.join(args.data_dir, "raw"), exist_ok=True)
    if args.dataset == "cifar10":
        download_and_extract_cifar10_data(root_dir=os.path.join(args.data_dir, "raw"))
    else:
        download_and_extract_cifar100_data(root_dir=os.path.join(args.data_dir, "raw"))
    logger.info("Data prepared under data/raw/train and data/raw/test.")


# --------------------------
# 2) Augment data (optional, keep structure)
# --------------------------
def augment_data(args):
    """
    结构保留：如 --offline_aug 打开，则对 raw 做离线增强到 data/augmented；
    否则不做离线增强（在线增强会在 build_dataloaders 里通过返回的 transforms 完成）
    """
    if not args.offline_aug:
        logger.info("Skip offline augmentation (recommended). Use --offline_aug to enable.")
        return

    raw_dir = os.path.join(args.data_dir, "raw")
    aug_dir = os.path.join(args.data_dir, "augmented")
    os.makedirs(aug_dir, exist_ok=True)

    # 调用统一入口：离线增强（仅几何/颜色；保留原图）
    augment_dataset(
        input_dir=raw_dir,
        output_dir=aug_dir,
        augmentations_per_image=args.offline_aug_times,
        seed=args.seed,
        dataset=args.dataset,
        save_to_disk=True,
    )
    logger.info(f"Offline augmentation finished → {aug_dir}")




# --------------------------
# 在线/离线下，构建 DataLoader
# --------------------------
# --------------------------
# 在线/离线下，构建 DataLoader
# --------------------------
def build_dataloaders(args):
    """
    - 在线增强：augment_dataset(..., save_to_disk=False) -> (train_tf, test_tf)
    - 训练集内做 80/20 切分：train 子集用 train_tf；val 子集用 test_tf（评估增强）
    - test 集仅在最终评估使用
    """
    from torch.utils.data import Subset
    train_root = os.path.join(args.data_dir, "raw", "train")
    test_root  = os.path.join(args.data_dir, "raw", "test")

    # 在线增强（train_tf = 训练增强；test_tf = 评估增强）
    train_tf, test_tf = augment_dataset(
        input_dir=None, output_dir=None,
        augmentations_per_image=1, seed=args.seed,
        dataset=args.dataset, save_to_disk=False
    )

    # 同一物理样本，两套数据对象绑定不同 transform
    full_train_for_train = datasets.ImageFolder(root=train_root, transform=train_tf)
    full_train_for_val   = datasets.ImageFolder(root=train_root, transform=test_tf)

    n_total = len(full_train_for_train)
    n_val   = int(n_total * 0.2)
    n_train = n_total - n_val

    g = torch.Generator()
    g.manual_seed(args.seed)
    indices = torch.randperm(n_total, generator=g)
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:]

    train_set = Subset(full_train_for_train, train_idx)
    val_set   = Subset(full_train_for_val,   val_idx)

    # 测试集（仅最终评估用）
    test_set  = datasets.ImageFolder(root=test_root, transform=test_tf)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    # 返回 test_loader，供最终评估
    return train_loader, val_loader, test_loader, full_train_for_train.classes



# --------------------------
# 3) Build model
# --------------------------
def build_model(args):
    num_classes = 10 if args.dataset == "cifar10" else 100
    model = create_model(num_classes=num_classes, device=args.device)
    return model


# --------------------------
# 4) Train
# --------------------------
def train(args, model):
    # 原：train_loader, val_loader, class_names = build_dataloaders(args)
    train_loader, val_loader, test_loader, class_names = build_dataloaders(args)

    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = define_optimizer_and_scheduler(
        model,
        total_epochs=args.num_epochs,
        warmup_epochs=args.warmup_epochs,
        base_lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_acc = 0.0
    os.makedirs(os.path.join(args.output_dir, "models"), exist_ok=True)

    for epoch in range(1, args.num_epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, args.device, epoch)
        val_loss, val_acc, val_preds, val_labels = validate_epoch(model, val_loader, criterion, args.device, epoch)

        num_classes = 10 if args.dataset == "cifar10" else 100
        f1_macro, f1_micro, f1_weighted = compute_f1_scores(val_labels, val_preds, num_classes)

        scheduler.step()

        logging.info(
            f"Epoch {epoch:03d}/{args.num_epochs} | "
            f"Train Loss {tr_loss:.4f} Acc {tr_acc:.2f}% | "
            f"Val Loss {val_loss:.4f} Acc {val_acc:.2f}% | "
            f"F1(macro) {f1_macro:.4f} | F1(micro) {f1_micro:.4f} | F1(weighted) {f1_weighted:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch, best_acc,
                path=os.path.join(args.output_dir, "models", "best.pth")
            )
            logging.info(f"  ↳ New best Acc {best_acc:.2f}%, checkpoint saved.")

    # 返回 test_loader 以便最终评估
    return model, test_loader, class_names, best_acc



# --------------------------
# 5) Evaluate
# --------------------------
def evaluate(args, model, test_loader, class_names):
    criterion = nn.CrossEntropyLoss()
    from scripts.evaluation_metrics import evaluate_model
    loss, acc, all_preds, all_labels, all_probs = evaluate_model(model, test_loader, criterion, args.device)

    num_classes = len(class_names)
    f1_macro, f1_micro, f1_weighted = compute_f1_scores(all_labels, all_preds, num_classes)

    logging.info(
        f"[Test] Loss {loss:.4f} Acc {acc:.2f}% | "
        f"F1(macro) {f1_macro:.4f} | F1(micro) {f1_micro:.4f} | F1(weighted) {f1_weighted:.4f}"
    )



def main():
    args = parse_args()
    set_random_seeds(args.seed)

    logger.info("===== Strong CIFAR Baseline Start =====")
    for k, v in vars(args).items():
        logger.info(f"{k}: {v}")

    collect_data(args)
    augment_data(args)  # 可选（离线）
    model = build_model(args)
    model, test_loader, class_names, best_acc = train(args, model)
    evaluate(args, model, test_loader, class_names)


    logger.info(f"Done. Best Val/Test Acc: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
