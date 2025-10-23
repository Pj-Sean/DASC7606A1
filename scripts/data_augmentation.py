import logging
import random
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

# ------------------------------------------------------------
# 日志
# ------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# CIFAR-Friendly 图像增强器（离线保存用：不包含 Normalize/RE）
# ============================================================
class DiskSafeAugmentor:
    """
    仅用于“写回磁盘”的增强器 —— 只做几何/颜色增强，避免把
    ToTensor/Normalize/RandomErasing 烘焙到图像文件里。
    """
    def __init__(
        self,
        augmentations_per_image: int = 1,
        seed: int = 42,
        save_original: bool = True,
        image_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
    ):
        self.augmentations_per_image = augmentations_per_image
        self.seed = seed
        self.save_original = save_original
        self.image_extensions = image_extensions
        self._set_seed()

        # 仅几何/颜色（与在线版保持风格一致，但不含 ToTensor/Normalize/RE）
        self.transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
        ])

    def _set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def _find_image_files(self, root: Path) -> List[Path]:
        files: List[Path] = []
        for ext in self.image_extensions:
            files.extend(root.rglob(f"*{ext}"))
        return files

    def augment_image(self, image: Image.Image) -> Image.Image:
        """
        离线增强：先用 torchvision 的 PIL 级别增强，再返回 PIL。
        """
        # transforms.Compose 里如果有 tensor op，需要 ToTensor；这里没有
        return self.transform(image)

    def process_directory(self, input_dir: str, output_dir: str) -> None:
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        image_files = self._find_image_files(input_path)
        logger.info(f"[Offline] Found {len(image_files)} images to augment → {output_path}")

        for img_path in image_files:
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                logger.warning(f"Failed to load image {img_path}: {e}")
                continue

            rel_dir = img_path.parent.relative_to(input_path)
            target_dir = output_path / rel_dir
            target_dir.mkdir(parents=True, exist_ok=True)

            # 保存原图（可选）
            if self.save_original:
                (target_dir / f"orig_{img_path.name}").write_bytes(img_path.read_bytes())

            # 生成增强图
            stem = img_path.stem
            suffix = img_path.suffix
            for i in range(self.augmentations_per_image):
                aug_img = self.augment_image(image.copy())
                aug_name = f"{stem}_aug{i+1}{suffix}"
                aug_img.save(target_dir / aug_name, quality=95)


# ============================================================
# Online 训练/验证的 transforms（返回给 DataLoader 用）
# ============================================================
def get_online_transforms(dataset: str = "cifar100"):
    dataset = dataset.lower()
    if dataset == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    else:
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.25),
    ])

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, test_tf


# ============================================================
# Mixup / CutMix（batch 级）—— 给 train_utils.train_epoch 使用
# ============================================================
def mixup_data(x, y, alpha=1.0, device="cuda"):
    """批次级 Mixup"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0, device="cuda"):
    """批次级 CutMix"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size, _, H, W = x.size()
    index = torch.randperm(batch_size).to(device)

    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))  # 修正 lam
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


def mixup_cutmix_criterion(criterion, preds, y_a, y_b, lam):
    """Mixup / CutMix 加权损失"""
    return lam * criterion(preds, y_a) + (1 - lam) * criterion(preds, y_b)


# ============================================================
# 统一入口：augment_dataset
# ============================================================
def augment_dataset(
    input_dir: Optional[str],
    output_dir: Optional[str],
    augmentations_per_image: int = 1,
    seed: int = 42,
    dataset: str = "cifar100",
    save_to_disk: bool = False,
):
    """
    统一增强入口：
      - save_to_disk=True：执行“离线增强”，把只含几何/颜色的增强图写入 output_dir。
      - save_to_disk=False：返回 (train_transform, test_transform) 供在线训练使用。
    """
    if save_to_disk:
        if input_dir is None or output_dir is None:
            raise ValueError("When save_to_disk=True, input_dir and output_dir must be provided.")
        augmenter = DiskSafeAugmentor(
            augmentations_per_image=augmentations_per_image,
            seed=seed,
            save_original=True,
        )
        augmenter.process_directory(input_dir, output_dir)
        return None, None  # 离线模式不需要返回 transforms
    else:
        return get_online_transforms(dataset=dataset)
