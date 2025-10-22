import logging
import random
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

# ------------------------------------------------------------
# 日志配置
# ------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# CIFAR-Friendly 图像增强器
# ============================================================
class ImageAugmenter:
    """
    CIFAR 数据增强器：
      - RandomCrop(32, padding=4)
      - RandomHorizontalFlip(0.5)
      - AutoAugment(CIFAR10 Policy)
      - ToTensor + Normalize
      - RandomErasing(p=0.25)
      - 支持 Mixup / CutMix（batch级接口）
    """

    def __init__(
        self,
        augmentations_per_image: int = 3,
        seed: int = 42,
        save_original: bool = True,
        image_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
        dataset: str = "cifar100",  # 🔹 新增：根据数据集选择 mean/std
    ):
        self.augmentations_per_image = augmentations_per_image
        self.seed = seed
        self.save_original = save_original
        self.image_extensions = image_extensions
        self.dataset = dataset.lower()

        self._set_seed()

        # ----------------------------------------------------
        # 按数据集类型设置均值和标准差
        # ----------------------------------------------------
        if self.dataset == "cifar10":
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2470, 0.2435, 0.2616)
        else:  # 默认使用 CIFAR-100 参数
            mean = (0.5071, 0.4867, 0.4408)
            std = (0.2675, 0.2565, 0.2761)

        # ----------------------------------------------------
        # 训练阶段数据增强 pipeline
        # ----------------------------------------------------
        self.transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.25)
        ])

        # ----------------------------------------------------
        # 验证/测试阶段 pipeline（无随机性）
        # ----------------------------------------------------
        self.eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    # ========================================================
    # 内部辅助函数
    # ========================================================
    def _set_seed(self):
        """固定随机种子，保证复现性"""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _find_image_files(self, root: Path) -> List[Path]:
        """递归查找图片文件"""
        files = []
        for ext in self.image_extensions:
            files.extend(root.rglob(f"*{ext}"))
        return files

    # ========================================================
    # 图像级增强接口（保持原结构）
    # ========================================================
    def augment_image(self, image: Image.Image) -> torch.Tensor:
        """对单张图像执行增强，返回增强后 Tensor"""
        return self.transform(image)

    def process_directory(self, input_dir: str, output_dir: str) -> None:
        """
        批量增强 input_dir 下的图片并保存到 output_dir。
        兼容原 main.py 调用方式。
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        count = 0

        image_files = self._find_image_files(input_path)
        logger.info(f"Found {len(image_files)} images to augment.")

        for img_path in image_files:
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                logger.warning(f"Failed to load image {img_path}: {e}")
                continue

            rel_dir = img_path.parent.relative_to(input_path)
            target_dir = output_path / rel_dir
            target_dir.mkdir(parents=True, exist_ok=True)

            # 保存原图
            if self.save_original:
                orig_name = f"orig_{img_path.name}"
                image.save(target_dir / orig_name)

            # 生成增强样本
            for i in range(self.augmentations_per_image):
                augmented_tensor = self.transform(image)
                augmented_img = transforms.ToPILImage()(augmented_tensor)
                aug_name = f"aug_{i}_{img_path.name}"
                augmented_img.save(target_dir / aug_name)
                count += 1

        logger.info(f"Data augmentation completed. {count} images saved to {output_dir}")


# ============================================================
# Mixup / CutMix 工具函数（供 train_utils.py 调用）
# ============================================================
def mixup_data(x, y, alpha=1.0, device="cuda"):
    """批次级 Mixup"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size()[0]
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
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


def mixup_cutmix_criterion(criterion, preds, y_a, y_b, lam):
    """Mixup / CutMix 加权损失"""
    return lam * criterion(preds, y_a) + (1 - lam) * criterion(preds, y_b)


# ============================================================
# 兼容旧接口（供 main.py 调用）
# ============================================================
def augment_dataset(input_dir: str,
                    output_dir: str,
                    augmentations_per_image: int = 3,
                    seed: int = 42,
                    dataset: str = "cifar100") -> None:
    """
    兼容主程序接口的增强入口函数。
    """
    augmenter = ImageAugmenter(
        augmentations_per_image=augmentations_per_image,
        seed=seed,
        save_original=True,
        dataset=dataset
    )
    augmenter.process_directory(input_dir, output_dir)
