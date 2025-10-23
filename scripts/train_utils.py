import random
import math
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from scripts.data_augmentation import mixup_data, cutmix_data, mixup_cutmix_criterion


# ==============================================================
# 1. 训练单个 epoch
# ==============================================================
def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    单轮训练（Mixup/CutMix 在这里做；不做 scheduler.step()）
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Training]", leave=False)

    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        # 按 50% 概率选择 Mixup 或 CutMix（α=1）
        if random.random() < 0.5:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=1.0, device=device)
            outputs = model(inputs)
            loss = mixup_cutmix_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels, alpha=1.0, device=device)
            outputs = model(inputs)
            loss = mixup_cutmix_criterion(criterion, outputs, targets_a, targets_b, lam)

        loss.backward()
        optimizer.step()

        # 统计
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "Acc": f"{100.0 * correct / total:.2f}%"
        })

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


# ==============================================================
# 2. 验证
# ==============================================================
@torch.no_grad()
def validate_epoch(model, dataloader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Validate]", leave=False)

    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "Acc": f"{100.0 * correct / total:.2f}%"
        })

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


# ==============================================================
# 3. 保存模型
# ==============================================================
def save_checkpoint(model, optimizer, epoch, best_acc, path="checkpoint.pth"):
    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_acc": best_acc,
    }
    torch.save(state, path)


# ==============================================================
# 4. 优化器与调度器（仅 1 个：Warmup + Cosine）
# ==============================================================
def define_optimizer_and_scheduler(model, total_epochs=300, warmup_epochs=20, base_lr=0.1, weight_decay=5e-4):
    """
    - 优化器：SGD(momentum=0.9, nesterov=True, weight_decay=5e-4)
    - 学习率：前 warmup_epochs 线性增至 base_lr；其后 Cosine 到 1e-6
    - 只返回 1 个 scheduler（LambdaLR），由 main.py 每个 epoch 手动 step 一次
    """
    optimizer = optim.SGD(
        model.parameters(),
        lr=base_lr,
        momentum=0.9,
        nesterov=True,
        weight_decay=weight_decay,
    )

    eta_min = 1e-6

    def lr_lambda(current_epoch: int):
        if current_epoch < warmup_epochs:
            # 线性 Warmup：从极小值线性到 1.0
            return (current_epoch + 1) / float(max(1, warmup_epochs))
        # Cosine 部分
        progress = (current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return max(eta_min / base_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler
