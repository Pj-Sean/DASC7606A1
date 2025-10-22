import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import math
import random

from scripts.data_augmentation import mixup_data, cutmix_data, mixup_cutmix_criterion


# ==============================================================
# 1. 训练单个 epoch
# ==============================================================
def train_epoch(model, dataloader, criterion, optimizer, device, epoch, warmup_scheduler=None):
    """
    单轮训练
    Args:
        model: 神经网络模型
        dataloader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 运行设备 (cpu/cuda)
        epoch: 当前 epoch 编号
        warmup_scheduler: 可选的 warm-up 调度器
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Training]", leave=False)

    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        # ----------------------------------------------------------
        # 🔹 随机启用 Mixup 或 CutMix（概率各 50%）
        # ----------------------------------------------------------
        r = random.random()
        if r < 0.5:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=1.0, device=device)
            outputs = model(inputs)
            loss = mixup_cutmix_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels, alpha=1.0, device=device)
            outputs = model(inputs)
            loss = mixup_cutmix_criterion(criterion, outputs, targets_a, targets_b, lam)

        # ----------------------------------------------------------
        # 🔹 反向传播与参数更新
        # ----------------------------------------------------------
        loss.backward()
        optimizer.step()

        # 🔹 Warmup 调度（在前 20 epoch 内线性增长）
        if warmup_scheduler is not None:
            warmup_scheduler.step()

        # ----------------------------------------------------------
        # 🔹 统计指标
        # ----------------------------------------------------------
        running_loss += loss.item() * inputs.size(0)

        # Mixup / CutMix 不具备真实标签，因此准确率仅在普通阶段统计
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
# 2. 验证阶段
# ==============================================================
@torch.no_grad()
def validate_epoch(model, dataloader, criterion, device, epoch):
    """
    验证阶段：不使用 Mixup / CutMix
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Validation]", leave=False)

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
# 3. 保存模型 Checkpoint
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
# 4. 定义优化器与学习率调度器
# ==============================================================
def define_optimizer_and_scheduler(model, total_epochs=300, warmup_epochs=20, base_lr=0.1):
    """
    构建 SGD 优化器 + Warmup + Cosine Annealing 调度器
    """
    optimizer = optim.SGD(
        model.parameters(),
        lr=base_lr,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True
    )

    # Cosine Annealing: 让 lr 从 base_lr → lr_min=1e-6
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=(total_epochs - warmup_epochs),
        eta_min=1e-6
    )

    # Linear Warmup Scheduler
    def lr_lambda(current_step):
        if current_step < warmup_epochs:
            return float(current_step) / float(max(1, warmup_epochs))
        else:
            progress = (current_step - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, warmup_scheduler, cosine_scheduler
