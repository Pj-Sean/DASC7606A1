import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------
# 基本瓶颈块（Bottleneck Block）
# -----------------------------------------------------
class Bottleneck(nn.Module):
    expansion = 4  # 每个block输出通道扩张4倍

    def __init__(self, in_channels, mid_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # 1×1 降维
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        # 3×3 卷积（stride控制空间下采样）
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        # 1×1 升维
        self.conv3 = nn.Conv2d(mid_channels, mid_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # shortcut调整
        self.stride = stride

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += identity
        out = self.relu(out)
        return out

# -----------------------------------------------------
# ResNet 主体结构
# -----------------------------------------------------
class ResNet(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet, self).__init__()
        # ---- 前置层（CIFAR友好） ----
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),  # 3×3/1
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # ---- 四个 Residual Layer ----
        self.layer1 = self._make_layer(64, 64, blocks=3, stride=1)
        self.layer2 = self._make_layer(256, 128, blocks=4, stride=2)
        self.layer3 = self._make_layer(512, 256, blocks=6, stride=2)
        # 最后一层E5版本不下采样 (stride=1)
        self.layer4 = self._make_layer(1024, 512, blocks=3, stride=1)

        # ---- 分类头 ----
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512 * Bottleneck.expansion, num_classes)

        # ---- 参数初始化 ----
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # 构造一个 stage（多个Bottleneck堆叠）
    def _make_layer(self, in_channels, mid_channels, blocks, stride):
        downsample = None
        # 下采样（ResNet-D风格：AvgPool+Conv1×1+BN）
        if stride != 1 or in_channels != mid_channels * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True) if stride != 1 else nn.Identity(),
                nn.Conv2d(in_channels, mid_channels * Bottleneck.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(mid_channels * Bottleneck.expansion),
            )
        layers = []
        layers.append(Bottleneck(in_channels, mid_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(Bottleneck(mid_channels * Bottleneck.expansion, mid_channels))
        return nn.Sequential(*layers)

    # 前向传播
    def forward(self, x):
        x = self.pre(x)             # 前置层
        x = self.layer1(x)          # Residual Layer1
        x = self.layer2(x)          # Residual Layer2
        x = self.layer3(x)          # Residual Layer3
        x = self.layer4(x)          # Residual Layer4
        x = self.avgpool(x)         # 全局平均池化
        x = torch.flatten(x, 1)
        x = self.classifier(x)      # 全连接分类
        return x

# -----------------------------------------------------
# 创建接口（供 main.py 调用）
# -----------------------------------------------------
def create_model(num_classes, device):
    model = ResNet(num_classes=num_classes)
    model = model.to(device)
    return model
