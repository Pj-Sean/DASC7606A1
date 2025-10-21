import torch.nn as nn
import torchvision.models as models


def _convert_to_resnetd(block: nn.Module, stride: int) -> None:
    """Modify the residual block's downsample path following ResNet-D."""

    if not hasattr(block, "downsample") or block.downsample is None:
        return

    in_channels = block.conv1.in_channels
    out_channels = block.conv3.out_channels

    modules = []
    if stride > 1:
        modules.append(
            nn.AvgPool2d(
                kernel_size=stride,
                stride=stride,
                ceil_mode=True,
                count_include_pad=False,
            )
        )
    modules.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False))
    modules.append(nn.BatchNorm2d(out_channels))

    block.downsample = nn.Sequential(*modules)


class ResNet50E5(nn.Module):
    """ResNet-50_E5 architecture packaged as an ``nn.Module``."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()

        model = models.resnet50(weights=None, replace_stride_with_dilation=[False, False, True])

        # CIFAR-style stem modifications: 3x3 stride1 conv and no initial max-pool.
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()

        # Apply ResNet-D downsample tweaks.
        _convert_to_resnetd(model.layer2[0], stride=2)
        _convert_to_resnetd(model.layer3[0], stride=2)
        # Final stage keeps stride=1 (already handled by replace_stride_with_dilation).
        _convert_to_resnetd(model.layer4[0], stride=1)

        # Replace classification head.
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        self.model = model

    def forward(self, x):
        return self.model(x)


def create_model(num_classes, device):
    """Create and initialize the model."""

    model = ResNet50E5(num_classes=num_classes)
    model = model.to(device)
    return model
