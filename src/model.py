import torch
import torch.nn as nn
from torchvision import models

def build_gear_cnn_model(
    model_name: str = "mobilenet_v2",
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = True
) -> nn.Module:
    """
    Constructs a transfer learning CNN classifier using MobileNetV2 or EfficientNet-B0.
    
    Classes:
      0: no_undercut ("No Undercut")
      1: undercut ("Undercut")
    """
    if model_name.lower() == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)

        if freeze_backbone and pretrained:
            for param in model.features.parameters():
                param.requires_grad = False

        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )

    elif model_name.lower() == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)

        if freeze_backbone and pretrained:
            for param in model.features.parameters():
                param.requires_grad = False

        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )

    else:
        raise ValueError(f"Unsupported model architecture: {model_name}. Supported: 'mobilenet_v2', 'efficientnet_b0'")

    return model
