# build_backbone.py
import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights, ResNet101_Weights
from timm import create_model

def build_backbone(args):
    """
    构建骨干网络，并将输出通道降到 256，用于下游 Transformer (d_model=256)。
    """
    if args.backbone in ['resnet50', 'resnet101']:
        if args.backbone == 'resnet50':
            backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            backbone = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
        # 去掉分类头 => [B,2048,H,W]
        backbone = nn.Sequential(*list(backbone.children())[:-2])

        # 1x1 conv => 2048->256
        backbone = nn.Sequential(
            backbone,
            nn.Conv2d(2048, 256, kernel_size=1)
        )

    elif args.backbone in ['swin_t', 'swin_s']:
        # 使用 timm.create_model 来构建 Swin
        # 注：swin_t 对应 "swin_tiny_patch4_window7_224"，swin_s 对应 "swin_small_patch4_window7_224"
        model_name = "swin_tiny_patch4_window7_224" if args.backbone == 'swin_t' else "swin_small_patch4_window7_224"
        backbone = create_model(model_name, pretrained=True, num_classes=0)
        # backbone.num_features 常见是 768 / 1024
        out_dim = backbone.num_features
        # 包一层 => [B,out_dim,H,W] -> [B,256,H,W]
        backbone = nn.Sequential(
            backbone,
            nn.Conv2d(out_dim, 256, kernel_size=1)
        )

    elif args.backbone == 'vit_b':
        # ViT-B/16 => 768通道
        backbone = create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        out_dim = backbone.num_features
        backbone = nn.Sequential(
            backbone,
            nn.Conv2d(out_dim, 256, kernel_size=1)
        )
    else:
        raise ValueError(f"Unknown backbone: {args.backbone}")

    return backbone
