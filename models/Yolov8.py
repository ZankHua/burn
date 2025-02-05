
import torch
import torch.nn as nn
import torchvision.models as models
import timm


class Yolov8(nn.Module):
    """
    可切换的 backbone 模块，用于 YOLOv8 模型中。

    支持三种骨干网络：
      - 'resnet50': 使用 torchvision 的 ResNet50（预训练权重）。
      - 'swin': 使用 timm 的 Swin Transformer（例如 swin_tiny_patch4_window7_224）。
      - 'vit': 使用 timm 的 Vision Transformer（例如 vit_base_patch16_224）。

    参数：
      - backbone_type: 字符串，指定骨干网络类型。可选值：'resnet50', 'swin', 'vit'
      - pretrained: 是否加载预训练权重（默认 True）。
    """

    def __init__(self, backbone_type='resnet50', pretrained=True):
        super(Yolov8, self).__init__()
        self.backbone_type = backbone_type

        if backbone_type == 'resnet50':
            # 使用 torchvision 的 resnet50，并去除分类器部分
            backbone = models.resnet50(pretrained=pretrained)
            # 删去最后两层（平均池化和全连接层）
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])
            # ResNet50 最后一层输出通道数为 2048
            self.out_channels = 2048

        elif backbone_type == 'swin':
            # 使用 timm 加载 Swin Transformer，features_only 返回一个特征图列表
            # 例如使用 swin_tiny_patch4_window7_224 模型
            backbone = timm.create_model('swin_tiny_patch4_window7_224', pretrained=pretrained, features_only=True)
            self.backbone = backbone
            # 获取最后一个特征层的通道数
            self.out_channels = backbone.feature_info[-1]['num_chs']

        elif backbone_type == 'vit':
            # 使用 timm 加载 ViT 模型，例如 vit_base_patch16_224
            # 注意：ViT 原生输出一般为一个嵌入向量序列，此处需根据任务自行处理
            backbone = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=0)
            self.backbone = backbone
            self.out_channels = backbone.embed_dim
        else:
            raise ValueError("Unsupported backbone type: {}".format(backbone_type))

    def forward(self, x):
        """
        前向传播：根据不同的 backbone_type，输出最后一层的特征图或嵌入。
        """
        if self.backbone_type == 'resnet50':
            # ResNet50 直接输出特征图，形状通常为 (batch, 2048, H/32, W/32)
            return self.backbone(x)
        elif self.backbone_type == 'swin':
            # Swin Transformer 使用 features_only 返回多个尺度的特征图，通常取最后一层
            features = self.backbone(x)
            return features[-1]  # 返回最高级别的特征
        elif self.backbone_type == 'vit':
            # ViT 输出为 (batch, num_tokens, embed_dim)
            # 这里直接返回模型的输出嵌入，可以后续根据需要转换为二维特征图
            return self.backbone(x)
        else:
            raise ValueError("Unsupported backbone type: {}".format(self.backbone_type))


# 示例用法
if __name__ == '__main__':
    dummy_input = torch.randn(1, 3, 224, 224)
    for btype in ['resnet50', 'swin', 'vit']:
        model = Yolov8(backbone_type=btype)
        output = model(dummy_input)
        print(f"Backbone: {btype}, Output shape: {output.shape}")
