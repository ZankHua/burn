import os
from collections import abc, OrderedDict
import yaml
import torch
import torch.nn as nn
import torchvision.models as models

from .swin_transformer import SwinTransformer
from .config import Config

from models.backbone import build_backbone
from models.decoder import build_decoder
from models.burnnet import BurnNet, SetCriterion, PostProcess


CONFIG_MAP = {
    "swin-t": "models/swin_transformer/configs/swin_tiny_patch4_window7_224.yaml",
    "swin-s": "models/swin_transformer/configs/swin_small_patch4_window7_224.yaml",
    "swin-b": "models/swin_transformer/configs/swin_base_patch4_window7_224.yaml",
    "swin-l": "models/swin_transformer/configs/swin_large_patch4_window7_224.yaml",
}

CHECKPOINT_MAP = {
    "swin-t": "/data/wangyi/swin_tiny_patch4_window7_224.pth",
    # 其它 ckpt 可按需添加
}


def load_config_yaml(cfg_file, config=None):
    """加载 Swin Transformer YAML 配置。"""
    if config is None:
        from collections import OrderedDict
        config = OrderedDict()
    with open(cfg_file, 'r') as f:
        config_src = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in config_src.setdefault('BASE', ['']):
        if cfg:
            load_config_yaml(
                os.path.join(os.path.dirname(cfg_file), cfg), config
            )
    print('=> merge config from {}'.format(cfg_file))
    _update_dict(config, config_src)
    return config


def _update_dict(tar, src):
    """递归地更新字典。"""
    for k, v in src.items():
        if isinstance(v, abc.Mapping):
            tar[k] = _update_dict(tar.get(k, {}), v)
        else:
            tar[k] = v
    return tar


def build_model(name, out_indices=(0, 1, 2, 3), frozen_stages=0, pretrained=True):
    """
    同时支持 Swin Transformer 与 ResNet。

    参数:
      name: 字符串, 可为 'swin-t', 'swin-s', 'resnet50', 'resnet101' 等
      out_indices: Swin 使用的输出特征层索引(不一定实际用到)
      frozen_stages: 冻结多少层
      pretrained: 是否加载预训练权重
    返回:
      一个 backbone 模块, 输出 shape = [B, 256, H, W] (若是Swin则embedding=256;若是ResNet,加1x1 conv降维)
    """
    # 1) Swin Transformer
    if name in CONFIG_MAP:
        # 加载 Swin 配置
        config_file = CONFIG_MAP[name]
        from .config import Config
        config = load_config_yaml(config_file, config=Config())
        config.freeze()

        model_type = config.MODEL.TYPE
        if model_type == 'swin':
            model = SwinTransformer(
                pretrain_img_size=config.DATA.IMG_SIZE,
                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                in_chans=config.MODEL.SWIN.IN_CHANS,
                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                depths=config.MODEL.SWIN.DEPTHS,
                num_heads=config.MODEL.SWIN.NUM_HEADS,
                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                qk_scale=config.MODEL.SWIN.QK_SCALE,
                drop_rate=config.MODEL.DROP_RATE,
                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                ape=config.MODEL.SWIN.APE,
                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                out_indices=out_indices,
                frozen_stages=frozen_stages
            )
        else:
            raise NotImplementedError(f"Unknown model_type: {model_type}")

        # 是否加载官方预训练
        if pretrained:
            ckpt_path = CHECKPOINT_MAP.get(name, None)
            if ckpt_path and os.path.isfile(ckpt_path):
                state_dict = torch.load(ckpt_path)
                model.load_state_dict(state_dict['model'], strict=False)
            else:
                print(f"[Warn] No local checkpoint for {name}, using default PyTorch init.")

        # 如果Swin最后输出通道可能是256(例如 tiny patch4 embed_dim=96 ->逐层扩大),
        # 若你想强制=256, 可加个1x1 conv, 不过Swin通常会在最后层输出 embed_dim=768 或1024之类,
        # 需根据实际“SwinTransformer” out_channels 改用Conv2d(...)
        # 这里省略,或视具体Swin配置在外部再加1x1 conv

        return model

    # 2) ResNet
    elif name in ["resnet50", "resnet101"]:
        if name == "resnet50":
            backbone = models.resnet50(pretrained=pretrained)
        else:
            backbone = models.resnet101(pretrained=pretrained)

        # 移除FC层
        backbone = nn.Sequential(*list(backbone.children())[:-2])  # [B,2048,H,W]

        # 在后面加1x1Conv把2048->256
        backbone = nn.Sequential(
            backbone,
            nn.Conv2d(2048, 256, kernel_size=1)
        )

        # 可根据 frozen_stages 冻结若干层
        # 例如 if frozen_stages >=1: for param in backbone[0][...].parameters(): param.requires_grad=False
        # 这里省略

        return backbone

    else:
        raise NotImplementedError(f"Unknown backbone name: {name}")

def build(args):
    device = torch.device(args.device)

    backbone = build_backbone(args)  # => [B,256,H,W]
    decoder = build_decoder(args)    # => Transformer decoder

    model = BurnNet(
        backbone=backbone,
        decoder=decoder,
        num_classes=4,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine
    )

    weight_dict = {
        'loss_ce': getattr(args, 'cls_loss_coef', 1.0),
        'loss_mask': getattr(args, 'mask_loss_coef', 1.0)
    }
    # 这里再组合 SetCriterion
    losses = ['labels', 'masks']
    criterion = SetCriterion(4, weight_dict, losses)  # for example
    criterion.to(device)

    postprocessors = {'bbox': PostProcess()}  # or your usage

    return model, criterion, postprocessors




