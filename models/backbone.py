# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from jinja2.utils import Joiner
from timm import create_model
from torch import nn
from torchvision.models import ResNet101_Weights, ResNet50_Weights
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
from models import swin_transformer
from torchvision import models
from util.misc import NestedTensor, is_main_process

# from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool, args):
        super().__init__()
        #swin
        if 'swin' in args.backbone:
            num_channels = [int(backbone.embed_dim * 2 ** i) for i in range(backbone.num_layers)]
            if return_interm_layers:
                return_layers = [2, 3, 4]
                self.strides = [8, 16, 32]
                self.num_channels = num_channels[1:]
            else:
                return_layers = [4]
                self.strides = [32]
                self.num_channels = num_channels[-1]
            self.body = backbone
        else:
            for name, parameter in backbone.named_parameters():
                if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                    parameter.requires_grad_(False)
            if return_interm_layers:
                # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
                return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
                self.strides = [8, 16, 32]
                self.num_channels = [512, 1024, 2048]
            else:
                return_layers = {'layer4': "0"}
                self.strides = [32]
                self.num_channels = [2048]
            self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool,
                 args):
        print(f"Backbone: {name}")
        pretrained = is_main_process() and not args.backbone_from_scratch and not args.scrl_pretrained_path
        if 'swin' in name:
            assert not dilation, "not supported"
            if not args.backbone_from_scratch and not args.finetune_early_layers:
                print("Freeze early layers.")
                frozen_stages = 2
            else:
                print('Finetune early layers as well.')
                frozen_stages = -1
            if return_interm_layers:
                out_indices = [1, 2, 3]
            else:
                out_indices = [3]

            backbone = swin_transformer.build_model(
                name, out_indices=out_indices, frozen_stages=frozen_stages, pretrained=pretrained)
        else:
            norm_layer = FrozenBatchNorm2d
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                pretrained=is_main_process(), norm_layer=norm_layer)

        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        super().__init__(backbone, train_backbone, return_interm_layers, args)
        if dilation and not "swin" in name:
            self.strides[-1] = self.strides[-1] // 2

'''
如果你要使用DETR类的方法的话就用这个
class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos
'''

'''
如果只使用backbone的多尺度特征就用这个
class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding=None):
        super().__init__(backbone)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels
        self.position_embedding = position_embedding  # 使位置嵌入可选

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        
        # 不再使用位置嵌入
        pos = []  
        
        for name, x in sorted(xs.items()):
            out.append(x)

        # 如果不使用位置嵌入，则直接返回特征图
        if self.position_embedding is not None:
            for x in out:
                pos.append(self.position_embedding(x).to(x.tensors.dtype))

        return out, pos
'''

def build_backbone(args):
    """
    Builds the backbone model (ResNet, Swin Transformer, or ViT)
    and ensures final output channels = 256 if needed.
    """
    # 1) 构建 base backbone
    if args.backbone in ['resnet50', 'resnet101']:
        if args.backbone == 'resnet50':
            backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:  # 'resnet101'
            backbone = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
        # 去除 FC
        backbone = nn.Sequential(*list(backbone.children())[:-2])  # => [B, 2048, H, W]

        # 在末尾加1x1 conv => [B,256,H,W]
        backbone = nn.Sequential(
            backbone,
            nn.Conv2d(2048, 256, kernel_size=1)
        )

    elif args.backbone in ['swin_t', 'swin_s']:
        # 例如 'swin_t' => 'swin_tiny_patch4_window7_224'
        model_name = "swin_tiny_patch4_window7_224" if args.backbone == 'swin_t' else "swin_small_patch4_window7_224"
        backbone = create_model(model_name, pretrained=True, num_classes=0)
        # backbone输出可能是 768 通道 (tiny) 或 768/1024 等
        # 需在外部加1x1 conv => [B,256,H,W]
        # 先检查 backbone 的输出通道 backbone.num_features
        out_dim = backbone.num_features  # e.g. 768
        # 用 sequential 包裹
        backbone = nn.Sequential(
            backbone,
            nn.Conv2d(out_dim, 256, kernel_size=1)
        )

    elif args.backbone == 'vit_b':
        # ViT-B/16 => 最终通道768
        backbone = create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        # 这里同理 => [B,768,H/16,W/16], 需要1x1 conv
        out_dim = backbone.num_features  # 768
        backbone = nn.Sequential(
            backbone,
            nn.Conv2d(out_dim, 256, kernel_size=1)
        )
    else:
        raise ValueError(f"Unknown backbone: {args.backbone}")

    return backbone