import torch
import torch.nn.functional as F
from torch import nn
import math

from util import box_ops
from .backbone import build_backbone
from .decoder import build_decoder


class BurnNet(nn.Module):
    def __init__(self, backbone, decoder, num_classes, num_feature_levels, aux_loss=True, with_box_refine=False):
        """
        Initializes the detection model.

        Parameters:
            backbone: 模块，用于特征提取。
            decoder: 模块，用于将骨干输出转换为检测结果。
            num_classes: 检测类别数（这里为 4：无烧伤, 烧伤等级1,2,3）。
            num_feature_levels: 特征金字塔的层数。
            aux_loss: 是否使用辅助损失（在每层 decoder 上计算损失）。
            with_box_refine: 是否进行迭代的边界框细化（此示例中未具体实现）。
        """
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.num_feature_levels = num_feature_levels
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.num_classes = num_classes

    def forward(self, sample):
        """
        前向传播：
          1. 通过 backbone 提取特征。
          2. 将提取到的特征传入 decoder 得到检测结果。

        参数:
            sample: 输入图片的 tensor 或经过预处理后的 batch 数据。

        返回:
            decoder 输出的结果，通常是一个 dict，包含 'pred_logits' 和 'pred_masks' 等字段。
        """
        features = self.backbone(sample)
        outputs = self.decoder(features)
        return outputs


class SetCriterion(nn.Module):
    """
    计算损失的模块。
    本示例实现了一个简单的分类交叉熵损失和分割 mask 的二值交叉熵损失。
    实际项目中通常需要使用 Hungarian Matcher 进行目标与预测匹配，并可能使用 Dice Loss 或 Focal Loss 进行分割损失计算。
    """
    def __init__(self, num_classes, weight_dict, losses, focal_alpha=0.25):
        """
        Parameters:
            num_classes: 类别数（不包括 no-object 类）。
            weight_dict: dict，指定各损失项的权重。
            losses: list，包含要计算的损失名称（例如 ['labels', 'masks']）。
            focal_alpha: Focal Loss 中的 alpha 参数（如使用）。
        """
        super().__init__()
        self.num_classes = num_classes
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.cls_loss = nn.CrossEntropyLoss()
        self.mask_loss = nn.BCELoss()  # 二值交叉熵损失，用于分割 mask

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """
        计算分类损失（交叉熵）。
        这里示例中简单取每个样本的第一个目标对应第一个预测计算损失，
        实际中应采用 Hungarian Matcher 对所有预测与目标进行匹配。
        """
        losses = {}
        # outputs['pred_logits'] 的 shape: (batch_size, num_queries, num_classes)
        pred_logits = outputs['pred_logits']
        target_labels = []
        for t in targets:
            if len(t["labels"]) > 0:
                target_labels.append(t["labels"][0])
            else:
                target_labels.append(torch.tensor(0, device=pred_logits.device))
        target_labels = torch.stack(target_labels)
        loss_ce = self.cls_loss(pred_logits[:, 0, :], target_labels)
        losses['loss_ce'] = loss_ce * self.weight_dict.get('loss_ce', 1.0)
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """
        计算分割 mask 的损失（二值交叉熵）。
        这里示例中简单取每个样本的第一个目标 mask 与第一个预测 mask 进行比较，
        实际中需采用更合理的匹配策略。
        """
        losses = {}
        # outputs['pred_masks'] 的 shape: (batch_size, num_queries, H, W)
        pred_masks = outputs['pred_masks']
        target_masks = []
        for t in targets:
            if "masks" in t and len(t["masks"]) > 0:
                # 使用第一个 mask 作为示例，确保为 float 类型
                mask = t["masks"][0].float()
                # 如果尺寸不匹配，进行插值调整
                if mask.shape != pred_masks.shape[2:]:
                    mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=pred_masks.shape[2:], mode='bilinear', align_corners=False)
                    mask = mask.squeeze(0).squeeze(0)
                target_masks.append(mask)
            else:
                target_masks.append(torch.zeros(pred_masks.shape[2:], device=pred_masks.device))
        target_masks = torch.stack(target_masks)  # shape: (batch_size, H, W)
        # 取第一个预测 mask作为示例
        pred_mask = pred_masks[:, 0, :, :]  # shape: (batch_size, H, W)
        loss_mask = self.mask_loss(pred_mask, target_masks)
        losses['loss_mask'] = loss_mask * self.weight_dict.get('loss_mask', 1.0)
        return losses

    def get_loss(self, loss, outputs, targets, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
        }
        return loss_map[loss](outputs, targets, None, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        num_boxes = sum(len(t["labels"]) for t in targets)
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, num_boxes))
        return losses


class PostProcess(nn.Module):
    """
    将模型输出转换为 COCO API 预期的格式。
    如果输出中包含 'pred_masks'，也将其返回（原始 tensor，需要进一步处理用于评估）。
    """
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        # 如果模型同时输出 bbox 和 mask，则进行相应处理（本示例主要关注 mask 部分）
        results = {}
        # 如果输出中包含边界框信息（备用）
        if 'pred_boxes' in outputs:
            out_logits = outputs['pred_logits']
            out_bbox = outputs['pred_boxes']
            assert len(out_logits) == len(target_sizes)
            assert target_sizes.shape[1] == 2
            prob = out_logits.sigmoid()
            topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
            scores = topk_values
            topk_boxes = topk_indexes // out_logits.shape[2]
            labels = topk_indexes % out_logits.shape[2]
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
            boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
            img_h, img_w = target_sizes.unbind(1)
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
            boxes = boxes * scale_fct[:, None, :]
            results['bbox'] = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        # 对于 mask，直接返回预测结果（注意：通常需要上采样到原图尺寸）
        if 'pred_masks' in outputs:
            results['masks'] = outputs['pred_masks']
        return results


def build(args):
    """
    构建整个检测模型，包括骨干网络、解码器、损失函数和后处理模块。
    针对烧伤检测任务，主要实现烧伤等级检测（分类）和烧伤位置分割（mask 分割）。
    """
    # 烧伤检测任务只有 4 个类别：无烧伤、烧伤等级1、烧伤等级2、烧伤等级3
    num_classes = 4
    device = torch.device(args.device)

    backbone = build_backbone(args)
    # 对于某些骨干（如 swin_t、swin_s、vit_b），需要添加卷积层统一特征通道数
    if args.backbone in ['swin_t', 'swin_s', 'vit_b']:
        backbone = nn.Sequential(
            backbone,
            nn.Conv2d(768 if args.backbone == 'vit_b' else 1024, 256, kernel_size=1)
        )
    decoder = build_decoder(args)
    model = BurnNet(
        backbone,
        decoder,
        num_classes=num_classes,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
    )
    # 构造损失权重字典，针对烧伤检测任务只计算分类损失和分割 mask 损失
    weight_dict = {
        'loss_ce': args.cls_loss_coef if hasattr(args, "cls_loss_coef") else 1.0,
        'loss_mask': args.mask_loss_coef if hasattr(args, "mask_loss_coef") else 1.0
    }
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    # 设置损失类型为 'labels' 和 'masks'
    losses = ['labels', 'masks']
    criterion = SetCriterion(num_classes, weight_dict, losses, focal_alpha=args.focal_alpha if hasattr(args, "focal_alpha") else 0.25)
    criterion.to(device)
    # 后处理模块，返回预测结果格式（包含 bbox 及 mask，如果有）
    postprocessors = {'bbox': PostProcess()}
    return model, criterion, postprocessors
