# burnnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from util import box_ops

# 假设你要在同目录下的 backbone.py / decoder.py 中导入函数
# 如果你包结构是 models/backbone.py, models/decoder.py, 则可以：
from .backbone import build_backbone
from .decoder import build_decoder


class BurnNet(nn.Module):
    def __init__(self, backbone, decoder, num_classes=4,
                 num_feature_levels=1, aux_loss=True, with_box_refine=False):
        """
        backbone: 输出 [B,256,H,W] 的骨干网络
        decoder: 需要 [S,B,256] 的 transformer decoder
        """
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.num_feature_levels = num_feature_levels
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.num_classes = num_classes

    def forward(self, sample):
        # 如果 sample 是 NestedTensor => x=sample.tensors; 否则 x=sample
        if hasattr(sample, "tensors"):
            x = sample.tensors
        else:
            x = sample

        # 1) 通过骨干 => [B,256,H,W]
        feat = self.backbone(x)

        # 2) flatten+permute => [H*W,B,256]
        B, C, H, W = feat.shape
        feat = feat.flatten(2)         # => [B,256,H*W]
        feat = feat.permute(2,0,1)     # => [H*W,B,256]

        # 3) decoder => dict { 'pred_logits','pred_masks',... }
        outputs = self.decoder(feat)
        return outputs


class SetCriterion(nn.Module):
    """
    计算损失的模块。
    本示例实现了一个简单分类交叉熵 + 分割 mask 的BCELoss。
    在 loss_labels 中，额外统计 4 个类别分别的错误率 + 总错误率。
    """
    def __init__(self, num_classes, weight_dict, losses, focal_alpha=0.25):
        super().__init__()
        self.num_classes = num_classes
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        # 简单CE
        self.cls_loss = nn.CrossEntropyLoss()
        # 简单二值交叉熵
        self.mask_loss= nn.BCELoss()

        # 如果想给每个类别命个好记的名字，也可以这样:
        # self.id2name = {
        #     0: 'not_burn',
        #     1: '1st_degree',
        #     2: '2nd_degree',
        #     3: '3rd_degree'
        # }
        # 或者你也可以直接在loss里写死

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """
        计算分类损失（交叉熵） + 额外统计整体 class_error + 每个类别的 class_error。
        """
        losses = {}
        # shape=(B,num_queries,num_classes)
        pred_logits = outputs['pred_logits']
        # 仅示例：把所有样本都当 label=0，或取 targets 的第一个 label 做演示
        target_labels = []
        for t in targets:
            if len(t["labels"])>0:
                target_labels.append(t["labels"][0])
            else:
                target_labels.append(torch.tensor(0, device=pred_logits.device))
        target_labels = torch.stack(target_labels)  # =>(B,)

        # -- 1) 计算交叉熵
        loss_ce = self.cls_loss(pred_logits[:,0,:], target_labels)
        losses['loss_ce'] = loss_ce * self.weight_dict.get('loss_ce',1.0)

        # -- 2) 计算整体的分类错误率 (class_error)
        with torch.no_grad():
            pred_class = pred_logits[:,0,:].argmax(dim=-1)  # (B,)
            acc_all = (pred_class == target_labels).float().mean()  # 整体准确率
            class_error_all = 100.0 * (1.0 - acc_all)               # 整体错误率(百分比)
        losses['class_error'] = class_error_all

        # -- 3) 计算每个类别的错误率
        #     注意，如果 batch 中没有该类别的数据，就跳过
        with torch.no_grad():
            for class_id in range(self.num_classes):
                # 取该类别的掩码
                class_mask = (target_labels == class_id)
                if class_mask.sum() == 0:
                    # 如果这一批里没有该类别的样本，可选择跳过或赋值为0/None
                    continue

                # 对子集中计算准确率
                pred_in_class = pred_class[class_mask]  # 该子集的预测
                acc_c = (pred_in_class == class_id).float().mean()
                class_err_c = 100.0 * (1.0 - acc_c)

                # 例如在 losses 字典中加一个 key
                # 格式可以自定义，如 "class_error_0" 或更详细
                losses[f'class_error_{class_id}'] = class_err_c

                # 如果你有一个 self.id2name 映射，也可以这样：
                # name = self.id2name[class_id]
                # losses[f'class_error_{name}'] = class_err_c

        return losses

    # -------------- 其它部分保持不变 ------------------
    def loss_masks(self, outputs, targets, indices, num_boxes):
        losses={}
        pred_masks = outputs['pred_masks']  # (B,num_queries,H,W)
        pred_mask = pred_masks[:,0,:,:]  # =>(B,H,W)

        target_masks=[]
        for t in targets:
            if "masks" in t and len(t["masks"])>0:
                mask = t["masks"][0].float().to(pred_mask.device)
                if mask.shape != pred_mask.shape[1:]:
                    mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0),
                                         size=pred_mask.shape[1:],
                                         mode='bilinear', align_corners=False)[0,0]
                target_masks.append(mask)
            else:
                target_masks.append(torch.zeros_like(pred_mask[0]))
        target_masks = torch.stack(target_masks)

        loss_mask = self.mask_loss(pred_mask, target_masks)
        losses['loss_mask'] = loss_mask * self.weight_dict.get('loss_mask',1.0)
        return losses

    def get_loss(self, loss, outputs, targets, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
        }
        return loss_map[loss](outputs, targets, None, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        num_boxes = sum(len(t["labels"]) for t in targets)
        losses={}
        for l in self.losses:
            losses.update(self.get_loss(l, outputs, targets, num_boxes))
        return losses


class PostProcess(nn.Module):
    """
    将模型输出转换为 COCO API 预期的格式(可做mask上采样等)。
    """
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        # 仅简单返回
        return outputs


###################################
# 一个build函数,把以上组件组合
###################################
def build(args):
    """
    将 backbone、decoder、BurnNet、SetCriterion、PostProcess 组装。
    """
    # 1) 构建backbone => [B,256,H,W]
    backbone = build_backbone(args) 

    # 2) 构建decoder => Transformer
    dec = build_decoder(args)

    # 3) 组装 BurnNet
    model = BurnNet(
        backbone=backbone,
        decoder=dec,
        num_classes=4,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine
    )
    model.to(args.device)

    # 4) 组装损失
    weight_dict = {
        'loss_ce': getattr(args,'cls_loss_coef',1.0),
        'loss_mask':getattr(args,'mask_loss_coef',1.0)
    }
    losses = ['labels','masks']
    criterion = SetCriterion(4, weight_dict, losses)
    criterion.to(args.device)

    # 5) 后处理
    postprocessors = {"bbox": PostProcess()}

    return model, criterion, postprocessors
