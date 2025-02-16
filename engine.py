"""
Train and eval functions used in main.py for the burn detection project.
"""

from typing import Iterable
import numpy as np
import torch
from datasets.data_prefetcher import data_prefetcher
import util.misc as utils
from datasets.coco_eval import CocoEvaluator


def train_one_epoch(model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    max_norm: float = 0):
    """
    单个训练周期 (epoch) 的训练函数。

    主要步骤：
      - 将模型和损失函数设置为训练模式；
      - 利用 data_prefetcher 预取数据；
      - 对每个 batch 执行前向传播、计算损失、反向传播、梯度裁剪和参数更新；
      - 记录并打印训练过程中的各种指标（损失、分类错误率、学习率、梯度范数等）。
    """

    # 1) 模型和损失函数切换到 train 模式
    model.train()
    criterion.train()

    # 2) 创建日志记录器
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 10

    # 3) 预取数据
    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    # 4) 迭代所有 batch
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        # 前向传播
        outputs = model(samples)
        # 计算损失
        loss_dict = criterion(outputs, targets)
        total_loss = sum(loss for loss in loss_dict.values())

        # 清空梯度 -> 反向传播 -> 梯度裁剪 -> 更新参数
        optimizer.zero_grad()
        total_loss.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        # 记录日志：loss、class_error、lr、grad_norm
        metric_logger.update(loss=total_loss.item())
        if "class_error" in loss_dict:
            metric_logger.update(class_error=loss_dict["class_error"].item())
        else:
            metric_logger.update(class_error=0)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        # 取下一批
        samples, targets = prefetcher.next()

    # 5) 同步并打印平均值
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # 返回平均日志值（供外部存档或打印）
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def cust_sumarize(self, catId=None):
    """
    Compute and display summary metrics for evaluation results.
    打印每个类别的 AP 等信息，支持传入 catId 对某一类别做单独统计。

    参数:
      self: 这里是将 coco_eval 对象当作第一个参数 (self)，
            因此在外部调用时是 cust_sumarize(coco_eval, catId=xx)
      catId: 要统计的类别 id (可为 None 或具体的类别索引)

    返回:
      stats:   各项指标的统计值 (数组)
      print_info: 格式化输出字符串
    """
    def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
        p = self.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap == 1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

        if ap == 1:
            s = self.eval['precision']  # shape: [TxRxKxAxM]
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            # 维度: T x R x K x A x M
            if isinstance(catId, int):
                s = s[:, :, catId, aind, mind]
            else:
                s = s[:, :, :, aind, mind]
        else:
            s = self.eval['recall']  # shape: [TxKxAxM]
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            # 维度: T x K x A x M
            if isinstance(catId, int):
                s = s[:, catId, aind, mind]
            else:
                s = s[:, :, aind, mind]

        # 若无有效数据 => -1, 否则计算均值
        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])

        print_string = iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
        return mean_s, print_string

    if not self.eval:
        raise Exception('Please run accumulate() first')

    stats, print_list = [0]*12, [""]*12
    stats[0],  print_list[0]  = _summarize(ap=1)
    stats[1],  print_list[1]  = _summarize(ap=1, iouThr=.5,  maxDets=self.params.maxDets[2])
    stats[2],  print_list[2]  = _summarize(ap=1, iouThr=.75, maxDets=self.params.maxDets[2])
    stats[3],  print_list[3]  = _summarize(ap=1, areaRng='small', maxDets=self.params.maxDets[2])
    stats[4],  print_list[4]  = _summarize(ap=1, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[5],  print_list[5]  = _summarize(ap=1, areaRng='large', maxDets=self.params.maxDets[2])
    stats[6],  print_list[6]  = _summarize(ap=0, maxDets=self.params.maxDets[0])
    stats[7],  print_list[7]  = _summarize(ap=0, maxDets=self.params.maxDets[1])
    stats[8],  print_list[8]  = _summarize(ap=0, maxDets=self.params.maxDets[2])
    stats[9],  print_list[9]  = _summarize(ap=0, areaRng='small', maxDets=self.params.maxDets[2])
    stats[10], print_list[10] = _summarize(ap=0, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[11], print_list[11] = _summarize(ap=0, areaRng='large', maxDets=self.params.maxDets[2])

    print_info = "\n".join(print_list)
    return stats, print_info


def evaluate(model: torch.nn.Module,
             criterion: torch.nn.Module,
             postprocessors: dict,
             data_loader: Iterable,
             base_ds,
             device: torch.device,
             output_dir: str):
    """
    评估函数：
      - 将模型和损失函数设置为评估模式；
      - 遍历验证集，计算损失并将预测结果转成 COCO 格式；
      - 利用 COCO API 累计统计量并输出 summary；
      - 可选地打印每类别的 AP 等信息。
    """
    # 1) 模型+损失切到 eval
    model.eval()
    criterion.eval()

    # 2) 记录器
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # 3) 初始化 COCO 评估器
    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    # 4) 遍历所有验证数据
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 前向 + 损失
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        total_loss = sum(loss for loss in loss_dict.values())

        metric_logger.update(loss=total_loss.item())
        if "class_error" in loss_dict:
            metric_logger.update(class_error=loss_dict["class_error"].item())
        else:
            metric_logger.update(class_error=0)

        # 后处理 => 将预测结果转换为 COCO 格式
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)

        # 更新到 coco_evaluator
        res = {t['image_id'].item(): r for t, r in zip(targets, results)}
        coco_evaluator.update(res)

    # 5) 同步、accumulate、输出
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    # 6) 统计信息
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # 7) 打印每个类别的 AP（可选）
    coco_eval = coco_evaluator.coco_eval["bbox"]  # 仅对 bbox
    coco_stats, print_coco = cust_sumarize(coco_eval)
    print(print_coco)

    # 示例：自定义类别映射
    class_ids = {'grade1': 1, 'grade2': 2, 'grade3': 3}
    category_index = {v: k for k, v in class_ids.items()}

    # 对每个类别依次做 summarize
    voc_map_info_list = []
    for i in range(len(category_index)):
        stats_i, _ = cust_sumarize(coco_eval, catId=i)
        cat_name = category_index.get(i+1, f'cat{i+1}')
        voc_map_info_list.append(f" {cat_name:15}: {stats_i}")
    print_voc = "\n".join(voc_map_info_list)
    print("Per-category AP:\n", print_voc)

    # 写到本地文件
    with open("record_mAP.txt", "w") as f:
        record_lines = ["COCO results:",
                        print_coco,
                        "",
                        "AP for each category:",
                        print_voc]
        f.write("\n".join(record_lines))

    # 8) 若要返回更多信息,可扩展
    panoptic_res = None
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]

    return stats, coco_evaluator
