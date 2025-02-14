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
      - 更新日志信息（损失、分类错误率、学习率、梯度范数）。
    """
    # 设置模型和损失函数为训练模式
    model.train()
    criterion.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 10

    # 利用数据预取器提高数据加载效率
    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        # 前向传播：将输入样本传入模型得到输出
        outputs = model(samples)
        # 使用损失函数计算模型输出与目标之间的损失，返回一个损失字典
        loss_dict = criterion(outputs, targets)
        # 将各损失项按权重加权后求和得到总损失
        total_loss = sum(loss for loss in loss_dict.values())

        # 清除之前的梯度
        optimizer.zero_grad()
        # 反向传播计算梯度
        total_loss.backward()
        # 梯度裁剪（如果设定了最大范数）
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        # 更新模型参数
        optimizer.step()

        # 更新日志：记录当前 batch 的总损失、分类误差、学习率和梯度范数
        metric_logger.update(loss=total_loss.item())
        # 如果 loss_dict 中有 'class_error'（例如分类误差），则更新，否则置 0
        if "class_error" in loss_dict:
            metric_logger.update(class_error=loss_dict["class_error"].item())
        else:
            metric_logger.update(class_error=0)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        # 获取下一个 batch
        samples, targets = prefetcher.next()

    # 同步所有进程的统计信息（用于分布式训练场景）
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    # 返回所有日志记录的全局平均值
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def cust_sumarize(self, catId=None):
    """
    Compute and display summary metrics for evaluation results.
    打印每个类别的 AP（平均精度），详见：https://blog.csdn.net/qq_37541097/article/details/112248194

    返回:
      stats: 各项指标的统计值
      print_string: 格式化后的输出信息
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
            s = self.eval['precision']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            if isinstance(catId, int):
                s = s[:, :, catId, aind, mind]
            else:
                s = s[:, :, :, aind, mind]
        else:
            s = self.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            if isinstance(catId, int):
                s = s[:, catId, aind, mind]
            else:
                s = s[:, :, aind, mind]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])
        print_string = iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
        return mean_s, print_string

    stats, print_list = [0] * 12, [""] * 12
    stats[0], print_list[0] = _summarize(1)
    stats[1], print_list[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
    stats[2], print_list[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
    stats[3], print_list[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
    stats[4], print_list[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[5], print_list[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
    stats[6], print_list[6] = _summarize(0, maxDets=self.params.maxDets[0])
    stats[7], print_list[7] = _summarize(0, maxDets=self.params.maxDets[1])
    stats[8], print_list[8] = _summarize(0, maxDets=self.params.maxDets[2])
    stats[9], print_list[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
    stats[10], print_list[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[11], print_list[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])

    print_info = "\n".join(print_list)

    if not self.eval:
        raise Exception('Please run accumulate() first')

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
      - 遍历验证集，计算损失并将预测结果转换为 COCO 格式；
      - 更新 COCO 评估器并汇总指标；
      - 计算每个类别的 AP 并输出。
    """
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # 初始化 COCO 评估器
    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # 前向传播：传入样本得到模型输出
        outputs = model(samples)
        # 计算损失并更新日志（评估时也计算损失以便监控）
        loss_dict = criterion(outputs, targets)
        total_loss = sum(loss for loss in loss_dict.values())
        metric_logger.update(loss=total_loss.item())
        if "class_error" in loss_dict:
            metric_logger.update(class_error=loss_dict["class_error"].item())
        else:
            metric_logger.update(class_error=0)

        # 取出原始图片尺寸（用于将预测结果从相对坐标转换为绝对坐标）
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        # 利用后处理器将模型输出转换为 COCO 预期格式（例如预测边界框）
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        # 将每个样本的 image_id 与其对应的预测结果构成字典
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        coco_evaluator.update(res)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # 调用 cust_sumarize 打印每个类别的 AP（如果需要）
    # 这里假设 cust_sumarize 函数已经定义在 coco_evaluator.coco_eval["bbox"] 中
    coco_eval = coco_evaluator.coco_eval["bbox"]
    coco_stats, print_coco = cust_sumarize(coco_eval)
    print(print_coco)

    # 此处可根据烧伤检测任务，将类别映射为烧伤等级（例如 1, 2, 3）及其它信息，并打印对应指标
    # 示例代码：
    class_ids = {'grade1': 1, 'grade2': 2, 'grade3': 3}
    category_index = {v: k for k, v in class_ids.items()}
    voc_map_info_list = []
    for i in range(len(category_index)):
        stats_i, _ = cust_sumarize(coco_eval, catId=i)
        voc_map_info_list.append(" {:15}: {}".format(category_index.get(i+1, f'cat{i+1}'), stats_i))
    print_voc = "\n".join(voc_map_info_list)
    print("Per-category AP:\n", print_voc)

    # 将验证结果保存到指定文件中
    with open("record_mAP.txt", "w") as f:
        record_lines = ["COCO results:",
                        print_coco,
                        "",
                        "AP for each category:",
                        print_voc]
        f.write("\n".join(record_lines))
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
          - 更新日志信息（损失、分类错误率、学习率、梯度范数）。
        """
        # 设置模型和损失函数为训练模式
        model.train()
        criterion.train()

        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        header = f'Epoch: [{epoch}]'
        print_freq = 10

        # 利用数据预取器提高数据加载效率
        prefetcher = data_prefetcher(data_loader, device, prefetch=True)
        samples, targets = prefetcher.next()

        for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
            # 前向传播：将输入样本传入模型得到输出
            outputs = model(samples)
            # 使用损失函数计算模型输出与目标之间的损失，返回一个损失字典
            loss_dict = criterion(outputs, targets)
            # 将各损失项按权重加权后求和得到总损失
            total_loss = sum(loss for loss in loss_dict.values())

            # 清除之前的梯度
            optimizer.zero_grad()
            # 反向传播计算梯度
            total_loss.backward()
            # 梯度裁剪（如果设定了最大范数）
            if max_norm > 0:
                grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            else:
                grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
            # 更新模型参数
            optimizer.step()

            # 更新日志：记录当前 batch 的总损失、分类误差、学习率和梯度范数
            metric_logger.update(loss=total_loss.item())
            # 如果 loss_dict 中有 'class_error'（例如分类误差），则更新，否则置 0
            if "class_error" in loss_dict:
                metric_logger.update(class_error=loss_dict["class_error"].item())
            else:
                metric_logger.update(class_error=0)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(grad_norm=grad_total_norm)

            # 获取下一个 batch
            samples, targets = prefetcher.next()

        # 同步所有进程的统计信息（用于分布式训练场景）
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        # 返回所有日志记录的全局平均值
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    def cust_sumarize(self, catId=None):
        """
        Compute and display summary metrics for evaluation results.
        打印每个类别的 AP（平均精度），详见：https://blog.csdn.net/qq_37541097/article/details/112248194

        返回:
          stats: 各项指标的统计值
          print_string: 格式化后的输出信息
        """

        def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap == 1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) if iouThr is None else '{:0.2f}'.format(
                iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

            if ap == 1:
                s = self.eval['precision']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if isinstance(catId, int):
                    s = s[:, :, catId, aind, mind]
                else:
                    s = s[:, :, :, aind, mind]
            else:
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if isinstance(catId, int):
                    s = s[:, catId, aind, mind]
                else:
                    s = s[:, :, aind, mind]

            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            print_string = iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
            return mean_s, print_string

        stats, print_list = [0] * 12, [""] * 12
        stats[0], print_list[0] = _summarize(1)
        stats[1], print_list[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
        stats[2], print_list[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
        stats[3], print_list[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
        stats[4], print_list[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
        stats[5], print_list[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
        stats[6], print_list[6] = _summarize(0, maxDets=self.params.maxDets[0])
        stats[7], print_list[7] = _summarize(0, maxDets=self.params.maxDets[1])
        stats[8], print_list[8] = _summarize(0, maxDets=self.params.maxDets[2])
        stats[9], print_list[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
        stats[10], print_list[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
        stats[11], print_list[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])

        print_info = "\n".join(print_list)

        if not self.eval:
            raise Exception('Please run accumulate() first')

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
          - 遍历验证集，计算损失并将预测结果转换为 COCO 格式；
          - 更新 COCO 评估器并汇总指标；
          - 计算每个类别的 AP 并输出。
        """
        model.eval()
        criterion.eval()

        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        header = 'Test:'

        # 初始化 COCO 评估器
        iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
        coco_evaluator = CocoEvaluator(base_ds, iou_types)

        for samples, targets in metric_logger.log_every(data_loader, 10, header):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # 前向传播：传入样本得到模型输出
            outputs = model(samples)
            # 计算损失并更新日志（评估时也计算损失以便监控）
            loss_dict = criterion(outputs, targets)
            total_loss = sum(loss for loss in loss_dict.values())
            metric_logger.update(loss=total_loss.item())
            if "class_error" in loss_dict:
                metric_logger.update(class_error=loss_dict["class_error"].item())
            else:
                metric_logger.update(class_error=0)

            # 取出原始图片尺寸（用于将预测结果从相对坐标转换为绝对坐标）
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            # 利用后处理器将模型输出转换为 COCO 预期格式（例如预测边界框）
            results = postprocessors['bbox'](outputs, orig_target_sizes)
            # 将每个样本的 image_id 与其对应的预测结果构成字典
            res = {target['image_id'].item(): output for target, output in zip(targets, results)}
            coco_evaluator.update(res)

        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

        # 调用 cust_sumarize 打印每个类别的 AP（如果需要）
        # 这里假设 cust_sumarize 函数已经定义在 coco_evaluator.coco_eval["bbox"] 中
        coco_eval = coco_evaluator.coco_eval["bbox"]
        coco_stats, print_coco = cust_sumarize(coco_eval)
        print(print_coco)

        # 此处可根据烧伤检测任务，将类别映射为烧伤等级（例如 1, 2, 3）及其它信息，并打印对应指标
        # 示例代码：
        class_ids = {'grade1': 1, 'grade2': 2, 'grade3': 3}
        category_index = {v: k for k, v in class_ids.items()}
        voc_map_info_list = []
        for i in range(len(category_index)):
            stats_i, _ = cust_sumarize(coco_eval, catId=i)
            voc_map_info_list.append(" {:15}: {}".format(category_index.get(i + 1, f'cat{i + 1}'), stats_i))
        print_voc = "\n".join(voc_map_info_list)
        print("Per-category AP:\n", print_voc)

        # 将验证结果保存到指定文件中
        with open("record_mAP.txt", "w") as f:
            record_lines = ["COCO results:",
                            print_coco,
                            "",
                            "AP for each category:",
                            print_voc]
            f.write("\n".join(record_lines))

        # 组合统计信息
        panoptic_res = None  # 如果有 panoptic 评估结果，可在此添加
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

    # 组合统计信息
    panoptic_res = None  # 如果有 panoptic 评估结果，可在此添加
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
