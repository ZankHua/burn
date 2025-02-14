# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime.
# Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------

import torch

def to_cuda(samples, targets, device):
    samples = samples.to(device, non_blocking=True)
    targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
    return samples, targets

class data_prefetcher:
    def __init__(self, loader, device, prefetch=True):
        """
        构造函数:
          loader: DataLoader 迭代器
          device: 训练设备 (可能是 'cuda', 'cpu' 或 'mps')
          prefetch: 是否预取数据 (默认 True)
        """
        self.loader = iter(loader)
        self.prefetch = prefetch
        self.device = device
        if self.prefetch and self.device.type == 'cuda':
            self.stream = torch.cuda.Stream()
        else:
            self.stream = None

        if self.prefetch:
            self.preload()

    def preload(self):
        """
        预取下一个 batch 数据并将其搬移到指定设备上
        """
        try:
            self.next_samples, self.next_targets = next(self.loader)
        except StopIteration:
            self.next_samples = None
            self.next_targets = None
            return

        if self.stream is not None:
            # 如果有 CUDA stream，则在该 stream 上进行数据搬移
            with torch.cuda.stream(self.stream):
                self.next_samples, self.next_targets = to_cuda(self.next_samples, self.next_targets, self.device)
        else:
            # 如果不是 CUDA (例如 'cpu' 或 'mps')，直接 to(device)
            self.next_samples, self.next_targets = to_cuda(self.next_samples, self.next_targets, self.device)

    def next(self):
        """
        返回当前 batch，并启动下一次预取 (如果有)。
        """
        if not self.prefetch:
            # 不使用预取时，直接从 DataLoader 取数据并转到指定设备
            try:
                samples, targets = next(self.loader)
                samples, targets = to_cuda(samples, targets, self.device)
            except StopIteration:
                samples, targets = None, None
            return samples, targets

        # 使用预取时，需要等待前一个 stream 完成数据拷贝
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)

        samples = self.next_samples
        targets = self.next_targets

        # 让当前主流记录 tensor 的使用，以避免异步数据搬运未完成
        if samples is not None and self.stream is not None:
            samples.record_stream(torch.cuda.current_stream())
        if targets is not None and self.stream is not None:
            for t in targets:
                for k, v in t.items():
                    v.record_stream(torch.cuda.current_stream())

        # 再次预取下一批数据
        self.preload()

        return samples, targets
