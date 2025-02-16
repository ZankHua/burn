"""
Misc functions, including distributed helpers.
Mostly copy-paste from torchvision references, but cleaned up for newer versions (>0.7).
"""

import os
import subprocess
import time
from collections import defaultdict, deque
import datetime
import pickle
from typing import Optional, List

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
from torch import Tensor


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count if self.count > 0 else 0

    @property
    def max(self):
        return max(self.deque) if len(self.deque) > 0 else 0

    @property
    def value(self):
        return self.deque[-1] if len(self.deque) > 0 else 0

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value
        )


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).
    Returns list[data] with data from each rank.
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialize to Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # gather sizes
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # pad tensors
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)

    # gather all
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, t in zip(size_list, tensor_list):
        buffer = t.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))
    return data_list


def reduce_dict(input_dict, average=True):
    """
    Reduce values in the dictionary from all processes so every process
    has the averaged/summed results.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        return {k: v for k, v in zip(names, values)}


class MetricLogger(object):
    """
    A simple logger that keeps track of metrics (via SmoothedValue) and
    prints them every log_freq steps.
    """
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int)), f"{k} must be float or int"
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        return super().__getattribute__(attr)

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {meter}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        """
        Prints logs every `print_freq` steps, in format:
          [   i/TOTAL] eta: ...
        where i is aligned with the length of the dataset.
        """
        i = 0
        if header is None:
            header = ''
        start_time = time.time()
        end = time.time()

        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')

        # 1) 计算宽度 => e.g. dataset有120个batch, 则 space_count=3 => i显示为  42 /120
        space_count = len(str(len(iterable)))
        # 2) 构造一个 "3d" 格式
        space_fmt = f"{space_count}d"

        # 3) 构造带占位符的log_msg，注意使用双花括号来保留 format() 的命名
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                f'[{{0:{space_fmt}}}/{len(iterable)}]',  # 这里 {{0:{space_fmt}}} =>  format(..., i) => i:3d
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                f'[{{0:{space_fmt}}}/{len(iterable)}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])

        MB = 1024.0 * 1024.0

        for obj in iterable:
            # 统计加载时间
            data_time.update(time.time() - end)

            yield obj

            # 统计处理时间
            iter_time.update(time.time() - end)

            # 每 print_freq 步 or 最后一轮，打印一次日志
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, eta=eta_string, meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB
                    ))
                else:
                    print(log_msg.format(
                        i, eta=eta_string, meters=str(self),
                        time=str(iter_time), data=str(data_time)
                    ))
            i += 1
            end = time.time()

        # 结尾打印耗时
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)")


def get_sha():
    """
    Optional: captures git commit info, can be useful for logging experiment versions.
    """
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()

    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        _run(['git', 'diff'])  # check changes
        diff_stat = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommitted changes" if diff_stat else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    return f"sha: {sha}, status: {diff}, branch: {branch}"


def collate_fn(batch):
    """
    Collate function that merges a list of samples to form a mini-batch of Tensor(s).
    Typically used in DataLoader.
    """
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)


def _max_by_axis(the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    """
    Converts a list of 3D tensors [C, H, W] to a single padded 4D tensor [B, C, H_max, W_max],
    along with a corresponding boolean mask for the padding regions.
    """
    if tensor_list[0].ndim == 3:
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # batch shape
        batch_shape = [len(tensor_list)] + max_size  # e.g. [B, C, H, W]
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)

        for i, img in enumerate(tensor_list):
            # C, H, W = img.shape
            tensor[i, :img.shape[0], :img.shape[1], :img.shape[2]] = img
            mask[i, :img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('nested_tensor_from_tensor_list: only 3D tensors are currently supported')

    return NestedTensor(tensor, mask)


class NestedTensor(object):
    """
    A wrapper of Tensors + their corresponding mask.
    Typically used for images with different shapes.
    """
    def __init__(self, tensors: Tensor, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device, non_blocking=False):
        cast_tensor = self.tensors.to(device, non_blocking=non_blocking)
        cast_mask = None
        if self.mask is not None:
            cast_mask = self.mask.to(device, non_blocking=non_blocking)
        return NestedTensor(cast_tensor, cast_mask)

    def record_stream(self, stream):
        self.tensors.record_stream(stream)
        if self.mask is not None:
            self.mask.record_stream(stream)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def setup_for_distributed(is_master):
    """
    Disables printing when not in master process.
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_local_size():
    if not is_dist_avail_and_initialized():
        return 1
    return int(os.environ.get('LOCAL_SIZE', 1))


def get_local_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return int(os.environ.get('LOCAL_RANK', 0))


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    """
    Initializes distributed training if environment variables are set.
    Otherwise falls back to single-process mode.
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.dist_url = 'env://'
        os.environ['LOCAL_SIZE'] = str(torch.cuda.device_count())
    elif 'SLURM_PROCID' in os.environ:
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(
            'scontrol show hostname {} | head -n1'.format(node_list))
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
        os.environ['LOCAL_SIZE'] = str(num_gpus)
        args.dist_url = 'env://'
        args.world_size = ntasks
        args.rank = proc_id
        args.gpu = proc_id % num_gpus
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    dist.barrier()
    setup_for_distributed(args.rank == 0)


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def interpolate(input: Tensor, size=None, scale_factor=None, mode="nearest", align_corners=None):
    """
    Simplified version for newer torchvision: just calls torch.nn.functional.interpolate
    without old-versions fallback.
    """
    return F.interpolate(input, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)


def get_total_grad_norm(parameters, norm_type=2):
    """
    Returns the norm of gradients for a list of parameters.
    """
    params = list(filter(lambda p: p.grad is not None, parameters))
    if len(params) == 0:
        return torch.tensor(0.)
    norm_type = float(norm_type)
    device = params[0].grad.device
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in params]),
        norm_type
    )
    return total_norm


def inverse_sigmoid(x, eps=1e-5):
    """
    Inverse of the sigmoid function: x -> log(x / (1 - x))
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)
