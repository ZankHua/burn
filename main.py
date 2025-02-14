import argparse
import datetime
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

import util.misc as utils
from datasets import get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
# 使用 COCO 数据集加载器，从 datasets/coco.py 中导入 build 函数并重命名为 build_dataset
from datasets.coco import build as build_dataset


import argparse

import argparse

import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('Burning Detector', add_help=False)

    # Training parameters
    parser.add_argument('--lr', default=2e-4, type=float, help="Main learning rate for training.")
    parser.add_argument('--batch_size', default=2, type=int, help="Batch size for training.")
    parser.add_argument('--epochs', default=50, type=int, help="Number of training epochs.")
    parser.add_argument('--lr_drop', default=40, type=int, help="Epoch when learning rate drops.")
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help="Maximum norm for gradient clipping.")
    parser.add_argument('--weight_decay', default=1e-4, type=float, help="L2 weight decay for regularization.")

    # Backbone parameters
    parser.add_argument('--backbone', default='resnet50', type=str,
                        choices=['resnet50', 'resnet101', 'swin_t', 'swin_s', 'vit_b'],
                        help="Choose the backbone network: resnet50, resnet101, swin_t, swin_s, vit_b.")
    parser.add_argument('--dilation', action='store_true',
                        help="Use dilation instead of stride in the last ResNet block (DC5).")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding used on image features (sine or learned).")
    parser.add_argument('--num_feature_levels', default=4, type=int,
                        help="Number of feature levels used in the backbone.")

    # Options for backbone training
    parser.add_argument('--backbone_from_scratch', default=False, action='store_true',
                        help="Train the backbone from scratch instead of using pretrained weights.")
    parser.add_argument('--finetune_early_layers', default=False, action='store_true',
                        help="Fine-tune the early layers of the backbone.")
    parser.add_argument('--scrl_pretrained_path', default='', type=str,
                        help="Path to additional pretrained model for the backbone.")

    # Transformer parameters
    parser.add_argument('--enc_layers', default=6, type=int, help="Number of Transformer encoder layers.")
    parser.add_argument('--dec_layers', default=6, type=int, help="Number of Transformer decoder layers.")
    parser.add_argument('--hidden_dim', default=256, type=int, help="Hidden dimension size in the Transformer.")
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout rate in the Transformer layers.")
    parser.add_argument('--nheads', default=8, type=int, help="Number of attention heads in the Transformer.")

    # Optimizer selection
    parser.add_argument('--optimizer', default='adamw', type=str, choices=['adamw', 'sgd'],
                        help="Choose the optimizer: adamw or sgd.")
    parser.add_argument('--sgd', action='store_true', help="Use SGD instead of AdamW.")

    # Learning rate scheduler selection
    parser.add_argument('--scheduler', default='step', type=str, choices=['step', 'cosine'],
                        help="Choose the learning rate scheduler: StepLR or CosineAnnealingLR.")

    # Enable AMP training
    parser.add_argument('--amp', action='store_true', help="Enable mixed-precision training for faster performance.")

    # Dataset parameters
    parser.add_argument('--dataset_file', default='coco', type=str, help="Dataset type (COCO).")
    parser.add_argument('--coco_path', default='./data/coco', type=str, help="Path to the COCO dataset.")

    # Device settings
    parser.add_argument('--device', default='mps', help="Device for training (cuda, mps or cpu).")
    parser.add_argument('--seed', default=42, type=int, help="Random seed for reproducibility.")
    parser.add_argument('--resume', default='', help="Path to checkpoint for resuming training.")
    parser.add_argument('--output_dir', default='./output', help="Path to save the trained model and logs.")

    # Additional parameters for parameter groups
    parser.add_argument('--lr_backbone_names', default=["backbone"], nargs='+',
                        help="Keywords for backbone parameters for learning rate.")
    parser.add_argument('--lr_linear_proj_names', default=["linear_proj"], nargs='+',
                        help="Keywords for linear projection layers for learning rate.")
    parser.add_argument('--lr_backbone', default=2e-5, type=float,
                        help="Learning rate for backbone parameters.")
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float,
                        help="Learning rate multiplier for linear projection layers.")
    parser.add_argument('--num_workers', default=4, type=int, help="Number of DataLoader workers.")

    # For evaluation mode
    parser.add_argument('--eval', action='store_true', help="Only perform evaluation.")
    parser.add_argument('--override_resumed_lr_drop', action='store_true',
                        help="Override resumed learning rate drop.")
    parser.add_argument('--start_epoch', default=0, type=int, help="Starting epoch for training.")

    # Additional parameters for COCO dataset loading
    parser.add_argument('--masks', action='store_true', help="Return masks in the dataset.")
    parser.add_argument('--cache_mode', action='store_true', help="Cache images in memory for faster loading.")
    parser.add_argument('--local_rank', default=0, type=int, help="Local rank for distributed training.")
    parser.add_argument('--local_size', default=1, type=int, help="Local size for distributed training.")

    # New: Loss weights and focal_alpha parameters (for burn detection task)
    parser.add_argument('--cls_loss_coef', default=1.0, type=float, help="Weight for classification loss.")
    parser.add_argument('--mask_loss_coef', default=1.0, type=float, help="Weight for mask loss.")
    parser.add_argument('--focal_alpha', default=0.25, type=float, help="Focal loss alpha parameter.")

    # New: Auxiliary loss switch (default enabled)
    parser.add_argument('--aux_loss', dest='aux_loss', action='store_true', help="Enable auxiliary loss (default: True)")
    parser.add_argument('--no-aux_loss', dest='aux_loss', action='store_false', help="Disable auxiliary loss")
    parser.set_defaults(aux_loss=True)

    # New: Iterative box refinement switch (default disabled)
    parser.add_argument('--with_box_refine', action='store_true',
                        help="Enable iterative box refinement in the model (default: False)")

    return parser



def main(args):
    device = torch.device(args.device)
    # 设置随机种子，确保结果可复现
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # 构建模型、损失函数和后处理模块
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    model_without_ddp = model  # 如果使用分布式训练，请在此处包装成 DDP 模型
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters:", n_parameters)

    # 构建 COCO 数据集（真实数据加载）
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                   pin_memory=True)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=True)

    # 调试用：打印所有参数名称
    def match_name_keywords(n, name_keywords):
        return any(keyword in n for keyword in name_keywords)

    for n, p in model_without_ddp.named_parameters():
        print(n)

    # 根据不同参数组设置不同学习率
    param_dicts = [
        {
            "params": [p for n, p in model_without_ddp.named_parameters()
                       if not match_name_keywords(n, args.lr_backbone_names)
                       and not match_name_keywords(n, args.lr_linear_proj_names)
                       and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters()
                       if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters()
                       if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]

    # 选择优化器
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    base_ds = get_coco_api_from_dataset(dataset_val)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 如果指定了 checkpoint，则恢复训练
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if missing_keys:
            print("Missing Keys:", missing_keys)
        if unexpected_keys:
            print("Unexpected Keys:", unexpected_keys)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            print("Optimizer param groups:", optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print("Warning: Overriding lr_drop in resumed lr_scheduler with args.lr_drop")
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = [group['initial_lr'] for group in optimizer.param_groups]
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1
        if not args.eval:
            test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                                  data_loader_val, base_ds, device, args.output_dir)

    # 如果指定为评估模式，则仅执行评估
    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    # 开始训练循环
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 5 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    print("Training time", str(datetime.timedelta(seconds=int(total_time))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
