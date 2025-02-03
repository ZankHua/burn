"""
modifications_explanation.py

本文件对烧伤检测项目中各关键文件的修改做详细说明，包括新增功能、修改功能及删减内容。

1. main.py:
   - 修改数据集加载部分，替换 Dummy 数据集为真实 COCO 数据集加载器：
       from datasets.coco import build as build_dataset
   - 增加命令行参数：--masks, --cache_mode, --local_rank, --local_size 用于 COCO 数据集和分布式训练。
   - 调整优化器参数组设置，使得不同参数组使用不同学习率（骨干 vs. 线性投影）。
   - 增加 checkpoint 恢复逻辑，加载优化器和调度器状态。

2. engine.py:
   - 在 train_one_epoch() 中，增加数据预取（data_prefetcher），前向传播、损失计算、反向传播、梯度裁剪、参数更新以及日志记录。
   - 增加 cust_sumarize() 辅助函数，用于格式化 COCO 评估指标输出（每个类别的 AP）。
   - 在 evaluate() 中，将模型输出转换为 COCO 格式，并调用 COCO 评估器累积指标，最终输出和保存评估结果。

3. models/decoder.py:
   - 新增 BurnDecoder 类：支持两个输出分支：
         - 分类分支：输出 4 类 logits（无烧伤+ 烧伤等级1、2、3）。
         - 分割分支：输出固定分辨率的 mask（例如 32×32），通过 sigmoid 激活归一化。
   - 新增 build_decoder(args) 函数，依据命令行参数构造并返回 BurnDecoder 实例。

4. models/burnet.py:
   - 修改类别数设置为 4，适应烧伤检测任务（背景 + 烧伤等级1、2、3）。
   - 集成 decoder 输出，支持同时进行分类和分割任务。
   - 在 SetCriterion 中增加 loss_masks() 函数，采用 BCELoss 计算 mask 分割损失，与分类损失一起构成总损失。
   - 扩展 PostProcess，除了处理边界框（bbox），也返回分割 mask 输出（供后续上采样处理）。
   - 在 build() 函数中，整合骨干、decoder、检测器模型 BurnNet，并构造损失函数、后处理模块。

5. 其他：
   - swin_transformer/config.py、models/backbone.py 等文件保持原样或仅需修改 YAML 文件以适配烧伤检测任务的参数要求。
   - datasets/coco.py：使用现有的 COCO 数据集加载器，数据预处理部分（transforms）可根据需求调整。
   - util/misc.py：包含辅助工具函数，用于日志记录、梯度统计、进程同步等，无需大幅修改。

"""

