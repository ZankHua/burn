import torch.nn as nn

class BurnDecoder(nn.Module):
    def __init__(self, dec_layers, hidden_dim, nheads, dropout, num_queries=100, num_classes=4, mask_size=(32, 32)):
        """
        构造用于烧伤检测的 decoder 模块。

        参数:
          dec_layers: Transformer decoder 层数。
          hidden_dim: Transformer 隐藏层维度。
          nheads: Transformer 注意力头数。
          dropout: Transformer dropout 概率。
          num_queries: 查询向量数量，默认 100。
          num_classes: 分类分支输出的类别数。这里假设 4 类：背景、烧伤等级1、烧伤等级2、烧伤等级3。
          mask_size: 分割分支输出的 mask 分辨率（高度, 宽度），默认 (32, 32)。
        """
        super(BurnDecoder, self).__init__()
        # 构造 Transformer decoder 层，并堆叠成多层 Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nheads, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=dec_layers)
        # 学习的查询向量
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        # 分类头：输出每个查询对应的类别 logits（例如 0：背景，1、2、3：烧伤等级）
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        # 分割头：将 decoder 输出映射到固定大小的 mask，后续可上采样至原始分辨率
        self.mask_embed = nn.Linear(hidden_dim, mask_size[0] * mask_size[1])
        self.mask_size = mask_size

    def forward(self, memory):
        """
        前向传播

        参数:
          memory: 来自骨干网络的特征，形状应为 (S, batch, hidden_dim)，
                  其中 S 为序列长度（例如特征图展平后的长度），batch 为批大小。

        返回:
          一个字典，包含：
            - 'pred_logits': 分类分支的输出，形状 (batch, num_queries, num_classes)
            - 'pred_masks': 分割分支的输出，形状 (batch, num_queries, H, W)，H, W 为 mask_size 指定的分辨率
        """
        batch_size = memory.shape[1]
        # 准备查询向量，形状为 (num_queries, batch, hidden_dim)
        queries = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)
        # 使用 Transformer decoder，将查询与 memory 进行交互，输出 shape: (num_queries, batch, hidden_dim)
        hs = self.transformer_decoder(queries, memory)
        # 转置为 (batch, num_queries, hidden_dim)
        hs = hs.transpose(0, 1)
        # 分类预测：输出类别 logits
        pred_logits = self.class_embed(hs)
        # 分割预测：通过 mask_embed 得到展平的 mask，再 reshape 为 (H, W)
        mask_logits = self.mask_embed(hs)  # 形状: (batch, num_queries, H*W)
        H, W = self.mask_size
        mask_logits = mask_logits.view(batch_size, -1, H, W)
        # 使用 sigmoid 激活，将 mask 值归一化到 [0,1]
        pred_masks = mask_logits.sigmoid()
        return {'pred_logits': pred_logits, 'pred_masks': pred_masks}


def build_decoder(args):
    """
    根据传入的参数构造并返回一个用于烧伤检测的 decoder 实例。

    参数（来自 args）：
      - dec_layers: Transformer decoder 层数。
      - hidden_dim: Transformer 隐藏层维度。
      - nheads: 注意力头数。
      - dropout: dropout 概率。

    其他设置：
      - num_queries: 固定为 100。
      - num_classes: 固定为 4（背景 + 烧伤等级1,2,3）。
      - mask_size: 固定为 (32, 32) 分辨率的分割输出，可根据需求修改。

    返回:
      一个 BurnDecoder 实例。
    """
    num_queries = 100
    num_classes = 4  # 假设 0 为背景，其余 1,2,3 分别为烧伤等级
    mask_size = (32, 32)
    return BurnDecoder(args.dec_layers, args.hidden_dim, args.nheads, args.dropout,
                       num_queries=num_queries, num_classes=num_classes, mask_size=mask_size)
