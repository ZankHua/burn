# decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class BurnDecoder(nn.Module):
    def __init__(self, dec_layers, hidden_dim, nheads, dropout,
                 num_queries=100, num_classes=4, mask_size=(32,32)):
        """
        一个简化的 decoder：
          - 使用 nn.TransformerDecoder
          - 查询向量 query_embed
          - 分类头 class_embed
          - mask_embed -> 分割
        """
        super().__init__()
        # 构建 Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nheads, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=dec_layers)

        # 学习的查询向量
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # 分类 & 分割
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.mask_embed = nn.Linear(hidden_dim, mask_size[0]*mask_size[1])
        self.mask_size = mask_size

    def forward(self, memory):
        """
        memory: 已是三维 [S, B, hidden_dim] or e.g. [H*W, B, 256].
        查询 shape => [num_queries, B, hidden_dim]
        """
        seq_len, batch_size, d_model = memory.shape

        # 构建 queries
        queries = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)
        # queries => [num_queries, B, d_model]

        hs = self.transformer_decoder(queries, memory)   # => [num_queries, B, d_model]
        hs = hs.transpose(0,1)  # => [B, num_queries, d_model]

        # 分类
        pred_logits = self.class_embed(hs)  # => [B, num_queries, num_classes]

        # 分割
        mask_logits = self.mask_embed(hs)   # => [B, num_queries, H*W]
        H, W = self.mask_size
        mask_logits = mask_logits.view(batch_size, -1, H, W)  # => [B, num_queries, H, W]
        pred_masks = mask_logits.sigmoid()

        return {
            'pred_logits': pred_logits,
            'pred_masks': pred_masks
        }

def build_decoder(args):
    return BurnDecoder(
        dec_layers=args.dec_layers,
        hidden_dim=args.hidden_dim,
        nheads=args.nheads,
        dropout=args.dropout,
        num_queries=100,
        num_classes=4,
        mask_size=(32,32)
    )
