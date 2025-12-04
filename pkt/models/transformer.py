import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from pkt.engine.registries import MODULES
from pkt.models.backbones import BACKBONE_REGISTRY


class Attention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        # q, k, v  [batch, head_num, seq_len, dim]
        head_dim = query.shape[-1]  # d_model
        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.matmul(query, key.transpose(-1, -2)) * scale  # [B, H, L, d_k] @ [B, H, d_k, S] -> [B, H, L, S]

        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask.to(dtype=torch.bool)
            neg_inf = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~mask, neg_inf)

        attention = F.softmax(scores, dim=-1)  # 表示沿着最后一维度做softmax，跨列
        attention = self.dropout(attention)
        output = torch.matmul(attention, value)  # [B, H, L, S] @ [B, H, S, d_k] -> [B, H, L, d_k]

        return output


def _build_attn_mask(x_mask, source_mask, N, L, S, device):
    if x_mask is None and source_mask is None:
        return None

    if x_mask is None:
        x_mask = torch.ones(N, L, dtype=torch.bool, device=device)
    elif x_mask.dtype != torch.bool:
        x_mask = x_mask.to(torch.bool)
    if source_mask is None:
        source_mask = torch.ones(N, S, dtype=torch.bool, device=device)
    elif source_mask.dtype != torch.bool:
        source_mask = source_mask.to(torch.bool)

    # x_mask: [N, L] -> [N, 1, L, 1]
    # source_mask: [N, S] -> [N, 1, 1, S]
    # 广播后得到 attn_mask: [N, 1, L, S]
    attn_mask = x_mask.unsqueeze(-1) & source_mask.unsqueeze(1)
    return attn_mask.unsqueeze(1)  # 扩展head维度 -> [N, 1, L, S]


class TransformerLayer(nn.Module):
    def __init__(self, d_model=128, heads_num=12, dropout=0.1):
        super(TransformerLayer, self).__init__()
        if d_model % heads_num != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by heads_num ({heads_num})")
        self.dim = d_model // heads_num
        self.head_num = heads_num
        self.d_model = d_model

        self.query = nn.Linear(self.d_model, self.d_model)
        self.key = nn.Linear(self.d_model, self.d_model)
        self.value = nn.Linear(self.d_model, self.d_model)
        self.merge_layer = nn.Linear(self.d_model, self.d_model)

        ff_dim = self.d_model * 4
        self.mlp = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, self.d_model),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.attn = Attention(dropout)

    def forward(self, x, source=None, x_pos=None, source_pos=None, x_mask=None, source_mask=None):
        if source is None:
            source = x
            source_pos = x_pos
            source_mask = x_mask

        # 位置编码
        query_input = x if x_pos is None else x + x_pos
        key_input = source if source_pos is None else source + source_pos
        value_input = source

        B, L, _ = query_input.shape  # [batch, seq_len, d_model]
        S = key_input.size(1)
        device = query_input.device

        # 线性映射
        q = self.query(query_input)  # [B, L, d_model]
        k = self.key(key_input)  # [B, S, d_model]
        v = self.value(value_input)  # [B, S, d_model]

        # 拆分多头
        q = q.view(B, L, self.head_num, self.dim).transpose(1, 2)  # [B, H, L, d_k]
        k = k.view(B, S, self.head_num, self.dim).transpose(1, 2)  # [B, H, S, d_k]
        v = v.view(B, S, self.head_num, self.dim).transpose(1, 2)  # [B, H, S, d_k]

        attn_mask = _build_attn_mask(x_mask, source_mask, B, L, S, device)
        message = self.attn(q, k, v, mask=attn_mask) # [B, H, L, d_k]

        # 合并多头
        message = message.transpose(1, 2).contiguous().view(B, L, self.d_model)
        message = self.merge_layer(message)
        message = self.dropout1(message)

        # Norm FFN
        x = self.norm1(x + message)
        ffn_output = self.mlp(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        return x


@MODULES.register("Transformer")
class Transformer(nn.Module):
    def __init__(self, d_model=128, heads_num=12, layers_num=2, layer_names=None, dropout=0.1):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.heads_num = heads_num
        self.layers_num = layers_num
        self.layer_names = self._validate_layer_names(layer_names, layers_num)
        self.layers = nn.ModuleList([
            TransformerLayer(d_model=d_model, heads_num=heads_num, dropout=dropout) for _ in range(layers_num)
        ])

    @staticmethod
    def _validate_layer_names(layer_names, layers_num):
        if layer_names is None:
            layer_names = ["self"] * layers_num
        if len(layer_names) != layers_num:
            raise ValueError(f"layer_names 长度({len(layer_names)})必须等于 layers_num ({layers_num})")
        invalid = [name for name in layer_names if name not in {"self", "cross"}]
        if invalid:
            raise ValueError(f"存在非法层类型: {invalid}，只允许 'self' 或 'cross'")
        return list(layer_names)

    def forward(self, x, source=None, x_pos=None, source_pos=None, x_mask=None, source_mask=None):
        if source is None:
            source = x
            if source_pos is None:
                source_pos = x_pos
            if source_mask is None:
                source_mask = x_mask

        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                x = layer(x, None, x_pos, None, x_mask, None)
            elif name == 'cross':
                x = layer(x, source, x_pos, source_pos, x_mask, source_mask)
            else:
                raise ValueError(f"未知的层类型: {name}. 只应为 'self' 或 'cross'。")
        return x


@BACKBONE_REGISTRY.register("transformer_encoder")
class TransformerBackbone(nn.Module):
    """Backbone wrapper around :class:`Transformer` for configuration驱动训练。"""

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        d_model: int = 128,
        heads_num: int = 8,
        layers_num: int = 2,
        dropout: float = 0.1,
        layer_types: Sequence[str] | None = None,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        if seq_len <= 0:
            raise ValueError("seq_len 必须为正整数")
        if input_dim % seq_len != 0:
            raise ValueError("input_dim 必须能被 seq_len 整除以便重排为序列")
        if pooling not in {"mean", "cls"}:
            raise ValueError("pooling 仅支持 'mean' 或 'cls'")

        token_dim = input_dim // seq_len
        self.seq_len = seq_len
        self.token_dim = token_dim
        self.input_dim = input_dim
        self.pooling = pooling
        self.use_cls_token = pooling == "cls"

        total_tokens = seq_len + (1 if self.use_cls_token else 0)

        self.token_proj = nn.Linear(token_dim, d_model)
        self.transformer = Transformer(
            d_model=d_model,
            heads_num=heads_num,
            layers_num=layers_num,
            layer_names=layer_types,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(d_model)
        self.output_dim = d_model

        self.pos_embedding = nn.Parameter(torch.zeros(1, total_tokens, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.register_parameter("cls_token", None)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dim() != 2 or inputs.size(1) != self.input_dim:
            raise ValueError(
                f"期望输入形状为 [batch, {self.input_dim}], 实际获得 {tuple(inputs.shape)}"
            )

        batch_size = inputs.size(0)
        tokens = inputs.view(batch_size, self.seq_len, self.token_dim)
        tokens = self.token_proj(tokens)

        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            tokens = torch.cat([cls_tokens, tokens], dim=1)

        pos = self.pos_embedding[:, : tokens.size(1), :].expand(batch_size, -1, -1)
        encoded = self.transformer(tokens, x_pos=pos)
        encoded = self.norm(encoded)

        if self.use_cls_token:
            return encoded[:, 0, :]
        return encoded.mean(dim=1)
