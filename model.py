from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, field_validator, model_validator


class ModelDimensions(BaseModel):
    vocab_size: int
    d_model: int
    block_size: int

    N: int
    head_size: int
    h: int

    dropout: float

    @field_validator("vocab_size", "d_model", "block_size", "N", "head_size", "h")
    def check_positive(cls, v):
        assert v > 0, "must be positive"
        return v

    @field_validator("dropout")
    def check_value_between_zero_and_one_inclusive(cls, v):
        assert 0.0 <= v <= 1.0, "must between between 0.0 and 1.0, inclusive"
        return v

    @model_validator(mode="after")
    def check_heads_dim(cls, m):
        assert m.h * m.head_size == m.d_model, "h * head_size must equal d_model"
        return m


class AttentionHead(nn.Module):
    def __init__(self, head_size: int, d_model: int, dropout: float, block_size: int):
        super().__init__()
        self._head_size = head_size

        self._query = nn.Linear(d_model, head_size, bias=False)
        self._key = nn.Linear(d_model, head_size, bias=False)
        self._value = nn.Linear(d_model, head_size, bias=False)

        self._dropout = nn.Dropout(dropout)

        self.register_buffer("_tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, t, _ = x.shape  # t <= block_size
        q = self._query(x)  # (b, t, d_model) @ (d_model, head_size) -> (b, t, head_size)
        k = self._key(x)  # (b, t, d_model) @ (d_model, head_size) -> (b, t, head_size)

        wei = q @ k.transpose(-2, -1) / (self._head_size ** 0.5)  # (b, t, head_size) @ (b, head_size, t) -> (b, t, t)
        wei = wei.masked_fill(self._tril[:t, :t] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self._dropout(wei)

        v = self._value(x)  # (b, t, d_model) @ (d_model, head_size) -> (b, t, head_size)
        return wei @ v  # (b, t, t) @ (b, t, head_size) -> (b, t, head_size)


class MultiHeadAttention(nn.Module):
    def __init__(self, h: int, head_size: int, d_model: int, dropout: float, block_size: int):
        super().__init__()
        self._attention_heads = nn.ModuleList([
            AttentionHead(head_size, d_model, dropout, block_size)
            for _ in range(h)
        ])
        self._proj = nn.Linear(h * head_size, d_model)

        self._dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([attention_head(x) for attention_head in self._attention_heads], dim=-1)
        out = self._proj(out)  # (b, t, h * head_size) @ (h * head_size, d_model) -> (b, t, d_model)
        out = self._dropout(out)
        return out


class MLP(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self._linear_1 = nn.Linear(d_model, 4 * d_model)
        self._linear_2 = nn.Linear(4 * d_model, d_model)
        self._relu = nn.ReLU()

        self._dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._linear_1(x)  # (b, t, d_model) @ (d_model, 4 * d_model) -> (b, t, 4 * d_model)
        out = self._relu(out)
        out = self._linear_2(out)  # (b, t, 4 * d_model) @ (b, 4 * d_model, d_model) -> (b, t, d_model)
        out = self._dropout(out)
        return out


class Block(nn.Module):
    def __init__(self, h: int, head_size: int, d_model: int, dropout: float, block_size: int):
        super().__init__()
        self._multi_head_attention = MultiHeadAttention(h, head_size, d_model, dropout, block_size)
        self._mlp = MLP(d_model, dropout)

        self._ln_1 = nn.LayerNorm(d_model)
        self._ln_2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x + self._multi_head_attention(self._ln_1(x))
        out = out + self._mlp(self._ln_2(out))
        return out


class GPT2(nn.Module):
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self._dims = dims

        self._token_embedding_table = nn.Embedding(dims.vocab_size, dims.d_model)
        self._position_embedding_table = nn.Embedding(dims.block_size, dims.d_model)

        self._blocks = nn.Sequential(*[
            Block(dims.h, dims.head_size, dims.d_model, dims.dropout, dims.block_size)
            for _ in range(dims.N)
        ])
        self._ln = nn.LayerNorm(dims.d_model)
        self._lm = nn.Linear(dims.d_model, dims.vocab_size)

    def forward(self, idx: torch.Tensor, inference: bool = False) -> torch.Tensor:
        _, t = idx.shape

        tok_emb = self._token_embedding_table(idx)
        pos_emb = self._position_embedding_table(torch.arange(t, device=idx.device))
        x = tok_emb + pos_emb

        out = self._blocks(x)
        out = self._ln(out)

        logits = self._lm(out) if not inference else self._lm(out[:, [-1], :])
        return logits

    def generate(self, idx: torch.Tensor, new_tokens: int):
        for _ in range(new_tokens):
            idx_context = idx[:, -self._dims.block_size:]
            logits = self(idx_context, inference=True)[:, -1, :]
            probs = F.softmax(logits, dim=-1)

            # sample next token
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
