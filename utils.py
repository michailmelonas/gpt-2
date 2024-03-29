from typing import List, Tuple

import torch

from model import GPT2


def encode(s: str, stoi: dict) -> List[int]:
    return [stoi[ch] for ch in s]


def decode(l: List[int], itos: dict) -> str:
    return "".join([itos[i] for i in l])


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i: i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix]).to(device)
    return x, y

