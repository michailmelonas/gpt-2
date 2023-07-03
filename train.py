import requests

import torch
import torch.nn as nn

from model import GPT2, ModelDimensions
from utils import encode, get_batch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# download Shakespeare collection
r = requests.get("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
text = r.text

# define vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)


# mapping from string to list of integers
itos = {i: ch for i, ch in enumerate(chars)}
# mapping from list of integers to string
stoi = {ch: i for i, ch in enumerate(chars)}


# create data object
data = torch.tensor(encode(text, stoi), dtype=torch.long)
# train and validation split
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


TRAIN_ITERS, EVAL_ITERS = 5000, 200
EVAL_INTERVAL = 500
LR = 3e-4
BATCH_SIZE = 64
BLOCK_SIZE = 256

dims = ModelDimensions(**{
    "vocab_size": vocab_size,
    "d_model": 384,
    "block_size": BLOCK_SIZE,
    "N": 6,
    "head_size": 64,
    "h": 6,
    "dropout": 0.2,
})
net = GPT2(dims).to(DEVICE)

optimizer = torch.optim.AdamW(net.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

for i in range(TRAIN_ITERS):
    if i % EVAL_INTERVAL == 0:
        net.eval()
        train_losses, val_losses = torch.zeros(EVAL_ITERS), torch.zeros(EVAL_ITERS)
        for j in range(EVAL_ITERS):
            # train eval
            x, y = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE, DEVICE)
            train_losses[j] = criterion(
                net(x).view(BATCH_SIZE * BLOCK_SIZE, vocab_size),
                y.view(BATCH_SIZE * BLOCK_SIZE)
            ).item()
            # val eval
            x, y = get_batch(val_data, BATCH_SIZE, BLOCK_SIZE, DEVICE)
            val_losses[j] = criterion(
                net(x).view(BATCH_SIZE * BLOCK_SIZE, vocab_size),
                y.view(BATCH_SIZE * BLOCK_SIZE)
            ).item()

        print(f"step {i}: train loss is {train_losses.mean()}, val loss is {val_losses.mean()}")

    x, y = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE, DEVICE)
    net.train()
    loss = criterion(net(x).view(BATCH_SIZE * BLOCK_SIZE, vocab_size), y.view(BATCH_SIZE * BLOCK_SIZE))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
