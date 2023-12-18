"""
Create synthetic data used in the paper.

References
1. https://github.com/openai/grok/blob/main/grok/data.py
2. https://github.com/danielmamay/grokking/blob/main/grokking/data.py
"""

import torch
from torch.utils.data import TensorDataset, random_split

from model import Config

def get_raw_data(operation: str, cfg: Config):
    #! Only tested on the modular addition for now.
    assert operation == "+"

    x = torch.arange(cfg.mod)
    y = torch.arange(cfg.mod)
    x,y = torch.cartesian_prod(x, y).T

    # Now just calculate the answer for each pair
    labels = (x + y) % cfg.mod
    
    eq_token = cfg.mod
    op_token = cfg.mod+1

    eq = torch.ones_like(x) * eq_token
    op = torch.ones_like(x) * op_token

    # the input is like [x op y = answer]
    inputs = torch.stack([x, op, y, eq], dim=1)

    return inputs, labels

def get_dataset(cfg: Config):

    # other tokens will be taken by the number themselves
    inputs, labels = get_raw_data("+", cfg)
    dataset = TensorDataset(inputs, labels)

    train_size = int(cfg.split_size * len(dataset))
    valid_size = len(dataset) - train_size

    # for reproducing the results
    generator1 = torch.Generator().manual_seed(42)

    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size], generator=generator1)

    return train_dataset, valid_dataset