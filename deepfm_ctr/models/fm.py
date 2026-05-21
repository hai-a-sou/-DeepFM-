"""Factorization Machine (FM) model — baseline for pairwise feature interactions."""

import torch
import torch.nn as nn
from .layers import FeaturesLinear, FeaturesEmbedding, FactorizationMachine


class FactorizationMachineModel(nn.Module):
    """FM: linear (1st-order) + pairwise (2nd-order) interactions.

    Returns raw logits. Apply sigmoid for probabilities.
    """

    def __init__(self, field_dims, num_numerical, embed_dim):
        super().__init__()
        self.linear = FeaturesLinear(field_dims, num_numerical)
        self.embedding = FeaturesEmbedding(field_dims, num_numerical, embed_dim)
        self.fm = FactorizationMachine()

    def forward(self, numerical, categorical):
        linear_out = self.linear(numerical, categorical)            # (batch, 1)
        embed = self.embedding(numerical, categorical)              # (batch, fields, embed_dim)
        fm_out = self.fm(embed)                                     # (batch, 1)
        return linear_out + fm_out                                  # logits
