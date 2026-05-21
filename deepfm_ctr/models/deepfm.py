"""DeepFM: Combines FM (1st + 2nd order) with a DNN (high-order) using shared embeddings.

Returns raw logits. Apply sigmoid for probabilities.
"""

import torch
import torch.nn as nn
from .layers import FeaturesLinear, FeaturesEmbedding, FactorizationMachine, MultiLayerPerceptron


class DeepFM(nn.Module):
    """DeepFM: y = w0 + <w,x> + FM(x) + DNN(x)

    The embedding layer is shared between the FM second-order component
    and the deep DNN component.
    """

    def __init__(self, field_dims, num_numerical, embed_dim, mlp_dims, dropout, use_batch_norm=True):
        super().__init__()
        total_fields = len(field_dims) + num_numerical

        self.linear = FeaturesLinear(field_dims, num_numerical)
        self.embedding = FeaturesEmbedding(field_dims, num_numerical, embed_dim)
        self.fm = FactorizationMachine()
        self.mlp = MultiLayerPerceptron(
            input_dim=total_fields * embed_dim,
            hidden_dims=mlp_dims,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
        )

    def forward(self, numerical, categorical):
        # First-order
        linear_out = self.linear(numerical, categorical)           # (batch, 1)

        # Shared embeddings
        embed = self.embedding(numerical, categorical)             # (batch, fields, embed_dim)

        # Second-order FM
        fm_out = self.fm(embed)                                    # (batch, 1)

        # High-order DNN
        flat = embed.view(embed.size(0), -1)                       # (batch, fields * embed_dim)
        deep_out = self.mlp(flat)                                  # (batch, 1)

        return linear_out + fm_out + deep_out                      # logits
