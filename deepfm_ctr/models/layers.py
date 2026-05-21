"""Shared building blocks for FM and DeepFM models.

FeaturesLinear    -- first-order (linear) term
FeaturesEmbedding -- shared embedding layer (per-field embeddings)
FactorizationMachine -- second-order pairwise interactions (O(nk))
MultiLayerPerceptron -- deep DNN component
"""

import torch
import torch.nn as nn


class FeaturesLinear(nn.Module):
    """First-order (linear) term: bias + sum of per-feature linear weights."""

    def __init__(self, field_dims, num_numerical):
        super().__init__()
        self.num_numerical = num_numerical
        self.num_categorical = len(field_dims)

        # Categorical: each field has its own EmbeddingBag (embedding dim = 1 → scalar weight per value)
        self.cat_embeddings = nn.ModuleList([
            nn.EmbeddingBag(dim, 1, mode="sum") for dim in field_dims
        ])
        # Numerical: one scalar weight per numerical field
        self.num_weights = nn.Parameter(torch.randn(num_numerical))
        self.bias = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        for emb in self.cat_embeddings:
            nn.init.xavier_uniform_(emb.weight)
        nn.init.normal_(self.num_weights, std=0.01)

    def forward(self, numerical, categorical):
        # Categorical linear terms
        cat_out = torch.stack([
            self.cat_embeddings[i](categorical[:, i].unsqueeze(1))
            for i in range(self.num_categorical)
        ], dim=1).sum(dim=1)  # (batch, 1)

        # Numerical linear terms
        num_out = (numerical * self.num_weights.unsqueeze(0)).sum(dim=1, keepdim=True)  # (batch, 1)

        return self.bias + cat_out + num_out


class FeaturesEmbedding(nn.Module):
    """Shared embedding layer: each field (cat + num) → embed_dim vector.

    Categorical: nn.Embedding(vocab_size, embed_dim) per field.
    Numerical: a single learnable vector per field, scaled by the feature value.
    """

    def __init__(self, field_dims, num_numerical, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_fields = len(field_dims) + num_numerical

        # Categorical embeddings
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(dim, embed_dim) for dim in field_dims
        ])
        # Numerical projections: one vector per numerical field
        self.num_projections = nn.Parameter(torch.randn(num_numerical, embed_dim))

        self._init_weights()

    def _init_weights(self):
        for emb in self.cat_embeddings:
            nn.init.xavier_uniform_(emb.weight)
        nn.init.xavier_uniform_(self.num_projections)

    def forward(self, numerical, categorical):
        # Categorical embeddings: (batch, num_cat, embed_dim)
        cat_embed = torch.stack([
            self.cat_embeddings[i](categorical[:, i])
            for i in range(len(self.cat_embeddings))
        ], dim=1)

        # Numerical embeddings: value * projection_vector → (batch, num_num, embed_dim)
        num_embed = numerical.unsqueeze(-1) * self.num_projections.unsqueeze(0)

        return torch.cat([num_embed, cat_embed], dim=1)  # (batch, total_fields, embed_dim)


class FactorizationMachine(nn.Module):
    """Efficient O(nk) second-order interaction:

    output = 0.5 * sum_k( (sum_i v_ik)^2 - sum_i(v_ik^2) )
    """

    def forward(self, embed):
        # embed: (batch, num_fields, embed_dim)
        sum_squared = embed.sum(dim=1).pow(2)     # (batch, embed_dim)
        square_sum = embed.pow(2).sum(dim=1)       # (batch, embed_dim)
        return 0.5 * (sum_squared - square_sum).sum(dim=1, keepdim=True)


class MultiLayerPerceptron(nn.Module):
    """Deep component: FC stack with BatchNorm, ReLU, Dropout."""

    def __init__(self, input_dim, hidden_dims, dropout, use_batch_norm=True):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Final output layer (no BN, no activation — raw logit)
        layers.append(nn.Linear(dims[-1], 1))

        self.mlp = nn.Sequential(*layers)

        # Initialize
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.mlp(x)
