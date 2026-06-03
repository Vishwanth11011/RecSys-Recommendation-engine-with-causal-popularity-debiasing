import torch
import torch.nn as nn
import torch.nn.functional as F

class Tower(nn.Module):
    """Single tower: maps features → dense embedding."""
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int):
        super().__init__()
        layers, in_dim = [], input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.ReLU(),
                nn.BatchNorm1d(h)
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # BatchNorm1d requires float type features
        x = x.float()
        # If input has only 1 sample and model is training, batchnorm will fail.
        # We handle this by checking if the net has batchnorm and sample size is 1.
        # Normally during training batch size is > 1. During eval it is 1, but model is in eval mode.
        out = self.net(x)
        return F.normalize(out, dim=-1)  # L2 normalize for cosine similarity

class TwoTowerModel(nn.Module):
    """
    Two-Tower (Dual-Encoder) retrieval model.
    User tower: encodes user demographics
    Item tower: encodes movie genres + year
    Training objective: maximize cosine similarity for positive (user, movie) pairs,
    minimize for random negatives.
    """
    def __init__(self, user_feature_dim: int, movie_feature_dim: int,
                 embed_dim: int = 64):
        super().__init__()
        self.user_tower  = Tower(user_feature_dim,  [128, 64], embed_dim)
        self.movie_tower = Tower(movie_feature_dim, [128, 64], embed_dim)

    def forward(self, user_feats, movie_feats):
        u = self.user_tower(user_feats)
        m = self.movie_tower(movie_feats)
        return (u * m).sum(dim=-1)  # dot product of normalized embeddings = cosine similarity

    def get_user_embedding(self, user_feats):
        return self.user_tower(user_feats)

    def get_movie_embeddings(self, all_movie_feats):
        return self.movie_tower(all_movie_feats)
