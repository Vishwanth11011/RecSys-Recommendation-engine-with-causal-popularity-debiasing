import torch
import torch.nn as nn

class NCF(nn.Module):
    """
    Neural Collaborative Filtering (He et al., 2017).
    Combines Generalized Matrix Factorization (GMF) with MLP.
    GMF: element-wise product of user/item embeddings
    MLP: concatenated embeddings through deep layers
    NeuMF: combines both paths before final prediction
    """
    def __init__(self, n_users: int, n_movies: int, embed_dim: int = 64,
                 mlp_layers: list = [128, 64, 32]):
        super().__init__()
        # GMF embeddings
        self.gmf_user_emb  = nn.Embedding(n_users, embed_dim)
        self.gmf_movie_emb = nn.Embedding(n_movies, embed_dim)
        
        # MLP embeddings (separate embedding space)
        self.mlp_user_emb  = nn.Embedding(n_users, embed_dim)
        self.mlp_movie_emb = nn.Embedding(n_movies, embed_dim)
        
        # MLP layers
        mlp_input_dim = embed_dim * 2
        layers = []
        for out_dim in mlp_layers:
            layers += [
                nn.Linear(mlp_input_dim, out_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ]
            mlp_input_dim = out_dim
        self.mlp = nn.Sequential(*layers)
        
        # Final prediction layer
        self.predict_layer = nn.Linear(embed_dim + mlp_layers[-1], 1)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        for emb in [self.gmf_user_emb, self.gmf_movie_emb,
                    self.mlp_user_emb, self.mlp_movie_emb]:
            nn.init.normal_(emb.weight, std=0.01)

    def forward(self, user_ids, movie_ids):
        # Ensure user_ids and movie_ids are 1D Tensors
        user_ids = user_ids.long()
        movie_ids = movie_ids.long()
        
        # GMF path
        gmf_out = self.gmf_user_emb(user_ids) * self.gmf_movie_emb(movie_ids)
        
        # MLP path
        mlp_input = torch.cat([
            self.mlp_user_emb(user_ids),
            self.mlp_movie_emb(movie_ids)
        ], dim=-1)
        mlp_out = self.mlp(mlp_input)
        
        # Combine and predict
        combined = torch.cat([gmf_out, mlp_out], dim=-1)
        preds = self.sigmoid(self.predict_layer(combined))
        
        # Squeeze if output is multiple dimensions but handle single/batch cases
        if preds.dim() > 0:
            return preds.squeeze(-1)
        return preds
