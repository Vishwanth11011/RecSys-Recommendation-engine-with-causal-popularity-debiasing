import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Ensure src is in python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.models.svd_model import SVDRecommender
from src.models.ncf_model import NCF
from src.models.two_tower import TwoTowerModel
from src.models.debiaser import IPSDebiaser

from src.data.loader import load_ratings, load_movies

class UnifiedRecommender:
    def __init__(self):
        self.device = torch.device("cpu")
        self.base_path = Path(__file__).resolve().parent.parent.parent
        self.models_dir = self.base_path / "saved_models"
        self.data_dir = self.base_path / "data" / "processed"
        
        self.user_enc = None
        self.movie_enc = None
        self.movie_metadata = {}
        
        # Models
        self.svd_model = None
        self.ncf_model = None
        self.ncf_debiased_model = None
        self.two_tower_model = None
        self.debiaser = None
        
        # Two-Tower features
        self.user_features_matrix = None
        self.movie_features_matrix = None
        
    def load_all_models(self):
        print("UnifiedRecommender: Loading encoders...")
        with open(self.data_dir / "user_encoder.pkl", "rb") as f:
            self.user_enc = pickle.load(f)
        with open(self.data_dir / "movie_encoder.pkl", "rb") as f:
            self.movie_enc = pickle.load(f)
            
        print("UnifiedRecommender: Loading movies metadata...")
        movies_df = load_movies()
        self.movie_metadata = {}
        for row in movies_df.itertuples():
            self.movie_metadata[row.movie_id] = {
                "title": row.title,
                "genres": row.genre_list if isinstance(row.genre_list, list) else [row.genres],
                "year": float(row.year) if not pd.isna(row.year) else None
            }
            
        n_users = len(self.user_enc.classes_)
        n_movies = len(self.movie_enc.classes_)
        
        # Load SVD
        print("UnifiedRecommender: Loading SVD model...")
        self.svd_model = SVDRecommender()
        self.svd_model.load(self.models_dir / "svd_model.pkl")
        
        # Load IPS Debiaser
        print("UnifiedRecommender: Loading IPS debiaser...")
        if (self.models_dir / "debiaser.pkl").exists():
            self.debiaser = IPSDebiaser.load(self.models_dir / "debiaser.pkl")
        else:
            self.debiaser = IPSDebiaser()
            
        # Load NCF Standard
        print("UnifiedRecommender: Loading standard NCF...")
        self.ncf_model = NCF(n_users, n_movies, embed_dim=64).to(self.device)
        self.ncf_model.load_state_dict(torch.load(self.models_dir / "ncf_model.pt", map_location=self.device))
        self.ncf_model.eval()
        
        # Load NCF Debiased (trained with IPS)
        print("UnifiedRecommender: Loading debiased NCF...")
        self.ncf_debiased_model = NCF(n_users, n_movies, embed_dim=64).to(self.device)
        if (self.models_dir / "ncf_debiased_model.pt").exists():
            self.ncf_debiased_model.load_state_dict(torch.load(self.models_dir / "ncf_debiased_model.pt", map_location=self.device))
        else:
            # Fallback to standard NCF if debiased model is not found
            self.ncf_debiased_model.load_state_dict(torch.load(self.models_dir / "ncf_model.pt", map_location=self.device))
        self.ncf_debiased_model.eval()
        
        # Load Two-Tower
        print("UnifiedRecommender: Loading Two-Tower model...")
        self.user_features_matrix = np.load(self.data_dir / "user_features.npy")
        self.movie_features_matrix = np.load(self.data_dir / "movie_features.npy")
        
        user_feat_dim = self.user_features_matrix.shape[1]
        movie_feat_dim = self.movie_features_matrix.shape[1]
        
        self.two_tower_model = TwoTowerModel(user_feat_dim, movie_feat_dim, embed_dim=64).to(self.device)
        self.two_tower_model.load_state_dict(torch.load(self.models_dir / "two_tower.pt", map_location=self.device))
        self.two_tower_model.eval()
        
        print("UnifiedRecommender: Loading user history...")
        ratings_df = load_ratings()
        self.user_history = ratings_df.groupby("user_id")["movie_id"].apply(list).to_dict()
        self.user_history_set = {uid: set(mids) for uid, mids in self.user_history.items()}
        self.user_ratings = ratings_df.groupby("user_id")[["movie_id", "rating"]].apply(lambda x: list(zip(x["movie_id"], x["rating"]))).to_dict()
        
        print("UnifiedRecommender: Loading complete!")
        
    def recommend(self, user_id: int, candidate_movie_ids: list, model_type: str = "svd", debias: bool = False) -> list:
        """
        Produce a list of (movie_id, score) predictions sorted in descending order of relevance.
        """
        # Verify user_id and movies exist in encoders
        if user_id not in self.user_enc.classes_:
            # Cold start fallback: return default candidate order or random
            return [(mid, 0.0) for mid in candidate_movie_ids]
            
        user_idx = self.user_enc.transform([user_id])[0]
        
        # Keep only valid candidates
        valid_movie_ids = [mid for mid in candidate_movie_ids if mid in self.movie_enc.classes_]
        if not valid_movie_ids:
            return []
            
        movie_idxs = self.movie_enc.transform(valid_movie_ids)
        
        if model_type == "svd":
            # If debias is requested for SVD, we multiply the prediction by the IPS score
            preds = []
            for mid in valid_movie_ids:
                raw_pred = self.svd_model.model.predict(user_id, mid).est
                if debias and self.debiaser:
                    # Propensity weighting corrects the predicted rating towards long-tail
                    weight = self.debiaser.get_weights([mid], is_index=False)[0]
                    # To prevent wild swings, we clip or apply a soft weighting
                    score = raw_pred * np.log1p(weight)
                else:
                    score = raw_pred
                preds.append((mid, float(score)))
            return sorted(preds, key=lambda x: x[1], reverse=True)
            
        elif model_type == "ncf":
            # Pick standard NCF or Debiased NCF trained with IPS loss
            model_to_use = self.ncf_debiased_model if debias else self.ncf_model
            
            user_tensor = torch.tensor([user_idx] * len(movie_idxs), dtype=torch.long, device=self.device)
            movie_tensor = torch.tensor(movie_idxs, dtype=torch.long, device=self.device)
            
            with torch.no_grad():
                preds = model_to_use(user_tensor, movie_tensor)
                scores = preds.cpu().numpy()
                
            results = [(mid, float(score)) for mid, score in zip(valid_movie_ids, scores)]
            return sorted(results, key=lambda x: x[1], reverse=True)
            
        elif model_type == "two_tower":
            # For Two-Tower, we do matrix vector multiplication for fast retrieval
            user_feats = self.user_features_matrix[user_idx]
            user_feats_tensor = torch.tensor(user_feats, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                user_emb = self.two_tower_model.get_user_embedding(user_feats_tensor) # (1, embed_dim)
                
                movie_feats = self.movie_features_matrix[movie_idxs]
                movie_feats_tensor = torch.tensor(movie_feats, dtype=torch.float32).to(self.device)
                movie_embs = self.two_tower_model.get_movie_embeddings(movie_feats_tensor) # (N, embed_dim)
                
                # Cosine similarities
                similarities = torch.matmul(movie_embs, user_emb.T).squeeze(-1) # (N,)
                scores = similarities.cpu().numpy()
                
            results = []
            for mid, idx, score in zip(valid_movie_ids, movie_idxs, scores):
                if debias and self.debiaser:
                    # Apply propensity score adjustment to similarities
                    weight = self.debiaser.get_weights([idx], is_index=True)[0]
                    # Soft scale: similarity ranges from -1 to 1. Map to [0, 1] then weight
                    prob = (score + 1.0) / 2.0
                    debiased_score = prob * np.log1p(weight)
                    results.append((mid, float(debiased_score)))
                else:
                    results.append((mid, float(score)))
            return sorted(results, key=lambda x: x[1], reverse=True)
            
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
