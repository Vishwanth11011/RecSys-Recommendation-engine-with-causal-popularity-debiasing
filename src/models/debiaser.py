import numpy as np
import pandas as pd
import pickle
from pathlib import Path

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'IPSDebiaser':
            return IPSDebiaser
        return super().find_class(module, name)

class IPSDebiaser:
    """
    Inverse Propensity Scoring (IPS) to correct popularity bias.
    Propensity score P(observed | user, item) approximated by:
        p_i = (count of ratings for item i) / (max count across all items)
    Weighted loss: L = sum_i [ (1 / p_i) * loss(y_hat_i, y_i) ]
    """
    def __init__(self, clip_min: float = 0.01, clip_max: float = 1.0):
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.propensities = {}

    def fit(self, ratings: pd.DataFrame):
        # We can fit on either movie_id or movie_idx. Let's fit on both if available, or handle dynamically.
        # Counts based on movie_id
        if "movie_id" in ratings.columns:
            counts = ratings["movie_id"].value_counts()
            max_count = counts.max()
            raw_prop = counts / max_count
            clipped = raw_prop.clip(self.clip_min, self.clip_max)
            self.propensities_id = clipped.to_dict()
        else:
            self.propensities_id = {}

        # Counts based on movie_idx
        if "movie_idx" in ratings.columns:
            counts_idx = ratings["movie_idx"].value_counts()
            max_idx_count = counts_idx.max()
            raw_prop_idx = counts_idx / max_idx_count
            clipped_idx = raw_prop_idx.clip(self.clip_min, self.clip_max)
            self.propensities_idx = clipped_idx.to_dict()
        else:
            self.propensities_idx = {}

    def get_weights(self, movie_identifiers, is_index=True) -> np.ndarray:
        """Return IPS weights (1/propensity) for a batch of movie identifiers."""
        prop_dict = self.propensities_idx if is_index else self.propensities_id
        # Convert torch tensor if passed
        if hasattr(movie_identifiers, "tolist"):
            movie_identifiers = movie_identifiers.tolist()
            
        props = np.array([prop_dict.get(mid, self.clip_min) for mid in movie_identifiers])
        return 1.0 / props

    def weighted_bce_loss(self, preds, labels, movie_identifiers, is_index=True):
        """Weighted Binary Cross-Entropy."""
        import torch
        weights = torch.tensor(self.get_weights(movie_identifiers, is_index=is_index), dtype=torch.float32)
        weights = weights.to(preds.device)
        bce = torch.nn.functional.binary_cross_entropy(preds, labels.float(), reduction="none")
        return (bce * weights).mean()

    def save(self, path="saved_models/debiaser.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path="saved_models/debiaser.pkl"):
        with open(path, "rb") as f:
            return CustomUnpickler(f).load()
