from surprise import SVD, Dataset, Reader
import pandas as pd
import pickle
from pathlib import Path

class SVDRecommender:
    def __init__(self, n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02):
        self.model = SVD(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr_all,
            reg_all=reg_all,
            biased=True   # include user/item bias terms
        )
        self.trainset = None

    def fit(self, ratings_df: pd.DataFrame):
        # Explicit rating scale is 1 to 5
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(ratings_df[["user_id", "movie_id", "rating"]], reader)
        trainset = data.build_full_trainset()
        self.model.fit(trainset)
        self.trainset = trainset

    def predict(self, user_id: int, movie_ids: list) -> list:
        """Returns list of (movie_id, predicted_rating) sorted by score descending."""
        preds = [(mid, self.model.predict(user_id, mid).est) for mid in movie_ids]
        return sorted(preds, key=lambda x: x[1], reverse=True)

    def save(self, path="saved_models/svd_model.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path="saved_models/svd_model.pkl"):
        with open(path, "rb") as f:
            self.model = pickle.load(f)
