import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

def build_encoders(ratings: pd.DataFrame):
    """
    Map raw user/movie IDs to contiguous 0-indexed integers.
    Required for embedding lookup tables in PyTorch.
    """
    user_enc = LabelEncoder()
    movie_enc = LabelEncoder()
    ratings = ratings.copy()
    ratings["user_idx"] = user_enc.fit_transform(ratings["user_id"])
    ratings["movie_idx"] = movie_enc.fit_transform(ratings["movie_id"])
    
    # Create directory if it does not exist
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save encoders for inference
    with open(PROCESSED_DIR / "user_encoder.pkl", "wb") as f:
        pickle.dump(user_enc, f)
    with open(PROCESSED_DIR / "movie_encoder.pkl", "wb") as f:
        pickle.dump(movie_enc, f)
        
    return ratings, user_enc, movie_enc

def split_leave_one_out(ratings: pd.DataFrame):
    """
    Standard evaluation protocol for recommendation:
    For each user, hold out their MOST RECENT interaction as test,
    second most recent as validation, rest as train.
    This simulates predicting what a user will interact with next.
    """
    ratings = ratings.sort_values(["user_id", "timestamp"])
    test = ratings.groupby("user_id").tail(1)
    val = ratings.drop(test.index).groupby("user_id").tail(1)
    train = ratings.drop(test.index).drop(val.index)
    return train, val, test

def binarize_ratings(ratings: pd.DataFrame, threshold: float = 3.5) -> pd.DataFrame:
    """
    Convert explicit ratings to implicit feedback.
    Ratings >= 3.5 → positive (1), else negative (0).
    Use this for NCF and Two-Tower training.
    """
    df = ratings.copy()
    df["label"] = (df["rating"] >= threshold).astype(float)
    return df
