import os
import sys
from pathlib import Path
import pandas as pd

# Ensure src is in python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.data.loader import load_ratings
from src.models.svd_model import SVDRecommender

def main():
    print("Loading data...")
    ratings = load_ratings()
    
    # Check if pre-split train ratings exists
    train_parquet = Path("data/processed/train_ratings.parquet")
    if train_parquet.exists():
        print("Using pre-split train ratings...")
        train_df = pd.read_parquet(train_parquet)
    else:
        print("Training on all ratings (fallback)...")
        train_df = ratings
        
    print("Fitting SVD Baseline...")
    svd = SVDRecommender(n_factors=100, n_epochs=20)
    svd.fit(train_df)
    
    os.makedirs("saved_models", exist_ok=True)
    svd.save("saved_models/svd_model.pkl")
    print("SVD Model saved successfully!")

if __name__ == "__main__":
    main()
