import os
import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time

# Ensure src is in python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.data.loader import load_ratings
from src.data.preprocessor import build_encoders, binarize_ratings, split_leave_one_out
from src.models.ncf_model import NCF
from src.models.debiaser import IPSDebiaser

class NCFDataset(Dataset):
    def __init__(self, users, items, labels):
        self.users = torch.tensor(users, dtype=torch.long)
        self.items = torch.tensor(items, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

def get_ncf_train_data(train_df, n_movies, n_neg=4):
    """
    Optimized negative sampling. For each positive rating, samples n_neg negative items
    that the user has not rated.
    """
    user_items = train_df.groupby("user_idx")["movie_idx"].apply(set).to_dict()
    users = train_df["user_idx"].values
    movies = train_df["movie_idx"].values
    
    user_input = []
    item_input = []
    labels = []
    
    # Positive samples
    user_input.extend(users)
    item_input.extend(movies)
    labels.extend([1.0] * len(users))
    
    # Negative samples
    for u, seen in user_items.items():
        n_user_neg = min(len(seen) * n_neg, n_movies - len(seen))
        negatives = []
        # Sample in batches to reduce loops
        while len(negatives) < n_user_neg:
            samples = np.random.randint(0, n_movies, size=n_user_neg * 2)
            for s in samples:
                if s not in seen and s not in negatives:
                    negatives.append(s)
                if len(negatives) == n_user_neg:
                    break
        user_input.extend([u] * n_user_neg)
        item_input.extend(negatives)
        labels.extend([0.0] * n_user_neg)
        
    return np.array(user_input), np.array(item_input), np.array(labels)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--debias", action="store_true", help="Train with IPS causal debiasing")
    parser.add_argument("--embed_dim", type=int, default=32, help="Embedding dimension")
    args = parser.parse_args()

    print("Loading data...")
    ratings = load_ratings().sample(frac=0.1, random_state=42)
    
    print("Preprocessing & encoding user/movie IDs...")
    ratings, user_enc, movie_enc = build_encoders(ratings)
    
    n_users = len(user_enc.classes_)
    n_movies = len(movie_enc.classes_)
    print(f"Dataset stats: {n_users} users, {n_movies} movies, {len(ratings)} ratings.")
    
    # Apply threshold for implicit label: ratings >= 3.5 -> positive, else negative
    ratings = binarize_ratings(ratings, threshold=3.5)
    
    # Split using leave-one-out
    train_ratings, val_ratings, test_ratings = split_leave_one_out(ratings)
    print(f"Split shapes - Train: {len(train_ratings)}, Val: {len(val_ratings)}, Test: {len(test_ratings)}")
    
    # Save the processed parquets
    os.makedirs("data/processed", exist_ok=True)
    train_ratings.to_parquet("data/processed/train_ratings.parquet")
    val_ratings.to_parquet("data/processed/val_ratings.parquet")
    test_ratings.to_parquet("data/processed/test_ratings.parquet")
    
    # Fit IPS debiaser on train data
    debiaser = IPSDebiaser()
    debiaser.fit(train_ratings)
    debiaser.save("saved_models/debiaser.pkl")
    
    print("Generating negative samples for training...")
    train_users, train_movies, train_labels = get_ncf_train_data(train_ratings, n_movies, n_neg=1)
    train_dataset = NCFDataset(train_users, train_movies, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Handle Validation (positives + 100 negatives per validation item for HR/NDCG speed)
    val_users = val_ratings["user_idx"].values
    val_movies = val_ratings["movie_idx"].values
    val_labels = val_ratings["label"].values
    val_dataset = NCFDataset(val_users, val_movies, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    model = NCF(n_users, n_movies, embed_dim=args.embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    if args.debias:
        print("Training with Inverse Propensity Scoring (IPS) Debiasing enabled.")
        criterion = None  # Handled inside the loop using debiaser weighted_bce_loss
    else:
        print("Training standard NCF model.")
        criterion = nn.BCELoss()
        
    os.makedirs("saved_models", exist_ok=True)
    
    best_val_loss = float("inf")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for batch_users, batch_movies, batch_labels in train_loader:
            batch_users = batch_users.to(device)
            batch_movies = batch_movies.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            preds = model(batch_users, batch_movies)
            
            if args.debias:
                loss = debiaser.weighted_bce_loss(preds, batch_labels, batch_movies, is_index=True)
            else:
                loss = criterion(preds, batch_labels)
                
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        epoch_time = time.time() - start_time
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation loss (Standard BCE)
        model.eval()
        val_loss = 0
        val_criterion = nn.BCELoss()
        with torch.no_grad():
            for batch_users, batch_movies, batch_labels in val_loader:
                batch_users = batch_users.to(device)
                batch_movies = batch_movies.to(device)
                batch_labels = batch_labels.to(device)
                preds = model(batch_users, batch_movies)
                loss = val_criterion(preds, batch_labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {epoch_time:.1f}s")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_filename = "saved_models/ncf_debiased_model.pt" if args.debias else "saved_models/ncf_model.pt"
            torch.save(model.state_dict(), save_filename)
            print(f" -> Saved best model weights to {save_filename}")

    print("Training finished.")

if __name__ == "__main__":
    main()
