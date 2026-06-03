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
import pickle

# Ensure src is in python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.data.loader import load_ratings, load_users, load_movies
from src.data.preprocessor import build_encoders, binarize_ratings, split_leave_one_out
from src.models.two_tower import TwoTowerModel
from src.training.train_ncf import NCFDataset, get_ncf_train_data

def prepare_user_features(users_df, user_enc):
    # gender to binary
    gender_map = {"M": 1.0, "F": 0.0}
    users_df = users_df.copy()
    users_df["gender_bin"] = users_df["gender"].map(gender_map)
    
    # age normalized (max age is 56 in MovieLens 1M)
    users_df["age_norm"] = users_df["age"] / 56.0
    
    # occupation one-hot (classes are 0-20)
    occupations_one_hot = pd.get_dummies(users_df["occupation"], prefix="occ")
    # Make sure all 21 columns exist (0 to 20)
    for i in range(21):
        col = f"occ_{i}"
        if col not in occupations_one_hot.columns:
            occupations_one_hot[col] = 0.0
            
    # sort columns to ensure deterministic ordering
    occ_cols = [f"occ_{i}" for i in range(21)]
    occupations_one_hot = occupations_one_hot[occ_cols].astype(float)
    
    user_features = pd.concat([users_df[["user_id", "gender_bin", "age_norm"]], occupations_one_hot], axis=1)
    
    # Filter only users present in encoder
    user_features = user_features[user_features["user_id"].isin(user_enc.classes_)].copy()
    user_features["user_idx"] = user_enc.transform(user_features["user_id"])
    
    user_features = user_features.sort_values("user_idx").drop(columns=["user_id", "user_idx"])
    return user_features

def prepare_movie_features(movies_df, movie_enc):
    genres_list = [
        "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
        "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
        "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]
    
    movies_df = movies_df.copy()
    movies_df["year"] = movies_df["year"].fillna(1990)
    # min-max scaling of year between 1900 and 2005
    movies_df["year_norm"] = (movies_df["year"] - 1900) / 105.0
    
    # Build multi-hot genres
    for genre in genres_list:
        movies_df[f"genre_{genre}"] = movies_df["genre_list"].apply(
            lambda x: 1.0 if isinstance(x, list) and genre in x else 0.0
        )
        
    genre_cols = [f"genre_{genre}" for genre in genres_list]
    
    # Filter only movies present in encoder
    movies_df = movies_df[movies_df["movie_id"].isin(movie_enc.classes_)].copy()
    movies_df["movie_idx"] = movie_enc.transform(movies_df["movie_id"])
    
    movie_features = movies_df[["movie_idx", "year_norm"] + genre_cols]
    movie_features = movie_features.sort_values("movie_idx").drop(columns=["movie_idx"])
    return movie_features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--embed_dim", type=int, default=32, help="Embedding dimension")
    args = parser.parse_args()

    print("Loading data...")
    ratings = load_ratings().sample(frac=0.1, random_state=42)
    users = load_users()
    movies = load_movies()
    
    # Load or rebuild encoders
    processed_dir = Path("data/processed")
    user_encoder_path = processed_dir / "user_encoder.pkl"
    movie_encoder_path = processed_dir / "movie_encoder.pkl"
    
    if user_encoder_path.exists() and movie_encoder_path.exists():
        print("Loading pre-built encoders...")
        with open(user_encoder_path, "rb") as f:
            user_enc = pickle.load(f)
        with open(movie_encoder_path, "rb") as f:
            movie_enc = pickle.load(f)
        ratings = ratings.copy()
        ratings = ratings[ratings["user_id"].isin(user_enc.classes_) & ratings["movie_id"].isin(movie_enc.classes_)].copy()
        ratings["user_idx"] = user_enc.transform(ratings["user_id"])
        ratings["movie_idx"] = movie_enc.transform(ratings["movie_id"])
    else:
        print("Building encoders...")
        ratings, user_enc, movie_enc = build_encoders(ratings)
        
    n_users = len(user_enc.classes_)
    n_movies = len(movie_enc.classes_)
    
    ratings = binarize_ratings(ratings, threshold=3.5)
    train_ratings, val_ratings, test_ratings = split_leave_one_out(ratings)
    
    print("Preparing user and movie features...")
    user_features = prepare_user_features(users, user_enc)
    movie_features = prepare_movie_features(movies, movie_enc)
    
    # Convert to PyTorch Tensors
    user_feats_np = user_features.values.astype(np.float32)
    movie_feats_np = movie_features.values.astype(np.float32)
    
    # Save the prepared feature arrays for quick loading in backend/inference
    np.save(processed_dir / "user_features.npy", user_feats_np)
    np.save(processed_dir / "movie_features.npy", movie_feats_np)
    
    user_feat_dim = user_feats_np.shape[1]
    movie_feat_dim = movie_feats_np.shape[1]
    print(f"User features dim: {user_feat_dim}, Movie features dim: {movie_feat_dim}")
    
    print("Generating negative samples for training Two-Tower...")
    train_users, train_movies, train_labels = get_ncf_train_data(train_ratings, n_movies, n_neg=1)
    train_dataset = NCFDataset(train_users, train_movies, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    val_users = val_ratings["user_idx"].values
    val_movies = val_ratings["movie_idx"].values
    val_labels = val_ratings["label"].values
    val_dataset = NCFDataset(val_users, val_movies, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    model = TwoTowerModel(user_feat_dim, movie_feat_dim, embed_dim=args.embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()
    
    user_feats_tensor = torch.tensor(user_feats_np, dtype=torch.float32).to(device)
    movie_feats_tensor = torch.tensor(movie_feats_np, dtype=torch.float32).to(device)
    
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
            
            # Fetch features using indexing
            u_feats = user_feats_tensor[batch_users]
            m_feats = movie_feats_tensor[batch_movies]
            
            sim = model(u_feats, m_feats)
            # Map cosine similarity [-1, 1] to positive probability using sigmoid with temperature
            preds = torch.sigmoid(sim * 5.0)
            
            loss = criterion(preds, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        epoch_time = time.time() - start_time
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_users, batch_movies, batch_labels in val_loader:
                batch_users = batch_users.to(device)
                batch_movies = batch_movies.to(device)
                batch_labels = batch_labels.to(device)
                
                u_feats = user_feats_tensor[batch_users]
                m_feats = movie_feats_tensor[batch_movies]
                
                sim = model(u_feats, m_feats)
                preds = torch.sigmoid(sim * 5.0)
                loss = criterion(preds, batch_labels)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {epoch_time:.1f}s")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "saved_models/two_tower.pt")
            print(" -> Saved best Two-Tower model weights to saved_models/two_tower.pt")
            
    print("Training finished.")

if __name__ == "__main__":
    main()
