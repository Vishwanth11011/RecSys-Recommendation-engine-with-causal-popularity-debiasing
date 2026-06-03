import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure src is in python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.inference.recommender import UnifiedRecommender
from src.evaluation.metrics import ndcg_at_k, hit_rate_at_k, precision_at_k, coverage

def get_eval_candidates(test_df, train_df, all_movie_ids, n_neg=99):
    """
    For each test user, returns their 1 positive movie and n_neg negative movies
    that they have not rated.
    """
    print("Preparing 100-candidate evaluation sets (1 positive + 99 negatives)...")
    user_pos = test_df.groupby("user_id")["movie_id"].first().to_dict()
    
    # Get all movies rated by each user in train + val + test
    user_seen = pd.concat([train_df, test_df]).groupby("user_id")["movie_id"].apply(set).to_dict()
    
    eval_sets = {}
    movie_array = np.array(all_movie_ids)
    
    for uid, pos_mid in user_pos.items():
        seen = user_seen.get(uid, set())
        negatives = []
        while len(negatives) < n_neg:
            samples = np.random.choice(movie_array, size=n_neg * 2, replace=False)
            for s in samples:
                if s not in seen and s != pos_mid and s not in negatives:
                    negatives.append(s)
                if len(negatives) == n_neg:
                    break
        eval_sets[uid] = {
            "pos": pos_mid,
            "candidates": [pos_mid] + negatives
        }
    return eval_sets

def main():
    print("Starting evaluation pipeline...")
    
    # Load splits
    processed_dir = Path("data/processed")
    train_ratings = pd.read_parquet(processed_dir / "train_ratings.parquet")
    test_ratings = pd.read_parquet(processed_dir / "test_ratings.parquet")
    
    recommender = UnifiedRecommender()
    recommender.load_all_models()
    
    all_movie_ids = list(recommender.movie_enc.classes_)
    n_movies = len(all_movie_ids)
    
    # Generate 100-candidate test sets for faster, standard NCF evaluation
    eval_sets = get_eval_candidates(test_ratings, train_ratings, all_movie_ids, n_neg=99)
    
    # Let's evaluate first 2000 users for quick & stable results
    test_users = list(eval_sets.keys())[:2000]
    print(f"Evaluating models on {len(test_users)} users using 100-candidate protocol...")
    
    results = {}
    
    models_to_evaluate = [
        {"name": "SVD Baseline", "key": "svd", "debias": False},
        {"name": "SVD Debiased", "key": "svd", "debias": True},
        {"name": "Neural CF (NCF)", "key": "ncf", "debias": False},
        {"name": "Neural CF + IPS", "key": "ncf", "debias": True},
        {"name": "Two-Tower", "key": "two_tower", "debias": False},
        {"name": "Two-Tower + IPS", "key": "two_tower", "debias": True}
    ]
    
    for m in models_to_evaluate:
        name = m["name"]
        key = m["key"]
        debias = m["debias"]
        
        print(f"Evaluating {name}...")
        ndcg_scores = []
        hr_scores = []
        prec_scores = []
        all_recs = []
        
        start_time = time.time()
        for uid in test_users:
            pos_mid = eval_sets[uid]["pos"]
            candidates = eval_sets[uid]["candidates"]
            
            # Predict and rank candidates
            ranked = recommender.recommend(uid, candidates, model_type=key, debias=debias)
            ranked_ids = [r[0] for r in ranked[:10]]
            
            ndcg_scores.append(ndcg_at_k(ranked_ids, {pos_mid}, k=10))
            hr_scores.append(hit_rate_at_k(ranked_ids, {pos_mid}, k=10))
            prec_scores.append(precision_at_k(ranked_ids, {pos_mid}, k=10))
            all_recs.append(ranked_ids)
            
        eval_time = time.time() - start_time
        
        # Coverage on test set recommendations
        cov = coverage(all_recs, n_movies)
        
        avg_ndcg = np.mean(ndcg_scores)
        avg_hr = np.mean(hr_scores)
        avg_prec = np.mean(prec_scores)
        
        print(f"  NDCG@10: {avg_ndcg:.4f} | HitRate@10: {avg_hr:.4f} | Coverage: {cov:.4f} | Time: {eval_time:.1f}s")
        
        results[name] = {
            "NDCG@10": round(float(avg_ndcg), 4),
            "HitRate@10": round(float(avg_hr), 4),
            "Precision@10": round(float(avg_prec), 4),
            "Coverage": round(float(cov), 4),
            "InferenceTimeMs": round(float((eval_time / len(test_users)) * 1000.0), 3)
        }
        
    # Write to saved_models/metrics.json for FastAPI backend
    os.makedirs("saved_models", exist_ok=True)
    with open("saved_models/metrics.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n================ EVALUATION SUMMARY ================")
    df = pd.DataFrame(results).T
    print(df.to_markdown())
    print("====================================================")
    
if __name__ == "__main__":
    main()
