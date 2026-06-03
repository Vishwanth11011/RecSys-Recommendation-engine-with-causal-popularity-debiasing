import numpy as np

def ndcg_at_k(recommended: list, relevant: set, k: int = 10) -> float:
    """
    Normalized Discounted Cumulative Gain @ K.
    Measures ranking quality — relevant items ranked higher contribute more.
    """
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, item in enumerate(recommended[:k])
        if item in relevant
    )
    ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

def hit_rate_at_k(recommended: list, relevant: set, k: int = 10) -> float:
    """HR@K: fraction of users where at least one recommended item is relevant."""
    return float(len(set(recommended[:k]) & relevant) > 0)

def precision_at_k(recommended: list, relevant: set, k: int = 10) -> float:
    return len(set(recommended[:k]) & relevant) / k

def coverage(all_recommendations: list, n_movies: int) -> float:
    """Catalog coverage: fraction of all movies ever recommended."""
    unique_recommended = set(item for recs in all_recommendations for item in recs)
    return len(unique_recommended) / n_movies

def evaluate_model(model_predict_fn, test_set, all_movie_ids, k=10) -> dict:
    """
    Run evaluation across all test users.
    model_predict_fn: callable(user_id, candidate_movies) → ranked list of (movie_id, score) or movie_id
    test_set: dict {user_id: set of relevant movie_ids}
    """
    ndcg_scores, hr_scores, prec_scores = [], [], []
    all_recs = []
    
    # Slice first 1000 users for faster evaluation if needed, but standard ml-1m is fast enough for SVD.
    # Let's write the code to evaluate all users.
    for user_id, relevant in test_set.items():
        # Get ranked movies for this user
        ranked = model_predict_fn(user_id, all_movie_ids)
        if not ranked:
            continue
        ranked = ranked[:k]
        # Handle list of tuples (movie_id, score) vs list of movie_ids
        ranked_ids = [r[0] for r in ranked] if isinstance(ranked[0], tuple) else list(ranked)
        
        ndcg_scores.append(ndcg_at_k(ranked_ids, relevant, k))
        hr_scores.append(hit_rate_at_k(ranked_ids, relevant, k))
        prec_scores.append(precision_at_k(ranked_ids, relevant, k))
        all_recs.append(ranked_ids)
        
    if not ndcg_scores:
        return {
            f"NDCG@{k}": 0.0,
            f"HitRate@{k}": 0.0,
            f"Precision@{k}": 0.0,
            "Coverage": 0.0
        }
        
    return {
        f"NDCG@{k}":      round(np.mean(ndcg_scores), 4),
        f"HitRate@{k}":   round(np.mean(hr_scores), 4),
        f"Precision@{k}": round(np.mean(prec_scores), 4),
        "Coverage":       round(coverage(all_recs, len(all_movie_ids)), 4),
    }
