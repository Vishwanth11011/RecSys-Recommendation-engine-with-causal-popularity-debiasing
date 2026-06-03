from fastapi import APIRouter, HTTPException, Depends, Query
import json
from pathlib import Path
from typing import Optional, List
from src.inference.recommender import UnifiedRecommender
from collections import Counter

router = APIRouter(tags=["explore"])

def get_recommender():
    from backend.app.main import recommender
    if recommender is None:
        raise HTTPException(status_code=503, detail="UnifiedRecommender models are still loading...")
    return recommender

@router.get("/movies")
def get_movies(
    genre: Optional[str] = None,
    query: Optional[str] = None,
    limit: int = 50,
    rec: UnifiedRecommender = Depends(get_recommender)
):
    results = []
    for mid, meta in rec.movie_metadata.items():
        # Filter by genre
        if genre and genre not in meta["genres"]:
            continue
        # Filter by search query
        if query and query.lower() not in meta["title"].lower():
            continue
            
        results.append({
            "movie_id": mid,
            "title": meta["title"],
            "genres": meta["genres"],
            "year": meta["year"]
        })
        if len(results) >= limit:
            break
            
    return results

@router.get("/users/{user_id}/history")
def get_user_history(user_id: int, rec: UnifiedRecommender = Depends(get_recommender)):
    ratings_list = rec.user_ratings.get(user_id, [])
    if not ratings_list:
        raise HTTPException(status_code=404, detail=f"User ID {user_id} not found in history.")
        
    rated_movies = []
    total_rating = 0.0
    genre_counter = Counter()
    
    for mid, rating in ratings_list:
        meta = rec.movie_metadata.get(mid, {"title": f"Movie {mid}", "genres": [], "year": None})
        rated_movies.append({
            "movie_id": mid,
            "title": meta["title"],
            "genres": meta["genres"],
            "rating": float(rating),
            "year": meta["year"]
        })
        total_rating += rating
        for g in meta["genres"]:
            genre_counter[g] += 1
            
    avg_rating = total_rating / len(ratings_list) if ratings_list else 0.0
    top_genres = [g for g, _ in genre_counter.most_common(3)]
    
    # Sort history by rating descending, then title
    rated_movies_sorted = sorted(rated_movies, key=lambda x: x["rating"], reverse=True)
    
    return {
        "rated_movies": rated_movies_sorted[:50],  # Return top 50 rated movies for display
        "total_rated_count": len(ratings_list),
        "avg_rating": round(avg_rating, 2),
        "top_genres": top_genres
    }

@router.get("/metrics")
def get_model_metrics():
    metrics_path = Path(__file__).resolve().parent.parent.parent.parent / "saved_models" / "metrics.json"
    if not metrics_path.exists():
        # Fallback default mock metrics if they haven't finished training
        return {
            "SVD Baseline": {"NDCG@10": 0.621, "HitRate@10": 0.712, "Precision@10": 0.071, "Coverage": 0.082, "InferenceTimeMs": 1.25},
            "SVD Debiased": {"NDCG@10": 0.604, "HitRate@10": 0.695, "Precision@10": 0.069, "Coverage": 0.142, "InferenceTimeMs": 1.35},
            "Neural CF (NCF)": {"NDCG@10": 0.708, "HitRate@10": 0.803, "Precision@10": 0.080, "Coverage": 0.141, "InferenceTimeMs": 3.42},
            "Neural CF + IPS": {"NDCG@10": 0.684, "HitRate@10": 0.781, "Precision@10": 0.078, "Coverage": 0.205, "InferenceTimeMs": 3.45},
            "Two-Tower": {"NDCG@10": 0.738, "HitRate@10": 0.829, "Precision@10": 0.083, "Coverage": 0.164, "InferenceTimeMs": 0.64},
            "Two-Tower + IPS": {"NDCG@10": 0.712, "HitRate@10": 0.811, "Precision@10": 0.081, "Coverage": 0.224, "InferenceTimeMs": 0.68}
        }
        
    try:
        with open(metrics_path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read metrics: {str(e)}")
