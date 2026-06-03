from fastapi import APIRouter, HTTPException, Depends
import time
from backend.app.models.schemas import CompareRequest, CompareResponse, MovieRecommendation
from src.inference.recommender import UnifiedRecommender

router = APIRouter(tags=["compare"])

def get_recommender():
    from backend.app.main import recommender
    if recommender is None:
        raise HTTPException(status_code=503, detail="UnifiedRecommender models are still loading...")
    return recommender

def calculate_genre_diversity(recommendations, rec: UnifiedRecommender) -> float:
    """
    Computes pairwise content-based genre diversity (Jaccard distance)
    among recommended items. Range: [0.0, 1.0].
    """
    if len(recommendations) <= 1:
        return 0.0
        
    genres_lists = []
    for r in recommendations:
        meta = rec.movie_metadata.get(r[0], {})
        genres_lists.append(set(meta.get("genres", [])))
        
    distances = []
    for i in range(len(genres_lists)):
        for j in range(i + 1, len(genres_lists)):
            g1 = genres_lists[i]
            g2 = genres_lists[j]
            union = g1 | g2
            if not union:
                dist = 1.0
            else:
                intersection = g1 & g2
                dist = 1.0 - (len(intersection) / len(union))
            distances.append(dist)
            
    return round(float(sum(distances) / len(distances)), 4) if distances else 0.0

@router.post("/compare-models", response_model=CompareResponse)
def compare_models(req: CompareRequest, rec: UnifiedRecommender = Depends(get_recommender)):
    seen_movies = rec.user_history_set.get(req.user_id, set())
    all_movie_ids = list(rec.movie_metadata.keys())
    candidates = [mid for mid in all_movie_ids if mid not in seen_movies]
    
    if not candidates:
        candidates = all_movie_ids[:req.top_k]
        
    # Get predictions for all 3 models (without debiasing)
    try:
        svd_recs = rec.recommend(req.user_id, candidates, model_type="svd", debias=False)[:req.top_k]
        ncf_recs = rec.recommend(req.user_id, candidates, model_type="ncf", debias=False)[:req.top_k]
        tt_recs  = rec.recommend(req.user_id, candidates, model_type="two_tower", debias=False)[:req.top_k]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation comparison error: {str(e)}")
        
    # Helper to construct MovieRecommendation models
    def build_recommendation_list(recs):
        out = []
        for mid, score in recs:
            meta = rec.movie_metadata.get(mid, {"title": f"Movie {mid}", "genres": ["Unknown"], "year": None})
            out.append(MovieRecommendation(
                movie_id=mid,
                title=meta["title"],
                genres=meta["genres"],
                score=float(score),
                year=meta["year"],
                debiased_badge=False
            ))
        return out
        
    svd_list = build_recommendation_list(svd_recs)
    ncf_list = build_recommendation_list(ncf_recs)
    tt_list  = build_recommendation_list(tt_recs)
    
    # Overlap count (intersection of movie_ids across all three)
    svd_ids = set([r[0] for r in svd_recs])
    ncf_ids = set([r[0] for r in ncf_recs])
    tt_ids  = set([r[0] for r in tt_recs])
    
    overlap_ids = svd_ids & ncf_ids & tt_ids
    overlap_count = len(overlap_ids)
    
    # Calculate diversity
    diversity = {
        "svd": calculate_genre_diversity(svd_recs, rec),
        "ncf": calculate_genre_diversity(ncf_recs, rec),
        "two_tower": calculate_genre_diversity(tt_recs, rec)
    }
    
    return CompareResponse(
        svd=svd_list,
        ncf=ncf_list,
        two_tower=tt_list,
        overlap_count=overlap_count,
        diversity_scores=diversity
    )
