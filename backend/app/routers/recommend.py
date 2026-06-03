from fastapi import APIRouter, HTTPException, Depends
import time
from backend.app.models.schemas import RecommendRequest, RecommendResponse, MovieRecommendation
from src.inference.recommender import UnifiedRecommender

router = APIRouter(tags=["recommend"])

def get_recommender():
    from backend.app.main import recommender
    if recommender is None:
        raise HTTPException(status_code=503, detail="UnifiedRecommender models are still loading...")
    return recommender

@router.post("/recommend", response_model=RecommendResponse)
def get_recommendations(req: RecommendRequest, rec: UnifiedRecommender = Depends(get_recommender)):
    # 1. Get user's seen history
    seen_movies = rec.user_history_set.get(req.user_id, set())
    
    # 2. Candidate movies: all catalog movies minus seen movies
    all_movie_ids = list(rec.movie_metadata.keys())
    candidates = [mid for mid in all_movie_ids if mid not in seen_movies]
    
    if not candidates:
        # Fallback if user saw everything (highly unlikely)
        candidates = all_movie_ids[:req.top_k]
        
    start_time = time.time()
    
    # 3. Predict & rank candidates
    try:
        ranked = rec.recommend(req.user_id, candidates, model_type=req.model, debias=req.debias)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation engine error: {str(e)}")
        
    inference_time = (time.time() - start_time) * 1000.0  # in ms
    
    # 4. Formulate response recommendations
    top_recommendations = ranked[:req.top_k]
    movie_recs = []
    
    for mid, score in top_recommendations:
        meta = rec.movie_metadata.get(mid, {"title": f"Movie {mid}", "genres": ["Unknown"], "year": None})
        
        # Causal debiasing badge logic:
        # If debias is true and the item is in the long-tail (propensity < 0.1), mark it!
        debiased_badge = False
        if req.debias and rec.debiaser:
            prop = rec.debiaser.propensities_id.get(mid, 0.0)
            if prop < 0.1:  # Long-tail item
                debiased_badge = True
                
        movie_recs.append(MovieRecommendation(
            movie_id=mid,
            title=meta["title"],
            genres=meta["genres"],
            score=float(score),
            year=meta["year"],
            debiased_badge=debiased_badge
        ))
        
    return RecommendResponse(
        recommendations=movie_recs,
        model_used=req.model,
        inference_time_ms=round(inference_time, 2)
    )
