from pydantic import BaseModel
from typing import List, Optional

class RecommendRequest(BaseModel):
    user_id: int
    model: str = "svd"  # "svd" | "ncf" | "two_tower"
    top_k: int = 10
    debias: bool = False

class MovieRecommendation(BaseModel):
    movie_id: int
    title: str
    genres: List[str]
    score: float
    year: Optional[float] = None
    debiased_badge: bool = False

class RecommendResponse(BaseModel):
    recommendations: List[MovieRecommendation]
    model_used: str
    inference_time_ms: float

class CompareRequest(BaseModel):
    user_id: int
    top_k: int = 10

class CompareResponse(BaseModel):
    svd: List[MovieRecommendation]
    ncf: List[MovieRecommendation]
    two_tower: List[MovieRecommendation]
    overlap_count: int
    diversity_scores: dict
