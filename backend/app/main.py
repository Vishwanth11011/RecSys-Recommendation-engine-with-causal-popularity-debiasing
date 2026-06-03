import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Ensure project root is in python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.inference.recommender import UnifiedRecommender

recommender = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global recommender
    print("FastAPI Lifespan: Loading all recommendation models...")
    recommender = UnifiedRecommender()
    recommender.load_all_models()
    yield
    print("FastAPI Lifespan: Shuting down...")

app = FastAPI(
    title="RecSys Engine API",
    description="FastAPI Backend for Hybrid Movie Recommendation Engine (SVD, NCF, Two-Tower)",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration for local development and production frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for dev/testing, customize in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include routers
from backend.app.routers import recommend, compare, explore

app.include_router(recommend.router)
app.include_router(compare.router)
app.include_router(explore.router)

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Movie Recommendation Engine Backend API is active.",
        "endpoints": [
            "POST /recommend",
            "POST /compare-models",
            "GET /movies",
            "GET /users/{user_id}/history",
            "GET /metrics"
        ]
    }
