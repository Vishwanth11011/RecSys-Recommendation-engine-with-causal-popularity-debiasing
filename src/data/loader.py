import pandas as pd
from pathlib import Path

# Base data directory
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "ml-1m"

def load_ratings() -> pd.DataFrame:
    """Load ratings.dat — UserID::MovieID::Rating::Timestamp"""
    return pd.read_csv(
        DATA_DIR / "ratings.dat",
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1"
    )

def load_movies() -> pd.DataFrame:
    """Load movies.dat — MovieID::Title::Genres"""
    df = pd.read_csv(
        DATA_DIR / "movies.dat",
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1"
    )
    df["year"] = df["title"].str.extract(r"\((\d{4})\)").astype(float)
    df["genre_list"] = df["genres"].str.split("|")
    return df

def load_users() -> pd.DataFrame:
    """Load users.dat — UserID::Gender::Age::Occupation::Zip"""
    return pd.read_csv(
        DATA_DIR / "users.dat",
        sep="::",
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zip"],
        encoding="latin-1"
    )
