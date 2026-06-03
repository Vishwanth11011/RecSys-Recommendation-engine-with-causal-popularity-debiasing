import pandas as pd
from pathlib import Path
import urllib.request
import zipfile
import io

# Base data directory
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "ml-1m"

def download_and_extract_metadata():
    """Download MovieLens 1M zip and extract raw files if missing."""
    if not (DATA_DIR / "movies.dat").exists() or not (DATA_DIR / "users.dat").exists() or not (DATA_DIR / "ratings.dat").exists():
        print("loader.py: Dataset files not found. Downloading from GroupLens...")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
        
        try:
            # Download zip in memory
            with urllib.request.urlopen(url) as response:
                zip_data = response.read()
            
            # Extract movies.dat, users.dat, and ratings.dat
            with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
                for member in z.namelist():
                    if member.endswith("movies.dat") or member.endswith("users.dat") or member.endswith("ratings.dat"):
                        filename = Path(member).name
                        with open(DATA_DIR / filename, "wb") as f:
                            f.write(z.read(member))
            print("loader.py: Dataset files downloaded and extracted successfully.")
        except Exception as e:
            print(f"loader.py: Failed to download MovieLens dataset: {str(e)}")

def load_ratings() -> pd.DataFrame:
    """Load ratings.dat — UserID::MovieID::Rating::Timestamp"""
    download_and_extract_metadata()
    return pd.read_csv(
        DATA_DIR / "ratings.dat",
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1"
    )

def load_movies() -> pd.DataFrame:
    """Load movies.dat — MovieID::Title::Genres"""
    download_and_extract_metadata()
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
    download_and_extract_metadata()
    return pd.read_csv(
        DATA_DIR / "users.dat",
        sep="::",
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zip"],
        encoding="latin-1"
    )
