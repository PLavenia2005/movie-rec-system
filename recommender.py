import os
import ast
import requests
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from difflib import get_close_matches


load_dotenv()
TMDB_API_KEY = os.getenv('TMDB_API_KEY')
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

DATA_CSV = os.path.join(os.path.dirname(__file__), "Data", "tmdb_5000_movies.csv")


def _parse_items(text):
    try:
        items = ast.literal_eval(text)
        if isinstance(items, list):
            names = [i.get("name", "") for i in items if isinstance(i, dict)]
            return " ".join(names)
    except Exception:
        pass
    return ""


def _load_data():
    df = pd.read_csv(DATA_CSV)
    df = df[['id', 'title', 'overview', 'genres', 'keywords']].copy()
    df['genres'] = df['genres'].fillna('[]').apply(_parse_items)
    df['keywords'] = df['keywords'].fillna('[]').apply(_parse_items)
    df['overview'] = df['overview'].fillna('')
    df['tags'] = (df['title'].fillna('') + ' ' + df['overview'] + ' ' + df['genres'] + ' ' + df['keywords']).str.lower()
    df = df.reset_index(drop=True)
    return df


# build data and similarity on import so the API is responsive
_df = _load_data()
_indices = pd.Series(_df.index, index=_df['title'].str.lower()).to_dict()

_vectorizer = CountVectorizer(max_features=5000, stop_words='english')
_vectors = _vectorizer.fit_transform(_df['tags'])
_similarity = cosine_similarity(_vectors)


def _fetch_poster_for_title(title: str):
    """Use TMDB search API to find poster path for a movie title. Returns full image URL or None."""
    if not TMDB_API_KEY:
        return None
    try:
        resp = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={"api_key": TMDB_API_KEY, "query": title, "page": 1},
            timeout=5,
        )
        data = resp.json()
        results = data.get('results') or []
        if results:
            poster = results[0].get('poster_path')
            if poster:
                return TMDB_IMAGE_BASE + poster
    except Exception:
        return None
    return None


def recommend(title: str, top_n: int = 10):
    """Return a list of recommended movies with optional posters.

    Returns a list of dicts: {"title": str, "poster": Optional[str]}
    """
    if not title:
        return []
    key = title.lower().strip()
    idx = _indices.get(key)

    # try substring match (partial title)
    if idx is None:
        for t, i in _indices.items():
            if key in t:
                idx = i
                break

    # try fuzzy close match
    if idx is None:
        candidates = get_close_matches(key, list(_indices.keys()), n=1, cutoff=0.6)
        if candidates:
            idx = _indices.get(candidates[0])

    if idx is None:
        return []
    distances = list(enumerate(_similarity[idx]))
    distances = sorted(distances, key=lambda x: x[1], reverse=True)
    results = []
    for i, score in distances[1: top_n + 1]:
        t = _df.loc[i, 'title']
        poster = _fetch_poster_for_title(t) if TMDB_API_KEY else None
        results.append({"title": t, "poster": poster})
    return results


def random_movies(top_n: int = 12):
    """Return a random sample of movies from the dataset with optional posters."""
    sample = _df.sample(n=min(top_n, len(_df))).reset_index(drop=True)
    results = []
    for _, row in sample.iterrows():
        t = row['title']
        poster = _fetch_poster_for_title(t) if TMDB_API_KEY else None
        results.append({"title": t, "poster": poster})
    return results


if __name__ == '__main__':
    print(recommend('Avatar'))
