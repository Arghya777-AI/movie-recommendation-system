"""
CineAI — Core Recommendation Engine
Hybrid content-based filtering using TF-IDF + Count Vectorizer + Weighted Ratings
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import warnings
warnings.filterwarnings('ignore')


class MovieRecommender:
    """
    Ultra-premium hybrid movie recommendation engine.
    Combines semantic (TF-IDF overview) + structural (genre/keyword soup) filtering
    with IMDB-style weighted ratings.
    """

    def __init__(self, data_path: str):
        self.df = None
        self.cosine_sim_tfidf = None
        self.cosine_sim_count = None
        self.cosine_sim_hybrid = None
        self.indices = None
        self._load_and_prepare(data_path)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _parse(val, key='name', limit=None):
        """Safely parse JSON-like string columns and extract a list of names."""
        try:
            items = ast.literal_eval(val) if isinstance(val, str) else val
            names = [item[key] for item in items if isinstance(item, dict) and key in item]
            return names[:limit] if limit else names
        except Exception:
            return []

    @staticmethod
    def _weighted_rating(row, C, m):
        v, R = row['vote_count'], row['vote_average']
        return (v / (v + m)) * R + (m / (v + m)) * C

    # ----------------------------------------------------------------- loading
    def _load_and_prepare(self, data_path: str):
        df = pd.read_csv(data_path)

        # Parse list-valued columns
        df['genres_list']    = df['genres'].apply(lambda x: self._parse(x))
        df['keywords_list']  = df['keywords'].apply(lambda x: self._parse(x, limit=12))
        df['companies_list'] = df['production_companies'].apply(lambda x: self._parse(x, limit=3))
        df['countries_list'] = df['production_countries'].apply(lambda x: self._parse(x))
        df['languages_list'] = df['spoken_languages'].apply(lambda x: self._parse(x))

        # Numeric cleanup
        for col in ['budget', 'revenue']:
            df[col] = pd.to_numeric(df[col], errors='coerce').replace(0, np.nan)
        for col in ['runtime', 'vote_average', 'vote_count', 'popularity']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
        df['overview']     = df['overview'].fillna('')
        df['tagline']      = df['tagline'].fillna('')
        df['profit']       = df['revenue'] - df['budget']
        df['roi']          = (df['profit'] / df['budget']) * 100

        # IMDB weighted rating (70th-percentile vote count threshold)
        C = df['vote_average'].mean()
        m = df['vote_count'].quantile(0.70)
        mask = df['vote_count'] >= m
        df.loc[mask, 'weighted_rating'] = df[mask].apply(
            lambda r: self._weighted_rating(r, C, m), axis=1
        )
        df['weighted_rating'] = df['weighted_rating'].fillna(0)

        # Feature strings
        def clean(lst): return ' '.join(s.lower().replace(' ', '') for s in lst)
        df['genres_str']   = df['genres_list'].apply(clean)
        df['keywords_str'] = df['keywords_list'].apply(clean)
        df['companies_str']= df['companies_list'].apply(clean)

        # Soup: genres weighted 3×, keywords 2×
        df['soup'] = (
            df['overview'] + ' '
            + (df['genres_str'] + ' ') * 3
            + (df['keywords_str'] + ' ') * 2
            + df['companies_str'] + ' '
            + df['tagline']
        )

        self.df = df.reset_index(drop=True)
        self.indices = pd.Series(self.df.index, index=self.df['title'].str.lower())

        # TF-IDF on overview (semantic similarity)
        tfidf = TfidfVectorizer(stop_words='english', max_features=12000, ngram_range=(1, 2))
        tfidf_mat = tfidf.fit_transform(df['overview'])
        self.cosine_sim_tfidf = cosine_similarity(tfidf_mat, tfidf_mat)

        # Count vectoriser on soup (structural / genre-keyword similarity)
        count = CountVectorizer(stop_words='english', max_features=6000)
        count_mat = count.fit_transform(df['soup'])
        self.cosine_sim_count = cosine_similarity(count_mat, count_mat)

        # Hybrid: 35% semantic + 65% structural
        self.cosine_sim_hybrid = (
            0.35 * self.cosine_sim_tfidf + 0.65 * self.cosine_sim_count
        )

    # --------------------------------------------------------- recommendation API
    def _resolve_index(self, title: str):
        t = title.strip().lower()
        if t in self.indices:
            idx = self.indices[t]
            return int(idx.iloc[0]) if isinstance(idx, pd.Series) else int(idx)
        # Partial-match fallback
        candidates = [k for k in self.indices.index if t in k]
        if candidates:
            idx = self.indices[candidates[0]]
            return int(idx.iloc[0]) if isinstance(idx, pd.Series) else int(idx)
        return None

    def recommend(self, title: str, n: int = 12, method: str = 'hybrid'):
        """
        Return (DataFrame of recommendations, error_string | None).
        method: 'hybrid' | 'semantic' | 'structural'
        """
        idx = self._resolve_index(title)
        if idx is None:
            return None, f"'{title}' not found in the database."

        matrix = {
            'hybrid':     self.cosine_sim_hybrid,
            'semantic':   self.cosine_sim_tfidf,
            'structural': self.cosine_sim_count,
        }.get(method, self.cosine_sim_hybrid)

        scores = list(enumerate(matrix[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n + 30]

        movie_indices = [s[0] for s in scores]
        similarity    = [s[1] for s in scores]

        result = self.df.iloc[movie_indices].copy()
        result['similarity_score'] = similarity
        return result.head(n), None

    def top_movies(self, n: int = 20, genre: str = None,
                   min_year: int = None, max_year: int = None,
                   min_votes: int = 200):
        """Top movies by IMDB-style weighted rating."""
        df = self.df[self.df['weighted_rating'] > 0].copy()
        if genre:
            df = df[df['genres_list'].apply(lambda g: genre in g)]
        if min_year:
            df = df[df['release_year'] >= min_year]
        if max_year:
            df = df[df['release_year'] <= max_year]
        df = df[df['vote_count'] >= min_votes]
        return df.nlargest(n, 'weighted_rating')

    def search(self, query: str, n: int = 12):
        """Fuzzy title + overview search."""
        q = query.lower()
        mask = (
            self.df['title'].str.lower().str.contains(q, na=False) |
            self.df['overview'].str.lower().str.contains(q, na=False)
        )
        return self.df[mask].sort_values('weighted_rating', ascending=False).head(n)

    def genre_recs(self, genres: list, n: int = 20, sort_by: str = 'weighted_rating'):
        """Recommendations filtered/ranked by genre overlap."""
        df = self.df.copy()
        df['genre_score'] = df['genres_list'].apply(
            lambda g: sum(1 for x in genres if x in g)
        )
        df = df[df['genre_score'] > 0]
        return df.sort_values(['genre_score', sort_by], ascending=[False, False]).head(n)

    def decade_stats(self):
        df = self.df.dropna(subset=['release_year']).copy()
        df['decade'] = (df['release_year'] // 10) * 10
        return df.groupby('decade').agg(
            count         = ('title', 'count'),
            avg_rating    = ('vote_average', 'mean'),
            avg_budget    = ('budget', 'mean'),
            avg_revenue   = ('revenue', 'mean'),
            avg_popularity= ('popularity', 'mean'),
        ).reset_index()

    def genre_stats(self):
        genres = [g for sub in self.df['genres_list'] for g in sub]
        return pd.Series(genres).value_counts()

    def all_genres(self):
        return sorted({g for sub in self.df['genres_list'] for g in sub})

    def movie_detail(self, title: str):
        idx = self._resolve_index(title)
        return self.df.iloc[idx] if idx is not None else None
