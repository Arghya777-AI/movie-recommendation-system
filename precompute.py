"""
Run this ONCE locally to generate precomputed.json.
Commit that file to the repo so Streamlit Cloud loads instantly.

Usage:
    python precompute.py
"""
import json, os
import pandas as pd
import numpy as np
from recommender import MovieRecommender

def fmt(v):
    try:
        v = float(v)
        if v <= 0 or np.isnan(v): return None
        if v >= 1e9: return f'${v/1e9:.1f}B'
        if v >= 1e6: return f'${v/1e6:.0f}M'
    except Exception:
        pass
    return None

base = os.path.dirname(os.path.abspath(__file__))

print("Loading recommender (builds similarity matrix)…")
rec = MovieRecommender(os.path.join(base, 'tmdb_5000_movies.csv'))
df  = rec.df
sim = rec.cosine_sim_hybrid
titles_arr = df['title'].tolist()

print("Building movie metadata…")
movies = {}
for pos, row in df.iterrows():
    t = row.get('title')
    if not t or pd.isna(t): continue
    movies[t] = {
        'y':  int(row['release_year'])           if pd.notna(row.get('release_year')) else None,
        'r':  round(float(row['vote_average']),1) if pd.notna(row.get('vote_average')) else None,
        'rt': int(row['runtime'])                if pd.notna(row.get('runtime'))      else None,
        'g':  row.get('genres_list', [])[:4],
        'o':  str(row.get('overview', ''))[:200],
        'b':  fmt(row.get('budget')),
        'v':  fmt(row.get('revenue')),
    }

print("Pre-computing top-12 recommendations for every movie…")
recs = {}
for pos in range(len(df)):
    t = titles_arr[pos]
    if not t or pd.isna(t): continue
    scores  = sim[pos]
    top_idx = np.argsort(scores)[::-1][1:13]
    recs[t] = [
        {'t': titles_arr[i], 's': int(round(float(scores[i]) * 100))}
        for i in top_idx
        if i < len(titles_arr) and titles_arr[i] in movies
    ]

print("Computing top picks…")
top_df    = rec.top_movies(n=12)
top_picks = [t for t in top_df['title'].tolist() if t in movies]

out = {'movies': movies, 'recs': recs, 'top': top_picks}
out_path = os.path.join(base, 'precomputed.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, separators=(',', ':'))

size_mb = os.path.getsize(out_path) / 1e6
print(f"Done — saved precomputed.json ({size_mb:.1f} MB)")
print("Now commit and push precomputed.json to your repo.")
