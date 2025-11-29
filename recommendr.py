# src/recommender.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class HybridRecommender:
    def __init__(self, movies_df):
        self.movies = movies_df.copy().reset_index(drop=True)
        self._build_content_matrix()

    def _build_content_matrix(self):
        # use 'description' and 'genres' as content features
        docs = (self.movies['description'].fillna('') + ' ' + self.movies['genres'].fillna('')).values
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        self.content_matrix = self.tfidf.fit_transform(docs)

    def recommend_by_title(self, title, topn=5):
        idx = self.movies[self.movies['title'].str.lower() == title.lower()].index
        if len(idx) == 0:
            return []
        idx = idx[0]
        sim = cosine_similarity(self.content_matrix[idx], self.content_matrix).flatten()
        similar_indices = np.argsort(-sim)[1:topn+1]
        return self.movies.loc[similar_indices][['movieId','title','genres']].to_dict(orient='records')

if __name__ == '__main__':
    df = pd.read_csv('data/sample_movies.csv')
    rec = HybridRecommender(df)
    print('Recommendations for "Toy Story (1995)":')
    print(rec.recommend_by_title('Toy Story (1995)', topn=3))
