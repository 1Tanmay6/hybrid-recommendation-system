import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error


class CBF:
    def __init__(self):
        self.movie_data = pd.read_csv('data/movies_grouplens/movies.dat',
                                      delimiter='::', header=None, engine='python', encoding='latin1')
        self.rating_data = pd.read_csv('data/movies_grouplens/ratings.dat',
                                       delimiter='::', header=None, engine='python', encoding='latin1')
        self.user_data = pd.read_csv('data/movies_grouplens/users.dat',
                                     delimiter='::', header=None, engine='python', encoding='latin1')
        self.movie_data.columns = ['MovieID', 'Title', 'Genres']
        self.rating_data.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
        self.user_data.columns = ['UserID', 'Gender',
                                  'Age', 'Occupation', 'Zip-code']

        self.movie_data['Genres'] = self.movie_data['Genres'].fillna('')

        self.user_movie_data = self.rating_data.merge(
            self.movie_data, on='MovieID')

        with open('models/CBF/tfidf_matrix.pkl', 'rb') as f:
            self.tfidf_matrix = pickle.load(f)

        with open('models/CBF/tfidf_vectorizer.pkl', 'rb') as f:
            self.tfidf = pickle.load(f)

        with open('models/CBF/user_profiles.pkl', 'rb') as f:
            self.user_profiles = pickle.load(f)

    def cbf_recommend_movies(self, user_id, top_n=5):

        if user_id not in self.user_profiles:
            print(f"No profile found for UserID {user_id}.")
            return None

        user_profile = self.user_profiles[user_id]

        similarity_scores = cosine_similarity(
            [user_profile], self.tfidf_matrix).flatten()

        scaled_scores = similarity_scores * 5

        movie_scores = pd.DataFrame({
            'MovieID': self.movie_data['MovieID'],
            'Score': scaled_scores
        })

        watched_movie_ids = []
        movie_scores = movie_scores[~movie_scores['MovieID'].isin(
            watched_movie_ids)]

        top_recommendations = movie_scores.sort_values(
            by='Score', ascending=False)

        top_recommendations = top_recommendations.merge(
            self.movie_data, on='MovieID', how='left')

        return top_recommendations[['MovieID', 'Title', 'Genres', 'Score']]


if __name__ == '__main__':
    cbf = CBF()
    print(cbf.cbf_recommend_movies(400))
