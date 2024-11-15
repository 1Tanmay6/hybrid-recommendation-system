# %%
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

# Load data
movie_data = pd.read_csv('data/movies_grouplens/movies.dat',
                         delimiter='::', header=None, engine='python', encoding='latin1')
rating_data = pd.read_csv('data/movies_grouplens/ratings.dat',
                          delimiter='::', header=None, engine='python', encoding='latin1')
user_data = pd.read_csv('data/movies_grouplens/users.dat',
                        delimiter='::', header=None, engine='python', encoding='latin1')

# Set column names
movie_data.columns = ['MovieID', 'Title', 'Genres']
rating_data.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
user_data.columns = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']

# Fill missing genres with empty strings
movie_data['Genres'] = movie_data['Genres'].fillna('')

# Create the TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movie_data['Genres'])

# Merge rating and movie data
user_movie_data = rating_data.merge(movie_data, on='MovieID')

# Create an index to locate movies by MovieID
movie_index = pd.Series(
    movie_data.index, index=movie_data['MovieID']).drop_duplicates()

# Initialize user profiles dictionary
user_profiles = {}

# Build user profiles
for user_id in rating_data['UserID'].unique():
    user_ratings = user_movie_data[user_movie_data['UserID'] == user_id]

    # Get TF-IDF weights for movies rated by the user
    tfidf_weights = tfidf_matrix[user_ratings['MovieID'].apply(
        lambda x: movie_index[x])]
    user_profile = np.dot(
        user_ratings['Rating'], tfidf_weights.toarray()) / user_ratings['Rating'].sum()

    # Normalize the user profile
    user_profile = user_profile / np.linalg.norm(user_profile)

    user_profiles[user_id] = user_profile

# Function to recommend movies for a user, excluding already rated ones

# Function to recommend movies for a user, retaining all highly-rated ones (rating >= 4) for precision@k calculation


def recommend_movies_for_user(user_id):
    user_profile = user_profiles[user_id]

    # Calculate similarity scores with all movies
    sim_scores = cosine_similarity([user_profile], tfidf_matrix)[0]

    # Movies the user has already rated with a rating >= 4
    relevant_rated_movies = user_movie_data[(user_movie_data['UserID'] == user_id) & (
        user_movie_data['Rating'] >= 4)]['MovieID'].values

    all_rated_movies = user_movie_data[user_movie_data['UserID']
                                       == user_id]['MovieID'].values

    # Filter similarity scores, keeping all relevant rated movies (rating >= 4)
    filtered_sim_scores = [(i, sim_scores[i]) for i in range(len(sim_scores))
                           if movie_data['MovieID'].iloc[i] not in all_rated_movies or movie_data['MovieID'].iloc[i] in relevant_rated_movies]

    # Sort by similarity score (highest first)
    filtered_sim_scores.sort(key=lambda x: x[1], reverse=True)
    recommended_movie_indices = [idx for idx, _ in filtered_sim_scores]

    return movie_data['Title'].iloc[recommended_movie_indices]


def precision_at_k(user_id, k=10, relevant_threshold=4.0):
    user_ratings = user_movie_data[user_movie_data['UserID'] == user_id]
    relevant_movies = set(
        user_ratings[user_ratings['Rating'] >= relevant_threshold]['MovieID'])

    recommendations = recommend_movies_for_user(user_id)

    recommended_movie_ids = movie_data[movie_data['Title'].isin(
        recommendations[:k])]['MovieID'].values

    relevant_recommendations = sum(
        1 for movie_id in recommended_movie_ids if movie_id in relevant_movies)

    precision = relevant_recommendations / k
    return precision


def recall_at_k(user_id, k=10, relevant_threshold=4.0):
    user_ratings = user_movie_data[user_movie_data['UserID'] == user_id]
    relevant_movies = set(
        user_ratings[user_ratings['Rating'] >= relevant_threshold]['MovieID'])

    recommendations = recommend_movies_for_user(user_id)

    recommended_movie_ids = movie_data[movie_data['Title'].isin(
        recommendations[:k])]['MovieID'].values

    relevant_recommendations = sum(
        1 for movie_id in recommended_movie_ids if movie_id in relevant_movies)

    recall = relevant_recommendations / \
        len(relevant_movies) if len(relevant_movies) > 0 else 0
    return recall


def ndcg_at_k(user_id, k=10, relevant_threshold=4.0):
    user_ratings = user_movie_data[user_movie_data['UserID'] == user_id]

    relevance_scores = {row['MovieID']: (1 if row['Rating'] >= relevant_threshold else 0)
                        for _, row in user_ratings.iterrows()}

    recommendations = recommend_movies_for_user(user_id)

    recommended_movie_ids = movie_data[movie_data['Title'].isin(
        recommendations[:k])]['MovieID'].values

    dcg = 0.0
    for i, movie_id in enumerate(recommended_movie_ids[:k], start=1):
        relevance = relevance_scores.get(movie_id, 0)
        dcg += relevance / np.log2(i + 1)

    ideal_relevance_scores = sorted(
        relevance_scores.values(), reverse=True)[:k]
    idcg = sum(rel / np.log2(i + 1)
               for i, rel in enumerate(ideal_relevance_scores, start=1))

    ndcg = dcg / idcg if idcg > 0 else 0
    return ndcg


k_values = [1, 5, 10, 15, 20, 25, 40, 50]
length_of_users = len(set(user_movie_data['UserID'].values))

#! For Precision Calculation
# precisions = {}
# for k in k_values:
#     precision = []
#     print(f'Starting Precision Calculation for k at {k}')
#     for user_id in set(user_movie_data['UserID'].values):
#         precision.append(precision_at_k(user_id, k))
#         if len(precision) % 604 == 0:
#             print('Completed:', (len(precision)/length_of_users)*100, '%')
#     precisions[k] = np.mean(precision)
#     print(f"Precision@{k}: {precisions[k]}")
# print('*'*20)
# print(precisions)

#! Run this
# recalls = {}
# for k in k_values:
#     recall = []
#     print(f'Starting Recall Calculation for k at {k}')
#     for user_id in set(user_movie_data['UserID'].values):
#         recall.append(recall_at_k(user_id, k))
#         if len(recall) % 604 == 0:
#             print('Completed:', len(recall) / length_of_users*100, '%')
#     recalls[k] = np.mean(recall)
#     print(f"Recall@{k}: {recalls[k]}")
# print('*'*20)
# print(recalls)

#! Run this
ncdgs = {}
for k in k_values:
    ncdg = []
    print(f'Starting NCDG Calculation for k at {k}')
    for user_id in set(user_movie_data['UserID'].values):
        ncdg.append(ndcg_at_k(user_id, k))
        if len(ncdg) % 604 == 0:
            print('Completed:', len(ncdg) / length_of_users*100, '%')
    ncdgs[k] = np.mean(ncdg)
    print(f"NCDG@{k}: {ncdgs[k]}")
print('*'*20)
print(ncdgs)
