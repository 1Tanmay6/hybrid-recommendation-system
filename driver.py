import joblib
import pandas as pd


df = pd.read_csv('database_data/business_recommendation_system_input.csv')
model = joblib.load('models/knn_model_recommendation_system_business.joblib')


def recommend(lat, long, keyword, state):
    df_state = df[df['state'] == state]
    
    if df_state.empty:
        return "No businesses found in the given state."
    _, indices = model.kneighbors([[lat, long]], n_neighbors=100)
    results = df.iloc[indices[0]]
    results = results[results['state'] == state]
    results = results[results['categories'].str.contains(keyword, case=False, na=False)]
    results = results.sort_values(['stars', 'review_count'], ascending=False)
    if results.empty:
        return "No businesses found with the given keyword in the categories list."
    
    return results['business_id'].to_list()


latitude = 34.42667934
longitude = -119.711197
preferences = 'doc'
state = 'CA'
print(recommend(latitude, longitude, preferences, state))
