from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the dataframe and the recommendation model
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    lat = float(request.form['latitude'])
    long = float(request.form['longitude'])
    keyword = request.form['keyword']
    state = request.form['state']
    
    recommendations = recommend(lat, long, keyword, state)
    
    return render_template('results.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
