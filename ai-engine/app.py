import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from thefuzz import process
from supabase import create_client, Client

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION (PASTE YOUR KEYS HERE) ---
SUPABASE_URL = "PASTE_YOUR_PROJECT_URL_HERE"
SUPABASE_KEY = "PASTE_YOUR_ANON_KEY_HERE"

# Connect to Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Load the CSV Logic (Read-Only Brain)
try:
    df = pd.read_csv('anime.csv')
    df['genre'] = df['genre'].fillna('')
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['genre'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df.index, index=df['name']).drop_duplicates()
except:
    print("CSV Error")
    df = pd.DataFrame()

# --- ROUTES ---

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    user_input = data.get('anime', '').strip()

    # 1. Fuzzy Search
    all_names = df['name'].tolist()
    best_match = process.extractOne(user_input, all_names)
    if best_match[1] < 50: return jsonify({"recommendations": [], "status": "Not found"}), 404
    matched_name = best_match[0]

    # 2. Get Standard Recommendations
    idx = indices[matched_name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    anime_indices = [i[0] for i in sim_scores]
    recommendations = df['name'].iloc[anime_indices].tolist()

    # 3. Fetch Discussions from SUPABASE (Real Database)
    try:
        response = supabase.table('discussions').select("*").eq('anime', matched_name).execute()
        chat_history = response.data
    except Exception as e:
        print(e)
        chat_history = []

    return jsonify({
        "recommendations": recommendations,
        "matched_name": matched_name,
        "discussions": chat_history
    })

@app.route('/discuss', methods=['POST'])
def post_discussion():
    data = request.get_json()
    
    # Insert into Supabase
    new_comment = {
        "anime": data['anime'],
        "text": data['text']
    }
    supabase.table('discussions').insert(new_comment).execute()
    
    return jsonify({"status": "Saved to DB"})

@app.route('/rate', methods=['POST'])
def rate_anime():
    data = request.get_json()
    
    # Insert into Supabase
    new_rating = {
        "anime": data['anime'],
        "score": int(data['score'])
    }
    supabase.table('ratings').insert(new_rating).execute()
    
    return jsonify({"status": "Rating Saved"})

if __name__ == '__main__':
    app.run(debug=True)