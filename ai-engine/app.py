from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from thefuzz import process
import datetime

app = Flask(__name__)
CORS(app)

# --- 1. THE DATABASE (Simulated for now) ---
# In a real startup, this would be PostgreSQL or MongoDB
discussions_db = []  # Stores comments: {"anime": "Naruto", "user": "Anon", "text": "Best arc!"}
ratings_db = {}      # Stores ratings: {"Naruto": [10, 9, 10], "Bleach": [8, 7]}

# Load the static dataset
try:
    df = pd.read_csv('anime.csv')
    df['genre'] = df['genre'].fillna('')
    
    # Pre-train the AI models
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['genre'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df.index, index=df['name']).drop_duplicates()
    
except Exception as e:
    print(f"Database Error: {e}")
    df = pd.DataFrame()

# --- 2. THE NEW "HYBRID" AI ENGINE ---
def get_hybrid_recommendation(anime_name):
    # A. Standard Content Matching
    idx = indices[anime_name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11] # Get top 10
    
    # B. The "Social Factor" (Ranking Boost)
    # We adjust the score based on what users have rated it in our app
    weighted_results = []
    
    for i, score in sim_scores:
        anime_idx = i
        name = df['name'].iloc[anime_idx]
        
        # Check if users rated this anime in our app
        user_ratings = ratings_db.get(name, [])
        if user_ratings:
            avg_rating = sum(user_ratings) / len(user_ratings)
            # Math: If rating is > 8, boost the similarity score by 10%
            if avg_rating > 8.0:
                score = score * 1.10
        
        weighted_results.append((name, score))
    
    # Re-sort based on new weighted scores
    weighted_results.sort(key=lambda x: x[1], reverse=True)
    
    # Return top 5 names
    return [x[0] for x in weighted_results[:5]]

# --- 3. API ENDPOINTS ---

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    user_input = data.get('anime', '').strip()

    if df.empty: return jsonify({"error": "DB error"}), 500

    # Fuzzy Search logic
    all_names = df['name'].tolist()
    best_match = process.extractOne(user_input, all_names)
    
    if best_match[1] < 50:
        return jsonify({"recommendations": [], "status": "Not found"}), 404

    matched_name = best_match[0]
    
    # Get the Smart Recommendations
    try:
        recommendations = get_hybrid_recommendation(matched_name)
        
        # Also return discussions for this anime
        relevant_chat = [d for d in discussions_db if d['anime'] == matched_name]
        
        return jsonify({
            "recommendations": recommendations,
            "matched_name": matched_name,
            "discussions": relevant_chat
        })
    except KeyError:
        return jsonify({"recommendations": []}), 404

# NEW: Post a Discussion
@app.route('/discuss', methods=['POST'])
def post_discussion():
    data = request.get_json()
    comment = {
        "anime": data['anime'],
        "text": data['text'],
        "date": str(datetime.datetime.now())
    }
    discussions_db.append(comment)
    return jsonify({"status": "Posted!", "comment": comment})

# NEW: Rate an Anime
@app.route('/rate', methods=['POST'])
def rate_anime():
    data = request.get_json()
    anime = data['anime']
    score = int(data['score']) # 1-10
    
    if anime not in ratings_db:
        ratings_db[anime] = []
    ratings_db[anime].append(score)
    
    return jsonify({"status": "Rated!", "new_avg": sum(ratings_db[anime])/len(ratings_db[anime])})

if __name__ == '__main__':
    app.run(debug=True)