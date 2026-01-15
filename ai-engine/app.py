from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from thefuzz import process

app = Flask(__name__)
CORS(app)

# Load the dataset
# Make sure your csv file is named correctly here!
try:
    df = pd.read_csv('anime.csv')
    # Fill missing values to avoid errors
    df['genre'] = df['genre'].fillna('')
except:
    print("Error: anime.csv not found. Make sure it is in the ai-engine folder.")
    df = pd.DataFrame()

# 1. TRAIN THE AI (TF-IDF)
# We combine Genres and Type to create a "content profile" for each anime
if not df.empty:
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['genre'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    # Create a mapping of Anime Name -> Index
    indices = pd.Series(df.index, index=df['name']).drop_duplicates()

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    user_input = data.get('anime', '').strip()

    if df.empty:
        return jsonify({"error": "Database not loaded"}), 500

    # --- FUZZY MATCHING (THE FIX) ---
    # Get a list of all anime names
    all_names = df['name'].tolist()
    
    # Find the single closest match to what the user typed
    # limit=1 means "give me the best one"
    best_match = process.extractOne(user_input, all_names)
    
    # best_match looks like: ("Naruto", 90) where 90 is the confidence score
    matched_name = best_match[0]
    score = best_match[1]

    # If the match is too weak (e.g., user typed "sdlkfjsd"), don't guess
    if score < 50:
        return jsonify({"recommendations": [], "status": "Not found"}), 404

    # Now use the CORRECTED name to find recommendations
    try:
        idx = indices[matched_name]
        
        # Get similarity scores for this anime
        sim_scores = list(enumerate(cosine_sim[idx]))
        
        # Sort them (highest score first)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get the top 5 (skipping the first one because it is the anime itself)
        sim_scores = sim_scores[1:6]
        
        # Get the anime names
        anime_indices = [i[0] for i in sim_scores]
        recommendations = df['name'].iloc[anime_indices].tolist()

        return jsonify({
            "recommendations": recommendations,
            "original_query": user_input,
            "matched_name": matched_name # Sending this back so Frontend knows!
        })

    except KeyError:
        return jsonify({"recommendations": []}), 404

if __name__ == '__main__':
    app.run(debug=True)