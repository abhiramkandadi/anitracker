from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)  # Allows your Framer site to talk to this API

# --- 1. CREATE DUMMY DATA (Replace this with a real CSV later) ---
data = {
    'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'title': [
        "Attack on Titan", "Naruto", "Death Note", "One Piece", 
        "Demon Slayer", "Fullmetal Alchemist: Brotherhood", 
        "My Hero Academia", "Steins;Gate", "Jujutsu Kaisen", "Cyberpunk: Edgerunners"
    ],
    'genres': [
        "Action Drama Fantasy", "Action Adventure Fantasy", "Mystery Psychological Thriller", 
        "Action Adventure Fantasy", "Action Fantasy Demons", "Action Adventure Drama", 
        "Action Comedy SuperPower", "Sci-Fi Thriller", "Action Supernatural School", "Sci-Fi Action"
    ]
}
df = pd.DataFrame(data)

# --- 2. TRAIN THE AI MODEL ---
# This converts genres into numbers so the AI can understand them
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(df['genres']).toarray()
similarity = cosine_similarity(vectors)

# --- 3. RECOMMENDATION FUNCTION ---
def recommend(anime_title):
    # Find the index of the anime matches the title
    try:
        anime_index = df[df['title'] == anime_title].index[0]
    except IndexError:
        return ["Anime not found in database"]
    
    # Get similarity scores for this anime
    distances = similarity[anime_index]
    
    # Sort the list to find the most similar ones (excluding itself)
    # This logic creates a list of (index, score)
    anime_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:4]
    
    recommendations = []
    for i in anime_list:
        recommendations.append(df.iloc[i[0]].title)
    return recommendations

# --- 4. API ENDPOINT ---
@app.route('/recommend', methods=['POST'])
def get_recommendation():
    user_data = request.json
    liked_anime = user_data.get('anime')
    
    print(f"User likes: {liked_anime}...")
    results = recommend(liked_anime)
    
    return jsonify({
        "original": liked_anime,
        "recommendations": results
    })

@app.route('/', methods=['GET'])
def home():
    return "Anime AI is Running!"

if __name__ == '__main__':
    app.run(debug=True, port=5000)