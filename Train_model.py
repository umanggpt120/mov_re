
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movie_data = pd.read_csv("movies.csv")

selected_features = [
    'genres',
    'keywords',
    'tagline',
    'cast',
    'director'
]

for feature in selected_features:
    movie_data[feature] = movie_data[feature].fillna('')

combined_features = (
    movie_data['genres'] + ' ' +
    movie_data['keywords'] + ' ' +
    movie_data['tagline'] + ' ' +
    movie_data['cast'] + ' ' +
    movie_data['director']
)

vectorizer = TfidfVectorizer()

feature_vectors = vectorizer.fit_transform(combined_features)

similarity = cosine_similarity(feature_vectors)


with open("movies_data.pkl", "wb") as f:
    pickle.dump(movie_data, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("similarity.pkl", "wb") as f:
    pickle.dump(similarity, f)

print("Files saved successfully")
