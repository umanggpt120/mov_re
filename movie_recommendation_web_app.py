import streamlit as st
import pickle
import difflib
import numpy as np

@st.cache_resource
def load_files():

    with open("movies_data.pkl", "rb") as f:
        movie_data = pickle.load(f)

    with open("similarity.pkl", "rb") as f:
        similarity = pickle.load(f)

    return movie_data, similarity


movie_data, similarity = load_files()


def predictive_system(movie_name):

    list_of_titles = movie_data['title'].tolist()

    close_match = difflib.get_close_matches(
        movie_name,
        list_of_titles
    )

    if not close_match:
        return np.array(["No match found. Try again."])

    close = close_match[0]

    index = movie_data[
        movie_data.title == close
    ].index[0]

    similarity_score = list(
        enumerate(similarity[index])
    )

    sorted_movies = sorted(
        similarity_score,
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for i, movie in enumerate(sorted_movies[:30], start=1):

        movie_index = movie[0]

        title = movie_data.iloc[movie_index]['title']

        recommendations.append(
            f"{i}. {title}"
        )

    return np.array(recommendations)


def main():

    st.title("Movie Recommendation System")

    movie_name = st.text_input(
        "Enter your favourite movie."
    )

    if st.button("Recommend Movies"):

        result = predictive_system(movie_name)

        for movie in result:
            st.success(movie)


if __name__ == "__main__":
    main()
