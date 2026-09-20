import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from recommender import movies, tfidf_matrix

# Page config
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

# Title
st.title("🎬 Movie Recommendation System")
st.markdown("Built using IMDB dataset · TF-IDF · Cosine Similarity")
st.divider()

# Input
movie_input = st.text_input("Enter a movie name:", placeholder="e.g. The Dark Knight")
n = st.slider("Number of recommendations:", min_value=5, max_value=20, value=10)

if st.button("Recommend"):
    if movie_input.strip() == "":
        st.warning("Please enter a movie name.")
    else:
        matches = movies[movies['primaryTitle'].str.lower() == movie_input.lower()]
        if matches.empty:
            matches = movies[movies['primaryTitle'].str.lower().str.contains(movie_input.lower())]
        if matches.empty:
            st.error(f"Movie '{movie_input}' not found!")
        else:
            idx = matches.index[0]
            pos = movies.index.get_loc(idx)
            sim_scores = cosine_similarity(tfidf_matrix[pos], tfidf_matrix).flatten()
            sim_scores[pos] = 0
            top_indices = sim_scores.argsort()[::-1][:n]

            results = movies.iloc[top_indices]
            results = results[['primaryTitle', 'genres', 'averageRating', 'numVotes']].copy()
            results['similarity'] = sim_scores[top_indices].round(3)
            results = results.reset_index(drop=True)
            results.index += 1

            st.success(f"Top {n} movies similar to **{matches.iloc[0]['primaryTitle']}**:")
            st.dataframe(results, width='stretch')