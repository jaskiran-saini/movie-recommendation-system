import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_resource
def load_data():
    movies = pd.read_csv('data/movies_clean.csv')
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['features'])
    return movies, tfidf_matrix

movies, tfidf_matrix = load_data()