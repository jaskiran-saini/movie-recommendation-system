import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity 
from recommender import movies,tfidfMatrix

#PAGE CONFIGURATION
st.set_page_config(page_title="Movie Recommender", page_icon="🎬",layout="wide")

#TITLE
st.title("🎬 MOVIE RECOMMENDATION SYSTEM")
st.markdown("Built using IMDB dataset . TF-IDF . Cosine Similarity")
st.divider()