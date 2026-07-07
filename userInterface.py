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

#INPUT
movieInput=st.text_input("Enter a movie name: ",placeholder="e.g. The Dark Knight")
n=st.slider("Number of recommendations: ",min_value=5,max_value=20, value=10)
if st.button("Recommend"):
    if movieInput.strip()=="":
        st.warning("Please enter a movie name!!!")
    else:
        matches=movies[movies['primaryTitle'].str.lower()==movieInput.lower()]
        if matches.empty:
            matches=movies[movies['primaryTitle'].str.lower().str.contains(movieInput.lower())]
        if matches.empty:
            st.error(f"Movie '{movieInput}' not found!!!")
        else:
            idx=matches.index[0]
            pos=movies.index.get_loc(idx)
            simScores=cosine_similarity(tfidfMatrix[pos],tfidfMatrix).flatten()
            simScores[pos]=0
            topIndices=simScores.argsort()[::-1][:n]
            results=movies.iloc[topIndices]
            results=results[['primaryTitle','genres','averageRating','numVotes']].copy()
            results['similarity']=simScores[topIndices].round(3)
            results=results.reset_index(drop=True)
            results.index+=1
            st.success(f"Top {n} movies similar to **{matches.iloc[0]['primaryTitle']}**: ")
            st.dataframe(results,use_container_width=True)