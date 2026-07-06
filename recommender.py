import pandas as pd

#LOADING DATASET
basics=pd.read_csv('Data/title.basics.tsv.gz',sep='\t',na_values='\\N',low_memory=False)
ratings=pd.read_csv('Data/title.ratings.tsv.gz',sep='\t',na_values='\\N')
crew=pd.read_csv('Data/title.crew.tsv.gz',sep='\t',na_values='\\N')
names=pd.read_csv('Data/name.basics.tsv.gz',sep='\t',na_values='\\N')
"""print(basics.shape,ratings.shape,crew.shape,names.shape)
print(basics.head())"""

#FILTERING DATA
movies=basics[basics['titleType']=='movie'].copy()
#print(f"Movies only: {movies.shape}")

#MERGE RATINGS AND CREW INTO MOVIES
movies=movies.merge(ratings,on='tconst',how='left')
movies=movies.merge(crew,on='tconst',how='left')
#print(f"After merge={movies.shape}")

#DATA CLEANING
movies=movies[movies['averageRating'].notna()]      #drop unrated movies
movies=movies[movies['genres'].notna()]     #drop missing genres
movies=movies[movies['numVotes']>=500]      #keep movies with enough votes
#print(f"After cleaning= {movies.shape}")
#print(movies[['primaryTitle','genres','averageRating','numVotes','directors']].head())

#FEATURE ENGINEERING
#---convert director Ids to actual names---building a lookup dictionary---
nameLookup=names.set_index('nconst')['primaryName'].to_dict()
def getDirectorNames(director_str):
    if pd.isna(director_str):
        return ''
    ids=director_str.split(',')
    return ' '.join([nameLookup.get(i,'')for i in ids])
movies['directorNames']=movies['directors'].apply(getDirectorNames)
#---clean genres---replace commas with spaces so TF-IDF treats each genre as a word---
movies['genresClean']=movies['genres'].str.replace(',',' ')
#---normalize rating to a label---
movies['ratingLabel']=pd.cut(movies['averageRating'],bins=[0,4,6,7,8,10],labels=['bad','average','good','great','excellent'])
#---combine all features into one string---
movies['features']=(movies['genresClean']+' '+movies['directorNames']+' '+movies['ratingLabel'].astype(str))
print(movies[['primaryTitle','genresClean','directorNames','ratingLabel','features']].head())