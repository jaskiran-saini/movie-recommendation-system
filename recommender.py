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
print(f"Movies only: {movies.shape}")

