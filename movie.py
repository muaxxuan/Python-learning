# Movie recommendation program using file.tsv and Movie_Id_Titles.csv
# import pandas library
import inline as inline
import matplotlib
import pandas as pd
# get the data
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
dataset = pd.read_csv("file.tsv", sep='\t', names=column_names)
print(dataset.head())
# read all movies and their IDs
movie_titles = pd.read_csv("Movie_Id_Titles.csv")
print(movie_titles)
# merge both dataset with movie titles
dataset_new = pd.merge(dataset, movie_titles, on='item_id')
print(dataset_new)
# calculate mean rating of all movies
print(dataset_new.groupby('title')['rating'].mean().sort_values(ascending=False).head())
# calculate count rating of all movies
print(dataset_new.groupby('title')['rating'].count().sort_values(ascending=False).head())
# create dataframe with 'rating' count values
ratings = pd.DataFrame(dataset_new.groupby('title')['rating'].mean())
ratings['num of ratings'] = pd.DataFrame(dataset_new.groupby('title')['rating'].count())
print(ratings.head())
# visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
# %matplotlib inline - add if want to displays output in jupyter notebook
# plot graph of num of ratings column
plt.figure(figsize=(10, 4))
ratings['num of ratings'].hist(bins=70)
plt.show()
# plot graph of ratings column
plt.figure(figsize=(10, 4))
ratings['rating'].hist(bins=70)
plt.show()
# sorting values according to the num of rating column
moviemat = dataset_new.pivot_table(index='user_id',
                            columns='title', values='rating')
print(moviemat.head())
print(ratings.sort_values('num of ratings', ascending=False).head(10))
# analysing correlation with similar movies
starwars_user_rating = moviemat['Star Wars (1977)']
liarliar_user_rating = moviemat['Liar Liar (1997)']
starwars_user_rating.head()
# analysing correlation with similar movies
similar_to_starwars = moviemat.corrwith(starwars_user_rating)
similar_to_liarliar = moviemat.corrwith(liarliar_user_rating)

corr_starwars = pd.DataFrame(similar_to_starwars, columns=['Correlation'])
corr_starwars.dropna(inplace=True)
print(corr_starwars.head())
#  similar movies like starwars
corr_starwars.sort_values('Correlation', ascending=False).head(10)
corr_starwars = corr_starwars.join(ratings['num of ratings'])

corr_starwars.head()
print(corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation', ascending=False).head())
# Similar movies as of liarliar
corr_liarliar = pd.DataFrame(similar_to_liarliar, columns=['Correlation'])
corr_liarliar.dropna(inplace=True)

corr_liarliar = corr_liarliar.join(ratings['num of ratings'])
print(corr_liarliar[corr_liarliar['num of ratings'] > 100].sort_values('Correlation', ascending=False).head())
