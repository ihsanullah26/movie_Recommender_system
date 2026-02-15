import pandas as pd
from lightfm.data import Dataset
from lightfm import LightFM
import numpy as np
from lightfm.evaluation import precision_at_k, auc_score
import pickle

Movies = pd.read_csv('movies.dat',sep="::",engine='python',names=['movie_id','title','genre'],encoding='latin-1')
Users = pd.read_csv('users.dat',sep="::",engine='python',names=['user_id','gender','age','occupation','zip-code'],encoding='latin-1')
Ratings = pd.read_csv('ratings.dat',sep="::",engine='python',names=['user-id','movie_id','rating','TimeStamp'],encoding='latin-1')


Movies['genre_list'] = Movies['genre'].str.split('|')

#finding all unique genres types
all_unique_genre= set(g for genre in Movies['genre_list'] for g in genre)

#find 

Users['gender_tag'] = "Gender_"+Users['gender']
Users['age_tag'] = "Age_"+Users['age'].astype(str) 
all_user_tags = set(Users['gender_tag']).union(set(Users['age_tag']))
print(all_user_tags)

dataset = Dataset()

dataset.fit(
    users=Users['user_id'],
    items=Movies['movie_id'],
    user_features=all_user_tags,
    item_features=all_unique_genre
)

#Build the interaction table
(interactions,weight) = dataset.build_interactions(
    (row['user-id'],row['movie_id'],row['rating'])
    for _,row in Ratings.iterrows()
)


# 3. Build Item Features (Content-based)
item_features = dataset.build_item_features(
    (row['movie_id'], row['genre_list']) 
    for _, row in Movies.iterrows()
)

# 4. Build User Features (Cold-Start demographic info)
user_features = dataset.build_user_features(
    (row['user_id'], [row['gender_tag'], row['age_tag']]) 
    for _, row in Users.iterrows()
)

model = LightFM(no_components=30,loss='warp',learning_rate = 0.05)

model.fit(
    interactions,
    user_features = user_features,
    item_features = item_features,
    epochs = 20
)


# Calculate how many of the top 10 recommended movies were actually liked by users
train_precision = precision_at_k(model, interactions, user_features=user_features, item_features=item_features, k=10).mean()

print(f"Precision at K: {train_precision:.2f}")

# Save the model
with open('lightfm_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save the dataset (essential for mapping new user features)
with open('dataset.pkl', 'wb') as f:
    pickle.dump(dataset, f)

# Map MovieID to Titles for easy lookup later
movie_id_to_title = dict(zip(Movies['movie_id'], Movies['title']))
with open('movie_titles.pkl', 'wb') as f:
    pickle.dump(movie_id_to_title, f)
    
print("Model and Dataset saved successfully!")
