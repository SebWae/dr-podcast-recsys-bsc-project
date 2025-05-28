import json
import math
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

# adding the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))

from config import (
    TRAIN_DATA_PATH,
    EMBEDDINGS_DESCR_PATH,
    SCORES_PATH_CF_INCL_VAL,
    UTILS_INTERACTIONS_PATH,
    EMBEDDING_DIM,
    WGHT_METHOD,
    LAMBDA_HYBRID,
    N_RECOMMENDATIONS,
    RECOMMENDATIONS_PATH,
    RECOMMENDATIONS_KEY_HYBRID,
)
import utils as utils


# loading train, validation data and descr embeddings
print("Loading data and embeddings.")
train_df = pd.read_parquet(TRAIN_DATA_PATH)
emb_df = pd.read_parquet(EMBEDDINGS_DESCR_PATH)

# loading cf scores
print("Loading scores from cf recommender.")
cf_scores_df = pd.read_parquet(SCORES_PATH_CF_INCL_VAL)

# converting the scores dataframes to dictionaries
print("Converting scores dataframe to dictionary.")
cf_scores = cf_scores_df.to_dict()

# all users
users = set(cf_scores.keys())

# extracting embeddings and storing them in a dictionary
print("Generating embedding dictionary and array.")
emb_dict = {}
for _, row in tqdm(emb_df.iterrows()):
    prd_number = row["episode"]
    embedding = row[1:].values.flatten()
    emb_dict[prd_number] = embedding

# item embeddings as numpy array
items = emb_dict.keys()
item_embeddings = np.array([emb_dict[item] for item in items], dtype=np.float64)

# loading utils dictionaries
print(f"Loading utils dictionaries from {UTILS_INTERACTIONS_PATH}")
with open(UTILS_INTERACTIONS_PATH, "r") as file:
    utils_dicts = json.load(file)

all_users_show_episodes_dict = utils_dicts["user_show_episodes_val"]

# generating user profiles
print("Generating user profiles.")
user_profiles = {}
for user in tqdm(users):
    # initialize user profile (embedding)
    user_interactions = train_df[train_df["user_id"] == user].reset_index()
    user_profile = utils.get_user_profile(emb_size=EMBEDDING_DIM,
                                          user_int=user_interactions,
                                          time_col="days_since",
                                          item_col="prd_number",
                                          emb_dict=emb_dict,
                                          wght_scheme=WGHT_METHOD)
    user_profiles[user] = user_profile

# generating hybrid scores and recommendations for each user
print("Generating hybrid scores and recommendations for each user.")
recs_dict = {}
for user in tqdm(users):
    # retrieving user profile from dictionary
    user_profile = user_profiles[user]

    # scores for user for each item not consumed by the user
    cb_scores_user = utils.get_cb_scores(user=user,
                                         show_episodes=all_users_show_episodes_dict,
                                         user_profile=user_profile,
                                         item_embeddings=item_embeddings,
                                         items=items)

    # retrieving cf scores for user
    cf_scores_user = cf_scores[user]

    # computing hybrid scores
    hybrid_scores = {}
    for item in items:
        cf_score = cf_scores_user[item] if item in cf_scores_user else 0
        cb_score = cb_scores_user[item] if item in cb_scores_user else 0
        hybrid_score = LAMBDA_HYBRID * cf_score + (1 - LAMBDA_HYBRID) * cb_score
        hybrid_scores[item] = hybrid_score
    
    # retrieving recommendations from item scores dictionary
    sorted_scores = dict(
        sorted(
            hybrid_scores.items(),
            key=lambda item: 0 if item[1] is None or (isinstance(item[1], float) and math.isnan(item[1])) else item[1],
            reverse=True
        )
    )
    recs = list(sorted_scores.keys())[:N_RECOMMENDATIONS]
    recs_dict[user] = recs

# saving recommendations
print(f"Saving recommendations to {RECOMMENDATIONS_PATH}.")
recs_dict_key = {RECOMMENDATIONS_KEY_HYBRID: recs_dict}
utils.save_dict_to_json(data_dict=recs_dict_key, 
                        file_path=RECOMMENDATIONS_PATH)
