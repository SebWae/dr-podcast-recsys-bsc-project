import argparse
import csv
import json
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

# adding the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))

from config import (
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    EMBEDDINGS_DESCR_PATH,
    SCORES_PATH_CF,
    UTILS_PATH,
    EMBEDDING_DIM,
    WGHT_METHOD,
    N_RECOMMENDATIONS,
    EXPERIMENTS_HYBRID_PATH,
)
import utils.utils as utils


# argument parser for input parameters
print("Parsing the input arguments.")
parser = argparse.ArgumentParser(description="Run hybrid recommender experiments with inputs for the lambda hyperparameter.")
parser.add_argument("--lambda_vals", type=str, required=True, help="Comma-separated sequence of lambda values: x,y,z")
args = parser.parse_args()

# parsing the input arguments
lambdas = [float(x) for x in args.lambda_vals.split(",")]

# loading train, validation data and descr embeddings
print("Loading data and embeddings.")
train_df = pd.read_parquet(TRAIN_DATA_PATH)
val_df = pd.read_parquet(VAL_DATA_PATH)
emb_df = pd.read_parquet(EMBEDDINGS_DESCR_PATH)

# loading cf scores
print("Loading scores from cf recommender.")
cf_scores_df = pd.read_parquet(SCORES_PATH_CF)

# converting the scores dataframes to dictionaries
print("Converting scores dataframe to dictionary.")
cf_scores = cf_scores_df.to_dict()

# all users
users = set(cf_scores.keys())

# dictionary containing completion rates for each user and consumed item in the validation data
print("Generating completion_rates_dict from validation data.")
completion_rates_dict = utils.get_ratings_dict(data=val_df, 
                                               user_col="user_id", 
                                               item_col="prd_number", 
                                               ratings_col="completion_rate") 

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
print(f"Loading utils dictionaries from {UTILS_PATH}")
with open(UTILS_PATH, "r") as file:
    utils_dicts = json.load(file)

all_users_show_episodes_dict = utils_dicts["user_show_episodes"]

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

# hyperparameter tuning for _lambda (weighting hyperparameter)
print("Performing hyperparameter tuning for weighting parameter lambda.")
print(f"Lambda values to test: {lambdas}.")

for _lambda in lambdas:
    print(f"\nTesting lambda={_lambda}.")

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
        item_scores = {}
        for item in items:
            cf_score = cf_scores_user[item] if item in cf_scores_user else 0
            cb_score = cb_scores_user[item] if item in cb_scores_user else 0
            hybrid_score = _lambda * cf_score + (1 - _lambda) * cb_score
            item_scores[item] = hybrid_score
        
        # retrieving recommendations from item scores dictionary
        sorted_scores = dict(sorted(item_scores.items(), key=lambda item: item[1], reverse=True))
        recs = list(sorted_scores.keys())[:N_RECOMMENDATIONS]
        recs_dict[user] = recs

    # computing NDCG@10
    print("Computing NDCG@10.")
    ndcgs = []
    for user_id, rec_items in recs_dict.items():
        gain_dict = completion_rates_dict[user_id]
        optimal_items = sorted(gain_dict, key=lambda x: gain_dict[x], reverse=True)[:N_RECOMMENDATIONS]
        dcg = utils.compute_dcg(rec_items, gain_dict)
        dcg_star = utils.compute_dcg(optimal_items, gain_dict)
        ndcg_user = dcg / dcg_star 
        ndcgs.append(ndcg_user)

    ndcg = np.mean(ndcgs)
    print(f"NDCG@10: {ndcg:.10f}")

    # saving experiment result
    print(f"Saving experiment results to {EXPERIMENTS_HYBRID_PATH}.")
    row = [_lambda, ndcg]
    with open(EXPERIMENTS_HYBRID_PATH, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(row)
    