import argparse
from collections import defaultdict
import csv
from datetime import datetime
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# adding the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))

from config import (
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    METADATA_PATH,
    EMBEDDINGS_TITLE_PATH,
    EMBEDDINGS_DESCR_PATH,
    UTILS_PATH,
    SPLIT_DATE_TRAIN_VAL,
    EMBEDDING_DIM,
    N_RECOMMENDATIONS,
    EXPERIMENTS_CB_PATH,
)
import utils.utils as utils


# argument parser for input parameters
print("Parsing the input arguments.")
parser = argparse.ArgumentParser(description="Run content-based combi experiments with inputs for the lambda hyperparameter.")
parser.add_argument("--lambda_vals", type=str, required=True, help="Comma-separated sequence of lambda values: x,y,z")
parser.add_argument("--wght_method", type=str, required=True, help="String, one of 'inverse' or 'linear'")
args = parser.parse_args()

# parsing the input arguments
lambdas = [float(x) for x in args.lambda_vals.split(",")]
wght_method = args.wght_method

# loading data and embeddings
print("Loading data and embeddings.")
train_df = pd.read_parquet(TRAIN_DATA_PATH)
val_df = pd.read_parquet(VAL_DATA_PATH)
meta_df = pd.read_parquet(METADATA_PATH)

# loading embeddings
title_emb_df = pd.read_parquet(EMBEDDINGS_TITLE_PATH)
descr_emb_df = pd.read_parquet(EMBEDDINGS_DESCR_PATH)

# list of unique users in train data and unique items in metadata
users = train_df["user_id"].unique()
items = meta_df["prd_number"].unique()

# adding days since train-val split date as a column in the train_df
print("Computing days since train-val date for each train interaction.")
reference_date = datetime.strptime(SPLIT_DATE_TRAIN_VAL, "%Y-%m-%d")
train_df["date"] = pd.to_datetime(train_df["date"])
train_df["days_since"] = (reference_date - train_df["date"]).dt.days

# loading utils dictionaries
print(f"Loading utils dictionaries from {UTILS_PATH}")
with open(UTILS_PATH, "r") as file:
    utils_dicts = json.load(file)

all_users_show_episodes_dict = utils_dicts["user_show_episodes"]

# dictionary containing completion rates for each user and consumed item in the validation data
print("Generating completion_rates_dict from validation data.")
completion_rates_dict = utils.get_ratings_dict(data=val_df, 
                                               user_col="user_id", 
                                               item_col="prd_number", 
                                               ratings_col="completion_rate") 

# hyperparameter tuning for _lambda (weighting hyperparameter)
print("Performing hyperparameter tuning for weighting parameter lambda.")
print(f"Lambda values to test: {lambdas}.")
print(f"Weighting method: {wght_method}")

for _lambda in tqdm(lambdas):
    print(f"\nTesting lambda={_lambda}.")
    # initializing dictionary to store scores for each user
    scores_dict = defaultdict(dict)

    # extracting embeddings and storing them in a dictionary
    emb_dict = {}
    for _, row in title_emb_df.iterrows():
        prd_number = row["episode"]
        title_embedding = row[1:].values.flatten()
        descr_embedding = descr_emb_df[descr_emb_df["episode"] == prd_number].iloc[:, 1:].values.flatten()
        embedding = _lambda * title_embedding + (1 - _lambda) * descr_embedding
        emb_dict[prd_number] = embedding

    # generate user profiles
    print("Generating user profile and recommendations for each user.")
    for user in users:
        # initialize user profile (embedding)
        user_interactions = train_df[train_df["user_id"] == user]
        user_profile = utils.get_user_profile(emb_size=EMBEDDING_DIM,
                                              user_int=user_interactions,
                                              time_col="days_since",
                                              item_col="prd_number",
                                              emb_dict=emb_dict,
                                              wght_method=wght_method)

        # initializing dictionary to store user scores
        user_scores = {}

        # items consumed by user
        user_show_episodes_dict = all_users_show_episodes_dict[user]
        user_items = [item for sublist in user_show_episodes_dict.values() for item in sublist]
        
        for item in items:
            # only computing scores for non-consumed items
            if item not in user_items:
                embedding = emb_dict[item]
                cos_sim = cosine_similarity(user_profile, embedding)
            user_scores[item] = cos_sim
        
        # normalizing the user scores
        values = np.array(list(user_scores.values()))
        norm = np.linalg.norm(values)
        normalized_user_scores = {key: value / norm for key, value in user_scores.items()}
        
        # adding user_scores to scores_dict
        scores_dict[user] = normalized_user_scores

    # extract recommendations from scores
    recs_dict = utils.extract_recs(scores_dict=scores_dict, n_recs=N_RECOMMENDATIONS)

    # computing ndcg@10
    ndcgs = []
    for user_id, rec_items in recs_dict.items():
        gain_dict = completion_rates_dict[user_id]
        optimal_items = sorted(gain_dict, key=lambda x: gain_dict[x], reverse=True)[:N_RECOMMENDATIONS]
        dcg = utils.compute_dcg(rec_items, gain_dict)
        dcg_star = utils.compute_dcg(optimal_items, gain_dict)
        ndcg_user = dcg / dcg_star 
        ndcgs.append(ndcg_user)

    ndcg = np.mean(ndcgs)
    print(f"ndcg@10: {ndcg:.10f}")

    # saving experiment result
    print(f"Saving experiment results to {EXPERIMENTS_CB_PATH}.")

    row = [_lambda, ndcg]
    with open(EXPERIMENTS_CB_PATH, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(row)
