from collections import defaultdict
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
    METADATA_PATH,
    EMBEDDINGS_TITLE_PATH,
    EMBEDDINGS_DESCR_PATH,
    EMBEDDINGS_COMBI_PATH,
    UTILS_PATH,
    SPLIT_DATE_TRAIN_VAL,
    EMBEDDING_DIM,
    RECOMMENDATIONS_KEY_CB_COMBI,
    RECOMMENDATIONS_KEY_CB_DESCR,
    RECOMMENDATIONS_KEY_CB_TITLE,
    CB_SCORES_PATH,
    N_RECOMMENDATIONS,
    RECOMMENDATIONS_PATH,
)
import utils.utils as utils


# loading train and metadata
print("loading train data, metadata and embeddings.")
train_df = pd.read_parquet(TRAIN_DATA_PATH)
meta_df = pd.read_parquet(METADATA_PATH)

# loading embeddings
title_emb_df = pd.read_parquet(EMBEDDINGS_TITLE_PATH)
descr_emb_df = pd.read_parquet(EMBEDDINGS_DESCR_PATH)
combi_emb_df = pd.read_parquet(EMBEDDINGS_COMBI_PATH)

# list of unique users in train data and unique items in metadata
users = train_df["user_id"].unique()
items = meta_df["prd_number"].unique()

# loading utils dictionaries
print(f"Loading utils dictionaries from {UTILS_PATH}")
with open(UTILS_PATH, "r") as file:
    utils_dicts = json.load(file)

all_users_show_episodes_dict = utils_dicts["user_show_episodes"]

# adding days since train-val split date as a column in the train_df
print("Computing days since train-val date for each train interaction.")
reference_date = datetime.strptime(SPLIT_DATE_TRAIN_VAL, "%Y-%m-%d")
train_df["days_since"] = (reference_date - train_df["date"]).dt.days

# iterating over levels of metadata
metadata_levels = [(title_emb_df, RECOMMENDATIONS_KEY_CB_TITLE), 
                   (descr_emb_df, RECOMMENDATIONS_KEY_CB_DESCR), 
                   (combi_emb_df, RECOMMENDATIONS_KEY_CB_COMBI)]

for emb_df, rec_key in tqdm(metadata_levels):
    # initializing dictionary to store scores for each user
    scores_dict = defaultdict(dict)

    print(f"Generating recommendations for {rec_key}.")
    # extracting embeddings and storing them in a dictionary
    emb_dict = {}
    for _, row in emb_df.iterrows():
        prd_number = row["episode"]
        embedding = row[1:].values.flatten()
        emb_dict[prd_number] = embedding

    # generate user profiles
    print("Generating user profile from training data.")

    for user in users:
        # initialize user profile (embedding)
        user_profile = np.zeros(EMBEDDING_DIM)

        # filter train_df on user
        user_interactions = train_df[train_df["user_id"] == user]

        # days since listened from train-val split date
        total_days = sum(user_interactions["days_since"])

        # computing weights
        weights = user_interactions["days_since"].to_list()
        weights = [weight / total_days for weight in weights].reverse()

        for i, row in user_interactions.iterrows():
            weight = weights[i]
            prd_number = row["prd_number"]
            embedding = emb_dict[prd_number]
            embedding *= weight
            user_profile += embedding

        # initializing dictionary to store user scores
        user_scores = {}

        # items consumed by user
        user_show_episodes_dict = all_users_show_episodes_dict[user]
        user_items = [item for sublist in user_show_episodes_dict.values() for item in sublist]
        
        for item in items:
            if item in user_items:
                cos_sim = -1
            else:
                embedding = emb_dict[item]
                cos_sim = cosine_similarity(user_profile, embedding)
            user_scores[item] = cos_sim
        
        # normalizing the user scores
        values = np.array(list(user_scores.values()))
        norm = np.linalg.norm(values)
        normalized_user_scores = {key: value / norm for key, value in user_scores.items()}
        
        # adding user_scores to scores_dict
        scores_dict[user] = normalized_user_scores

    # saving scores
    key_scores_dict = {rec_key: scores_dict}
    utils.save_dict_to_json(data_dict=key_scores_dict, file_path=CB_SCORES_PATH)

    # extract recommendations from scores
    recs_dict = utils.extract_recs(scores_dict=scores_dict, n_recs=N_RECOMMENDATIONS)

    # saving recommendations
    recs_dict_key = {rec_key: recs_dict}
    utils.save_dict_to_json(data_dict=recs_dict_key, file_path=RECOMMENDATIONS_PATH)
