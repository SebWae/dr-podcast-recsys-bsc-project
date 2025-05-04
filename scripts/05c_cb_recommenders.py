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
    UTILS_PATH,
    SPLIT_DATE_TRAIN_VAL,
    RECOMMENDATIONS_KEY_CB_TITLE,
    RECOMMENDATIONS_KEY_CB_DESCR,
    RECOMMENDATIONS_KEY_CB_COMBI,
    SCORES_PATH_CB_COMBI,
    SCORES_PATH_CB_DESCR,
    SCORES_PATH_CB_TITLE,
    LAMBDA_CB,
    EMBEDDING_DIM,
    WGHT_METHOD,
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
metadata_levels = {"title": {"emb_df": title_emb_df,
                             "rec_key": RECOMMENDATIONS_KEY_CB_TITLE,
                             "scores_path": SCORES_PATH_CB_TITLE},
                    "descr": {"emb_df": descr_emb_df,
                             "rec_key": RECOMMENDATIONS_KEY_CB_DESCR,
                             "scores_path": SCORES_PATH_CB_DESCR},
                    "combi": {"emb_df": title_emb_df,
                             "rec_key": RECOMMENDATIONS_KEY_CB_COMBI,
                             "scores_path": SCORES_PATH_CB_COMBI}
                    }

for level in metadata_levels.values():
    # unpacking values from sub dictionary
    emb_df = level["emb_df"]
    rec_key = level["rec_key"]
    scores_path = level["scores_path"]

    # initializing dictionary to store scores for each user
    scores_dict = defaultdict(dict)

    print(f"Generating recommendations for {rec_key}.")
    # extracting embeddings and storing them in a dictionary
    emb_dict = {}
    for _, row in emb_df.iterrows():
        prd_number = row["episode"]
        if rec_key == RECOMMENDATIONS_KEY_CB_COMBI:
            title_embedding = row[1:].values.flatten()
            descr_embedding = descr_emb_df[descr_emb_df["episode"] == prd_number].iloc[:, 1:].values.flatten()
            embedding = LAMBDA_CB * title_embedding + (1 - LAMBDA_CB) * descr_embedding
        else:
            embedding = row[1:].values.flatten()
        emb_dict[prd_number] = embedding

    # item embeddings as numpy array
    item_embeddings = np.array([emb_dict[item] for item in items], dtype=np.float64)

    # generate user profiles
    print("Generating user profile and recommendations for each user.")
    for user in tqdm(users):
        # initialize user profile (embedding)
        user_interactions = train_df[train_df["user_id"] == user].reset_index()
        user_profile = utils.get_user_profile(emb_size=EMBEDDING_DIM,
                                              user_int=user_interactions,
                                              time_col="days_since",
                                              item_col="prd_number",
                                              emb_dict=emb_dict,
                                              wght_method=WGHT_METHOD)

        # reshaping the user profile to a 2D numpy array
        user_profile_rshpd = user_profile.reshape(1, -1)

        # items consumed by the user
        user_show_episodes_dict = all_users_show_episodes_dict[user]
        user_items = {item for sublist in user_show_episodes_dict.values() for item in sublist}

        # computing all cosine similarities at once for all items
        cos_sim = cosine_similarity(user_profile_rshpd, item_embeddings).flatten()

        # filtering out items already consumed by the user
        user_scores = {}
        for idx, item in enumerate(items):
            if item not in user_items:
                user_scores[item] = cos_sim[idx]

        # normalizing the user scores
        values = np.array(list(user_scores.values()))
        norm = np.linalg.norm(values)
        normalized_user_scores = {key: value / norm for key, value in user_scores.items()}

        # storing the results
        scores_dict[user] = normalized_user_scores

    # saving scores to parquet
    scores_df = pd.DataFrame(scores_dict)
    scores_df.to_parquet(scores_path)

    # extract recommendations from scores
    recs_dict = utils.extract_recs(scores_dict=scores_dict, n_recs=N_RECOMMENDATIONS)

    # saving recommendations
    recs_dict_key = {rec_key: recs_dict}
    utils.save_dict_to_json(data_dict=recs_dict_key, file_path=RECOMMENDATIONS_PATH)
