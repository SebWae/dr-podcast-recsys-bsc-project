from collections import defaultdict
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
    METADATA_PATH,
    EMBEDDINGS_TITLE_PATH,
    EMBEDDINGS_DESCR_PATH,
    EMBEDDINGS_COMBI_PATH,
    UTILS_INTERACTIONS_PATH,
    RECOMMENDATIONS_KEY_CB_TITLE,
    RECOMMENDATIONS_KEY_CB_DESCR,
    RECOMMENDATIONS_KEY_CB_COMBI,
    EMBEDDING_DIM,
    WGHT_METHOD,
    N_RECOMMENDATIONS,
    RECOMMENDATIONS_PATH,
)
import utils as utils


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
print(f"Loading utils dictionaries from {UTILS_INTERACTIONS_PATH}")
with open(UTILS_INTERACTIONS_PATH, "r") as file:
    utils_dicts = json.load(file)

all_users_show_episodes_dict = utils_dicts["user_show_episodes_val"]

# iterating over levels of metadata
metadata_levels = {
    "title": {"emb_df": title_emb_df,
              "rec_key": RECOMMENDATIONS_KEY_CB_TITLE},
    "descr": {"emb_df": descr_emb_df,
              "rec_key": RECOMMENDATIONS_KEY_CB_DESCR},
    "combi": {"emb_df": combi_emb_df,
              "rec_key": RECOMMENDATIONS_KEY_CB_COMBI}
    }

for level in metadata_levels.values():
    # unpacking values from sub dictionary
    emb_df = level["emb_df"]
    rec_key = level["rec_key"]

    # initializing dictionary to store scores for each user
    scores_dict = defaultdict(dict)

    print(f"Generating recommendations for {rec_key}.")
    # extracting embeddings and storing them in a dictionary
    emb_dict = {}
    for _, row in emb_df.iterrows():
        prd_number = row["episode"]
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
                                              wght_scheme=WGHT_METHOD)

        # scores for user for each item not consumed by the user
        normalized_user_scores = utils.get_cb_scores(user=user,
                                                     show_episodes=all_users_show_episodes_dict,
                                                     user_profile=user_profile,
                                                     item_embeddings=item_embeddings,
                                                     items=items)

        # storing the results
        scores_dict[user] = normalized_user_scores

    # extract recommendations from scores
    print("Extracting recommendations from scores_dict.")
    recs_dict = utils.extract_recs(scores_dict=scores_dict, n_recs=N_RECOMMENDATIONS)

    # saving recommendations
    print(f"Saving recommendations to {RECOMMENDATIONS_PATH}.")
    recs_dict_key = {rec_key: recs_dict}
    utils.save_dict_to_json(data_dict=recs_dict_key, file_path=RECOMMENDATIONS_PATH)
