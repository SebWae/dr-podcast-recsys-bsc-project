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
    TEST_DATA_PATH,
    RECOMMENDATIONS_PATH,
    RECOMMENDERS,
    USER_EVAL_PATH,
    RECOMMENDER_EVAL_PATH,
)
import utils


# importing training data
train_df = pd.read_parquet(TRAIN_DATA_PATH)

# importing test data
test_df = pd.read_parquet(TEST_DATA_PATH)

# Open and load the JSON file
with open(RECOMMENDATIONS_PATH, "r") as file:
    data = json.load(file)

completion_rate_dict = {}

# iterating through the rows of the test_df to build the dictionary
for _, row in test_df.iterrows():
    user = row['user_id']
    prd = row['prd_number']
    completion_rate = row['completion_rate']
    
    # If the user_id is not already in the dictionary, add it with an empty dictionary
    if user not in completion_rate_dict:
        completion_rate_dict[user] = {}
    
    # Add the prd_number and completion_rate to the user's dictionary
    completion_rate_dict[user][prd] = completion_rate

for recommender in tqdm(RECOMMENDERS):
    # retrieving relevant recommendations
    recommendations = data[recommender]

    # dictionaries to store evaluation metrics for each recommender
    recommender_dict = {}
    user_dict = {}

    hit_dict = {user_id: 0 for user_id in recommendations.keys()}
    ndcg_dict = hit_dict.copy()

    for user_id, rec_items in recommendations.items():
        # computing hit-rate (binary) for each user
        rec_items = set(rec_items)
        true_items = set(completion_rate_dict[user_id].keys())
        correct_recs = rec_items.intersection(true_items)
        n_correct_recs = len(correct_recs)
        if n_correct_recs > 0:
            hit_dict[user_id] += 1

        # computing NDCG for each user
        gain_dict = completion_rate_dict[user_id]
        optimal_items = sorted(gain_dict, key=lambda x: gain_dict[x], reverse=True)
        dcg = utils.compute_dcg(rec_items, gain_dict)
        dcg_star = utils.compute_dcg(optimal_items, gain_dict)
        ndcg = dcg / dcg_star 
        ndcg_dict[user_id] = ndcg

    # adding hit_dict to user_dict
    user_dict["hit_rate"] = hit_dict
    
    # adding ndcg_dict to user_dict
    user_dict["ndcg"] = ndcg_dict

    # calculating global hit rate
    hit_rate = np.mean(list(hit_dict.values()))
    recommender_dict["hit_rate"] = hit_rate
    
    # calculating global ndcg
    ndcg = np.mean(list(ndcg_dict.values()))
    recommender_dict["ndcg"] = ndcg

    # final dictionaries
    final_user_dict = {recommender: user_dict}
    final_recommender_dict = {recommender: recommender_dict}

    # saving the results
    utils.save_dict_to_json(data_dict=final_user_dict, 
                            file_path=USER_EVAL_PATH)
    utils.save_dict_to_json(data_dict=final_recommender_dict, 
                            file_path=RECOMMENDER_EVAL_PATH)