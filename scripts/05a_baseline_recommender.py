import os
import sys

import numpy as np
import pandas as pd

# adding the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))

from config import (
    TRANSFORMED_DATA_PATH,
    TEST_DATA_PATH,
    USER_EVAL_PATH,
    RECOMMENDER_EVAL_PATH,
)
import utils

# the baseline recommender is being implemented differently from the other recommenders
# recommends the 10 most listened episodes for all users
# since the recommendations are identical for every user, there is no need to store them
# thus, the baseline recommender will be evaluated directly in this script

# loading the transformed data
transformed_df = pd.read_parquet(TRANSFORMED_DATA_PATH)

# loading the test data
test_df = pd.read_parquet(TEST_DATA_PATH)

# finding the most popular episodes
episode_counts = transformed_df.groupby("prd_number").size().reset_index(name="count")
top_10_episodes = episode_counts.nlargest(10, "count")["prd_number"].tolist()

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


# dictionaries to store evaluation metrics for each recommender
recommender_dict = {}
user_dict = {}

hit_dict = {user_id: 0 for user_id in completion_rate_dict.keys()}
ndcg_dict = hit_dict.copy()

for user_id, gain_dict in completion_rate_dict.items():
    # computing hit-rate (binary) for each user
    true_items = set(gain_dict.keys())
    correct_recs = true_items.intersection(top_10_episodes)
    n_correct_recs = len(correct_recs)
    if n_correct_recs > 0:
        hit_dict[user_id] += 1

    # computing NDCG for each user
    optimal_items = sorted(gain_dict, key=lambda x: gain_dict[x], reverse=True)
    dcg = utils.compute_dcg(top_10_episodes, gain_dict)
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
final_user_dict = {"pop_baseline": user_dict}
final_recommender_dict = {"pop_baseline": recommender_dict}

# saving the results
utils.save_dict_to_json(data_dict=final_user_dict, 
                        file_path=USER_EVAL_PATH)
utils.save_dict_to_json(data_dict=final_recommender_dict, 
                        file_path=RECOMMENDER_EVAL_PATH)
