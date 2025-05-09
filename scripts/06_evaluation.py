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
    TEST_DATA_PATH,
    EMBEDDINGS_DESCR_PATH,
    RECOMMENDATIONS_PATH,
    RECOMMENDERS,
    RECOMMENDATIONS_KEY_CB_TITLE,
    RECOMMENDATIONS_KEY_CB_DESCR,
    RECOMMENDATIONS_KEY_CB_COMBI,
    USER_EVAL_PATH,
    USER_EVAL_CB_PATH,
    RECOMMENDER_EVAL_PATH,
)
import utils.utils as utils


# loading training data, test data and embeddings
print("Loading data and embeddings.")
train_df = pd.read_parquet(TRAIN_DATA_PATH)
test_df = pd.read_parquet(TEST_DATA_PATH)
emb_df = pd.read_parquet(EMBEDDINGS_DESCR_PATH)

# loading recommendations
print("Loading recommendations.")
with open(RECOMMENDATIONS_PATH, "r") as file:
    data = json.load(file)

# constructing the completion rate dictionary
print("Constructing completion rate dictionary.")
completion_rate_dict = utils.get_ratings_dict(data=test_df,
                                              user_col="user_id",
                                              item_col="prd_number",
                                              ratings_col="completion_rate")

# construction dictionary containing embeddings
print("Constructing dictionary containing embeddings.")
emb_dict = {}
for _, row in tqdm(emb_df.iterrows()):
    prd_number = row["episode"]
    embedding = row[1:].values.flatten()
    emb_dict[prd_number] = embedding

# iterating over the recommenders to evaluate them
for recommender in RECOMMENDERS:
    print(f"\nEvaluating the {recommender}.")
    # retrieving relevant recommendations
    recommendations = data[recommender]

    # dictionaries to store evaluation metrics for each recommender
    recommender_dict = defaultdict(dict)
    user_dict = defaultdict(dict)

    # evaluation levels (@2, @6, and @10)
    eval_levels = [2, 6, 10]    

    for level in eval_levels:
        print(f"Evaluation @{level}.")
        # initializing dictionaries to store metrics per user
        hit_dict = {user_id: 0 for user_id in recommendations.keys()}
        ndcg_dict = defaultdict(int)
        diversity_dict = hit_dict.copy()

        # constructing item pair weights dictionary
        print("Constructing weights dictionary for item pairs.")
        weights_dict = utils.get_pair_weights(level)

        for user_id, rec_items in tqdm(recommendations.items()):
            # slicing rec_items according to level
            rec_items = rec_items[:level]
            
            # computing hit-rate (binary) for each user
            true_items = set(completion_rate_dict[user_id].keys())
            correct_recs = true_items.intersection(rec_items)
            n_correct_recs = len(correct_recs)
            if n_correct_recs > 0:
                hit_dict[user_id] += 1

            # computing NDCG for each user
            gain_dict = completion_rate_dict[user_id]
            optimal_items = sorted(gain_dict, key=lambda x: gain_dict[x], reverse=True)[:level]
            dcg = utils.compute_dcg(rec_items, gain_dict)
            dcg_star = utils.compute_dcg(optimal_items, gain_dict)
            ndcg_user = dcg / dcg_star 
            ndcg_dict[user_id] = ndcg_user

            # computing diversity for each user
            diversity_user = utils.compute_diversity(recommendations=rec_items, 
                                                     emb_dict=emb_dict, 
                                                     weights_dict=weights_dict)
            diversity_dict[user_id] = diversity_user

        # adding metric dictionaries to user_dict
        user_dict[level]["hit_rate"] = hit_dict
        user_dict[level]["ndcg"] = ndcg_dict
        user_dict[level]["diversity"] = diversity_dict

        # calculating global hit rate
        hit_rate = np.mean(list(hit_dict.values()))
        recommender_dict[level]["hit_rate"] = hit_rate
        
        # calculating global ndcg
        ndcg = np.mean(list(ndcg_dict.values()))
        recommender_dict[level]["ndcg"] = ndcg

        # calculating global diversity
        diversity = np.mean(list(diversity_dict.values()))
        recommender_dict[level]["diversity"] = diversity

    # final dictionaries
    print("Saving evaluation results.")
    final_user_dict = {recommender: user_dict}
    final_recommender_dict = {recommender: recommender_dict}

    # saving the results
    cb_recommenders = {RECOMMENDATIONS_KEY_CB_TITLE, 
                       RECOMMENDATIONS_KEY_CB_DESCR, 
                       RECOMMENDATIONS_KEY_CB_COMBI,
                       }
    if recommender in cb_recommenders:
        utils.save_dict_to_json(data_dict=final_user_dict, 
                                file_path=USER_EVAL_CB_PATH)
    else:
        utils.save_dict_to_json(data_dict=final_user_dict, 
                                file_path=USER_EVAL_PATH)
    utils.save_dict_to_json(data_dict=final_recommender_dict, 
                            file_path=RECOMMENDER_EVAL_PATH)