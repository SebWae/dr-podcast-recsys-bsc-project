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
    VAL_DATA_PATH,
    SCORES_PATH,
    RECOMMENDATIONS_KEY_CF,
    RECOMMENDATIONS_KEY_CB_COMBI,
    N_RECOMMENDATIONS,
    EXPERIMENTS_HYBRID_PATH,
    LAMBDA,
    RECOMMENDATIONS_PATH,
    RECOMMENDATIONS_KEY_HYBRID,
)
import utils.utils as utils

# loading validation data
print("Loading validation data.")
val_df = pd.read_parquet(VAL_DATA_PATH)

# loading scores from cf and cb recommender
print(f"Loading utils dictionaries from {SCORES_PATH}.")
with open(SCORES_PATH, "r") as file:
    scores_dicts = json.load(file)

cf_scores = scores_dicts[RECOMMENDATIONS_KEY_CF]
cb_scores = scores_dicts[RECOMMENDATIONS_KEY_CB_COMBI]

# all users
cf_users = set(cf_scores.keys())
cb_users = set(cb_scores.keys())
users = cf_users.union(cb_users)

# dictionary containing completion rates for each user and consumed item in the validation data
print("Generating completion_rates_dict from validation data.")
completion_rates_dict = utils.get_ratings_dict(data=val_df, 
                                               user_col="user_id", 
                                               item_col="prd_number", 
                                               ratings_col="completion_rate") 

# hyperparameter tuning for _lambda (weighting hyperparameter)
print("Performing hyperparameter tuning for weighting parameter lambda.")
lambdas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
print(f"Lambda values to test: {lambdas}.")

for _lambda in tqdm(lambdas):
    print(f"\nTesting lambda={_lambda}.")
    hybrid_scores = utils.get_hybrid_scores(scores_dict_1=cf_scores,
                                            scores_dict_2=cb_scores,
                                            users=users,
                                            _lambda=_lambda)
    
    recs_dict = utils.extract_recs(scores_dict=hybrid_scores,
                                   n_recs=N_RECOMMENDATIONS)

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

    # stopping if ndcg@10 decreases compared to previous epoch 
    if ndcg <= prev_ndcg:
        print("ndcg@10 has decreased or is unchanged.")
        print("Stopping early.")
        print(f"Saving experiment results to {EXPERIMENTS_HYBRID_PATH}.")

        # writing row to csv
        row = [_lambda, prev_ndcg]
        with open(EXPERIMENTS_HYBRID_PATH, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(row)

        break

    prev_ndcg = ndcg

# generating recommendations from hybrid recommender with optimal lambda hyperparameter value
print("Generating recommendations from hybrid recommender with optimal lambda hyperparameter value")
hybrid_scores = utils.get_hybrid_scores(scores_dict_1=cf_scores,
                                        scores_dict_2=cb_scores,
                                        users=users,
                                        _lambda=LAMBDA)
    
hybrid_recs = utils.extract_recs(scores_dict=hybrid_scores,
                                 n_recs=N_RECOMMENDATIONS)

# saving recommendations
print(f"Saving recommendations to {RECOMMENDATIONS_PATH}.")
recs_dict_key = {RECOMMENDATIONS_KEY_HYBRID: hybrid_recs}
utils.save_dict_to_json(data_dict=recs_dict_key, 
                        file_path=RECOMMENDATIONS_PATH)
