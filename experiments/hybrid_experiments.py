import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

# adding the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))

from config import (
    VAL_DATA_PATH,
    SCORES_PATH_CF,
    SCORES_PATH_CB_COMBI,
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
lambdas = [int(x) for x in args.lambda_vals(",")]

# loading validation data
print("Loading validation data.")
val_df = pd.read_parquet(VAL_DATA_PATH)

# loading cf and cb scores
cf_scores_df = pd.read_parquet(SCORES_PATH_CF)
cb_scores_df = pd.read_parquet(SCORES_PATH_CB_COMBI)

# converting the scores dataframes to dictionaries
cf_scores = cf_scores_df.to_dict()
cb_scores = cb_scores_df.to_dict()

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

    # saving experiment result
    print(f"Saving experiment results to {EXPERIMENTS_HYBRID_PATH}.")

    row = [_lambda, ndcg]
    with open(EXPERIMENTS_HYBRID_PATH, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(row)
    