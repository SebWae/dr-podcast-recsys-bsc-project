from collections import defaultdict
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
    METADATA_PATH,
    EMBEDDINGS_COMBI_PATH,
    SPLIT_DATE_VAL_TEST,
    USER_EVAL_PATH,
    RECOMMENDER_EVAL_PATH,
)
import utils

# the baseline recommender is being implemented differently from the other recommenders
# identifies the 10 most listened shows in the training data 
# recommends the first episode in the test set of the top 10 shows
# if a show has not published any episodes in the test period, the newest episode overall will be recommended 
# since the recommendations are identical for every user, there is no need to store them
# thus, the baseline recommender will be evaluated directly in this script

# loading train, test and metadata
train_df = pd.read_parquet(TRAIN_DATA_PATH)
test_df = pd.read_parquet(TEST_DATA_PATH)
meta_df = pd.read_parquet(METADATA_PATH)

# loading embeddings
emb_df = pd.read_parquet(EMBEDDINGS_COMBI_PATH)

# left joining the metadata onto the train data
train_w_meta = pd.merge(train_df, meta_df, on="prd_number", how="left")

# identifying the top 10 shows
show_counts = train_w_meta.groupby("series_title").size().reset_index(name="count")
top_10_shows = show_counts.nlargest(10, "count")["series_title"].tolist()

# finding most recent episode for each top 10 show after val-test split date
recommendations = []
for show in top_10_shows:
    show_filtered = meta_df[(meta_df["series_title"] == show) & (meta_df["pub_date"] >= SPLIT_DATE_VAL_TEST)]
    shows_after_split_date = len(show_filtered)

    if shows_after_split_date == 0:
        show_filtered = meta_df[meta_df["series_title"] == show]

    show_filtered_sorted = show_filtered.sort_values(by="pub_date")
    first_prd_number = show_filtered_sorted.iloc[0]["prd_number"]
    recommendations.append(first_prd_number)

# iterating through the rows of the test_df to build dictionary of completion rates
completion_rate_dict = {}

for _, row in test_df.iterrows():
    user = row['user_id']
    prd = row['prd_number']
    completion_rate = row['completion_rate']
    
    # adding new users with an empty dictionary
    if user not in completion_rate_dict:
        completion_rate_dict[user] = {}
    
    completion_rate_dict[user][prd] = completion_rate

# dictionaries to store evaluation metrics for each recommender
recommender_dict = defaultdict(dict)
user_dict = defaultdict(dict)

# evaluation levels (@2, @6, and @10)
eval_levels = [2, 6, 10]

for level in tqdm(eval_levels):
    # initializing dictionaries to store metrics per user
    hit_dict = {user_id: 0 for user_id in completion_rate_dict.keys()}
    ndcg_dict = hit_dict.copy()
    diversity_dict = hit_dict.copy()

    # number of recommendations according to level
    recs = recommendations[:level]

    for user_id, gain_dict in completion_rate_dict.items():
        # computing hit-rate (binary) for each user
        true_items = set(gain_dict.keys())
        correct_recs = true_items.intersection(recs)
        n_correct_recs = len(correct_recs)
        if n_correct_recs > 0:
            hit_dict[user_id] += 1

        # computing NDCG for each user
        optimal_items = sorted(gain_dict, key=lambda x: gain_dict[x], reverse=True)
        dcg = utils.compute_dcg(recs, gain_dict)
        dcg_star = utils.compute_dcg(optimal_items, gain_dict)
        ndcg = dcg / dcg_star 
        ndcg_dict[user_id] = ndcg

    # adding hit_dict to user_dict
    user_dict[level]["hit_rate"] = hit_dict

    # adding ndcg_dict to user_dict
    user_dict[level]["ndcg"] = ndcg_dict

    # computing global hit rate
    hit_rate = np.mean(list(hit_dict.values()))
    recommender_dict[level]["hit_rate"] = hit_rate

    # computing global ndcg
    ndcg = np.mean(list(ndcg_dict.values()))
    recommender_dict[level]["ndcg"] = ndcg

    # computing global diversity (same for all users)
    diversity = utils.compute_diversity(recommendations=recs, 
                                        item_features=emb_df, 
                                        item_id_name="episode")
    recommender_dict[level]["diversity"] = diversity

# final dictionaries
final_user_dict = {"pop_baseline": user_dict}
final_recommender_dict = {"pop_baseline": recommender_dict}

# saving the results
utils.save_dict_to_json(data_dict=final_user_dict, 
                        file_path=USER_EVAL_PATH)
utils.save_dict_to_json(data_dict=final_recommender_dict, 
                        file_path=RECOMMENDER_EVAL_PATH)
