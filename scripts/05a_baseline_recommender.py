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
    EMBEDDINGS_DESCR_PATH,
    N_RECOMMENDATIONS,
    SPLIT_DATE_VAL_TEST,
    USER_EVAL_PATH_2,
    USER_EVAL_PATH_6,
    USER_EVAL_PATH_10,
    RECOMMENDATIONS_KEY_BASELINE,
    RECOMMENDER_EVAL_PATH,
)
import utils.utils as utils

# the baseline recommender is being implemented differently from the other recommenders
# identifies the 10 most listened shows in the training data 
# recommends the first episode in the test set of the top 10 shows
# if a show has not published any episodes in the test period, the newest episode overall will be recommended 
# since the recommendations are identical for every user, there is no need to store them
# thus, the baseline recommender will be evaluated directly in this script

# loading train, test and metadata
print("Loading data and embeddings.")
train_df = pd.read_parquet(TRAIN_DATA_PATH)
test_df = pd.read_parquet(TEST_DATA_PATH)
meta_df = pd.read_parquet(METADATA_PATH)

# loading embeddings
emb_df = pd.read_parquet(EMBEDDINGS_DESCR_PATH)

# left joining the metadata onto the train data
train_w_meta = pd.merge(train_df, meta_df, on="prd_number", how="left")

# identifying the top shows
print(f"Identifying the top {N_RECOMMENDATIONS} shows.")
show_counts = train_w_meta.groupby("series_title")["user_id"].nunique()
top_shows = show_counts.sort_values(ascending=False)[:N_RECOMMENDATIONS].index.tolist()

# finding most recent episode for each top show after val-test split date
print("Generating recommendations")
recommendations = []
for show in tqdm(top_shows):
    show_filtered = meta_df[(meta_df["series_title"] == show) & (meta_df["pub_date"] >= SPLIT_DATE_VAL_TEST)]
    shows_after_split_date = len(show_filtered)

    if shows_after_split_date == 0:
        show_filtered = meta_df[meta_df["series_title"] == show]

    show_filtered_sorted = show_filtered.sort_values(by="pub_date")
    first_prd_number = show_filtered_sorted.iloc[0]["prd_number"]
    recommendations.append(first_prd_number)

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

# paths for user evaluation results
user_eval_paths = {2: USER_EVAL_PATH_2,
                   6: USER_EVAL_PATH_6,
                   10: USER_EVAL_PATH_10}

# dictionaries to store evaluation metrics for each recommender
recommender_dict = defaultdict(dict)

# evaluation levels (@2, @6, and @10)
eval_levels = [2, 6, 10]

print("Evaluating baseline recommender.")
for level in tqdm(eval_levels):
    # initializing dictionaries to store metrics per user
    user_dict = defaultdict(dict)
    hit_dict = {user_id: 0 for user_id in completion_rate_dict.keys()}
    ndcg_dict = hit_dict.copy()
    diversity_dict = hit_dict.copy()

    # number of recommendations according to level
    recs = recommendations[:level]

    # constructing item pair weights dictionary
    print("Constructing weights dictionary for item pairs.")
    weights_dict = utils.get_pair_weights(level)

    for user_id, gain_dict in completion_rate_dict.items():
        # computing hit-rate (binary) for each user
        true_items = set(gain_dict.keys())
        correct_recs = true_items.intersection(recs)
        n_correct_recs = len(correct_recs)
        if n_correct_recs > 0:
            hit_dict[user_id] += 1

        # computing NDCG for each user
        optimal_items = sorted(gain_dict, key=lambda x: gain_dict[x], reverse=True)[:level]
        dcg = utils.compute_dcg(recs, gain_dict)
        dcg_star = utils.compute_dcg(optimal_items, gain_dict)
        ndcg = dcg / dcg_star 
        ndcg_dict[user_id] = ndcg

    # adding hit_dict to user_dict
    user_dict["hit_rate"] = hit_dict

    # adding ndcg_dict to user_dict
    user_dict["ndcg"] = ndcg_dict

    # saving user evaluation results
    final_user_dict = {RECOMMENDATIONS_KEY_BASELINE: user_dict}
    user_eval_path = user_eval_paths[level] 
    utils.save_dict_to_json(data_dict=final_user_dict, 
                            file_path=user_eval_path)

    # computing global hit rate
    hit_rate = np.mean(list(hit_dict.values()))
    recommender_dict[level]["hit_rate"] = hit_rate

    # computing global ndcg
    ndcg = np.mean(list(ndcg_dict.values()))
    recommender_dict[level]["ndcg"] = ndcg

    # computing global diversity (same for all users)
    diversity = utils.compute_diversity(recommendations=recs, 
                                        emb_dict=emb_dict, 
                                        weights_dict=weights_dict)
    recommender_dict[level]["diversity"] = diversity

# final dictionaries
print("Saving results.")
final_recommender_dict = {RECOMMENDATIONS_KEY_BASELINE: recommender_dict}

# saving the results
utils.save_dict_to_json(data_dict=final_recommender_dict, 
                        file_path=RECOMMENDER_EVAL_PATH)
