import json
import os
import sys

import pandas as pd

# adding the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))

import utils
from config import (
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    RECOMMENDATIONS_PATH,
)


# importing training data
train_df = pd.read_parquet(TRAIN_DATA_PATH)

# importing test data
test_df = pd.read_parquet(TEST_DATA_PATH)

# grouping by user_id and counting the number of consumed items (not for completion_rate)
train_df_grouped = train_df.groupby("user_id")["prd_number"].count().reset_index()

# Open and load the JSON file
with open(RECOMMENDATIONS_PATH, "r") as file:
    data = json.load(file)

# dictionaries to store evaluation metrics
recommender_dict_global = {}
user_dict_global = {}

recommenders = ["cf_recommendations"]

for recommender in recommenders:
    # retrieving relevant recommendations
    recommendations = data["cf_recommendations"]

    # dictionaries to store evaluation metrics for each recommender
    recommender_dict = {}
    user_dict = {}

    # dictionary containing test items for each user
    test_items = {}
    for id, row in test_df.iterrows():
        if row["user_id"] not in test_items:
            test_items[row["user_id"]] = set()
        test_items[row["user_id"]].add(row["prd_number"])

    hit_dict = {user_id: 0 for user_id in recommendations.keys()}

    for user_id, rec_items in recommendations.items():
        rec_items = set(rec_items)
        true_items = test_items[user_id]
        correct_recs = rec_items.intersection(true_items)
        n_correct_recs = len(correct_recs)
        if n_correct_recs > 0:
            hit_dict[user_id] += 1
    
    # adding hit_dict to user_dict
    user_dict["hit_rate"] = hit_dict