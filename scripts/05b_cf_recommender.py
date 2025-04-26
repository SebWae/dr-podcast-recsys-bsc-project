import csv
import os
import sys

from lightfm import LightFM
import numpy as np
import pandas as pd
from tqdm import tqdm

# adding the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))

from config import (
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    METADATA_PATH,
    RANDOM_STATE,
    N_EPOCHS,
    EXPERIMENTS_CF_PATH,
    N_RECOMMENDATIONS,
    RECOMMENDATIONS_KEY_CF,
    RECOMMENDATIONS_PATH,
)
import utils.utils as utils


# loading train, validation and metadata
print("Loading data.")
train_df = pd.read_parquet(TRAIN_DATA_PATH)
val_df = pd.read_parquet(VAL_DATA_PATH)
meta_df = pd.read_parquet(METADATA_PATH)

# left joining the metadata onto the train data
train_w_meta = pd.merge(train_df, meta_df, on="prd_number", how="left")

# grouping by user_id and series_title
print("Constructing user-show interaction matrix.")
grouped_df = train_w_meta.groupby(["user_id", "series_title"]).agg(
    avg_completion_rate =   ("completion_rate", "mean"),
    n_episodes =            ("prd_number", "count")
    ).reset_index()

# computing ratings 
grouped_df["cf_rating"] = grouped_df["avg_completion_rate"] * np.log10(grouped_df["n_episodes"] + 10)

# preparing the interaction matrix
interaction_matrix = utils.prep_interaction_matrix(df=grouped_df, 
                                                   user_col="user_id", 
                                                   item_col="series_title", 
                                                   rating_col="cf_rating")

# list of users and items
user_list = sorted(train_df['user_id'].unique().tolist())
item_list = sorted(train_w_meta['series_title'].unique().tolist())
all_items = meta_df["prd_number"].unique()

# user and item mappings
user_mapping = {user: i for i, user in enumerate(user_list)}
show_mapping = {i: item for i, item in enumerate(item_list)}

# dictionary containing completion rates for each user and consumed item in the validation data
completion_rates_dict = utils.get_ratings_dict(data=val_df, 
                                               user_col="user_id", 
                                               item_col="prd_number", 
                                               ratings_col="completion_rate") 

# values for no_components to test
n_components_values = [10, 20, 30, 40, 50, 60, 70, 80]

# performing hyperparameter experiments for cf recommender
print("Performing hyperparameter experiments.")
print(f"Values for n_components to test: {n_components_values}")

for n_components in n_components_values:
    print(f"n_components: {n_components}")
    # initializing LightFM model
    cf_model = LightFM(loss="logistic", 
                       no_components=n_components, 
                       random_state=RANDOM_STATE)
    prev_ndcg = 0

    for epoch in tqdm(range(N_EPOCHS)):
        print("\n Epoch", epoch + 1)

        # fitting the model
        cf_model.fit_partial(interaction_matrix)

        # getting scores for each item for each user
        episode_scores = utils.get_scores_all_items(model=cf_model, 
                                                    interaction_matrix=interaction_matrix, 
                                                    user_mapping=user_mapping, 
                                                    item_mapping=show_mapping,
                                                    item_list=all_items)
        
        recs_dict = utils.extract_recs(scores_dict=episode_scores,
                                       n_recs=10)

        # computing ndcg@10
        ndcgs = []
        for user_id, rec_items in recs_dict.items():
            gain_dict = completion_rates_dict[user_id]
            optimal_items = sorted(gain_dict, key=lambda x: gain_dict[x], reverse=True)
            dcg = utils.compute_dcg(rec_items, gain_dict)
            dcg_star = utils.compute_dcg(optimal_items, gain_dict)
            ndcg_user = dcg / dcg_star 
            ndcgs.append(ndcg_user)

        ndcg = np.mean(ndcgs)
        print(f"ndcg@10: {ndcg:.4f}")

        # stopping if ndcg@10 decreases compared to previous epoch 
        if ndcg < prev_ndcg:
            print("ndcg@10 has decreased.")
            print("Stopping early.")
            print(f"Saving experiment results to {EXPERIMENTS_CF_PATH}.")

            # writing row to csv
            row = [n_components, ndcg]

            with open(EXPERIMENTS_CF_PATH, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(row)

            break

        prev_ndcg = ndcg

# training the actual cf recommender with the chosen value for n_components and saving the final recommendations

