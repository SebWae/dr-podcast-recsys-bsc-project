import csv
from itertools import product
import os
import sys

from lenskit.algorithms.als import BiasedMF
import numpy as np
import pandas as pd
from tqdm import tqdm

# adding the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))

from config import (
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    METADATA_PATH,
    N_EPOCHS,
    RANDOM_STATE,
    EXPERIMENTS_CF_PATH,
    N_RECOMMENDATIONS,
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
print("Constructing user-show dataframe.")
grouped_df = train_w_meta.groupby(["user_id", "series_title"]).agg(
    avg_completion_rate =   ("completion_rate", "mean"),
    n_episodes =            ("prd_number", "count")
    ).reset_index()

# computing ratings 
print("Computing user-show ratings.")
grouped_df["cf_rating"] = grouped_df["avg_completion_rate"] * np.log10(grouped_df["n_episodes"] + 10)

# renaming columns
ratings_df = grouped_df.rename(columns={"user_id": "user",
                                        "series_title": "item",
                                        "cf_rating": "rating"})

# list of users and items
user_list = sorted(train_df['user_id'].unique())
item_list = sorted(train_w_meta['series_title'].unique().tolist())

# mapping indices to shows
show_mapping = {i: item for i, item in enumerate(item_list)}

# dictionary containing completion rates for each user and consumed item in the validation data
print("Generating completion_rates_dict from validation data.")
completion_rates_dict = utils.get_ratings_dict(data=val_df, 
                                               user_col="user_id", 
                                               item_col="prd_number", 
                                               ratings_col="completion_rate") 

# initial hyperparameter values to test
# n_components_values = [50, 100, 150]
# damping_values = [1, 5, 10]
# reg_values = [0.05, 0.1, 0.15]

# # performing initial hyperparameter experiments for cf recommender
# print("Performing initial hyperparameter experiments.")
# print(f"Testing values: n_components={n_components_values}, damping={damping_values}, reg={reg_values}")

# for n_components, damping, reg in tqdm(list(product(n_components_values, damping_values, reg_values))):
#     print(f"\nTesting combination: features={n_components}, damping={damping}, reg={reg}")
#     prev_ndcg = 0
#     for epochs in range(1, N_EPOCHS+1):
#         print(f"\n Epoch {epochs}:")

#         # initializing BiasedMF model
#         mf = BiasedMF(features=n_components, 
#                       damping=damping, 
#                       reg=reg, 
#                       iterations=epochs, 
#                       rng_spec=RANDOM_STATE)

#         # fitting the model
#         mf.fit(ratings_df)

#         # getting scores for each item for each user
#         episode_scores = utils.get_cf_scores(model=mf, 
#                                              items=item_list,
#                                              users=user_list,
#                                              item_mapping=show_mapping)

#         recs_dict = utils.extract_recs(scores_dict=episode_scores,
#                                        n_recs=N_RECOMMENDATIONS)

#         # computing ndcg@10
#         ndcgs = []
#         for user_id, rec_items in recs_dict.items():
#             gain_dict = completion_rates_dict[user_id]
#             optimal_items = sorted(gain_dict, key=lambda x: gain_dict[x], reverse=True)[:N_RECOMMENDATIONS]
#             dcg = utils.compute_dcg(rec_items, gain_dict)
#             dcg_star = utils.compute_dcg(optimal_items, gain_dict)
#             ndcg_user = dcg / dcg_star 
#             ndcgs.append(ndcg_user)

#         ndcg = np.mean(ndcgs)
#         print(f"ndcg@10: {ndcg:.10f}")

#         # stopping if ndcg@10 decreases compared to previous epoch 
#         if ndcg <= prev_ndcg:
#             print("ndcg@10 has decreased or is unchanged.")
#             print("Stopping early.")
#             print(f"Saving experiment results to {EXPERIMENTS_CF_PATH}.")

#             # writing row to csv
#             row = [n_components, damping, reg, prev_ndcg]
#             with open(EXPERIMENTS_CF_PATH, mode="a", newline="") as file:
#                 writer = csv.writer(file)
#                 writer.writerow(row)

#             break

#         prev_ndcg = ndcg


# # enhanced hyperparameter values to test
# n_components_values = [200, 250, 300]
# damping_values = [20, 30, 40]
# reg_values = [0.0025, 0.005, 0.01, 0.025]

# # performing enhanced hyperparameter experiments for cf recommender
# print("Performing enhanced hyperparameter experiments.")
# print(f"Testing values: n_components={n_components_values}, damping={damping_values}, reg={reg_values}")

# for n_components, damping, reg in tqdm(list(product(n_components_values, damping_values, reg_values))):
#     print(f"\nTesting combination: features={n_components}, damping={damping}, reg={reg}")
#     prev_ndcg = 0
#     for epochs in range(1, N_EPOCHS+1):
#         print(f"\n Epoch {epochs}:")

#         # initializing BiasedMF model
#         mf = BiasedMF(features=n_components, 
#                       damping=damping, 
#                       reg=reg, 
#                       iterations=epochs, 
#                       rng_spec=RANDOM_STATE)

#         # fitting the model
#         mf.fit(ratings_df)

#         # getting scores for each item for each user
#         episode_scores = utils.get_cf_scores(model=mf, 
#                                              items=item_list,
#                                              users=user_list,
#                                              item_mapping=show_mapping)

#         recs_dict = utils.extract_recs(scores_dict=episode_scores,
#                                        n_recs=N_RECOMMENDATIONS)

#         # computing ndcg@10
#         ndcgs = []
#         for user_id, rec_items in recs_dict.items():
#             gain_dict = completion_rates_dict[user_id]
#             optimal_items = sorted(gain_dict, key=lambda x: gain_dict[x], reverse=True)[:N_RECOMMENDATIONS]
#             dcg = utils.compute_dcg(rec_items, gain_dict)
#             dcg_star = utils.compute_dcg(optimal_items, gain_dict)
#             ndcg_user = dcg / dcg_star 
#             ndcgs.append(ndcg_user)

#         ndcg = np.mean(ndcgs)
#         print(f"ndcg@10: {ndcg:.10f}")

#         # stopping if ndcg@10 decreases compared to previous epoch 
#         if ndcg <= prev_ndcg:
#             print("ndcg@10 has decreased or is unchanged.")
#             print("Stopping early.")
#             print(f"Saving experiment results to {EXPERIMENTS_CF_PATH}.")

#             # writing row to csv
#             row = [n_components, damping, reg, prev_ndcg]
#             with open(EXPERIMENTS_CF_PATH, mode="a", newline="") as file:
#                 writer = csv.writer(file)
#                 writer.writerow(row)

#             break

#         prev_ndcg = ndcg


# # enhanced hyperparameter values (v2) to test
# n_components_values = [400, 500, 600]
# damping_values = [50, 60, 70]
# reg_values = [0.001, 0.002, 0.003]

# # performing enhanced hyperparameter experiments (v2) for cf recommender
# print("Performing enhanced hyperparameter experiments (v2).")
# print(f"Testing values: n_components={n_components_values}, damping={damping_values}, reg={reg_values}")

# for n_components, damping, reg in tqdm(list(product(n_components_values, damping_values, reg_values))):
#     print(f"\nTesting combination: features={n_components}, damping={damping}, reg={reg}")
#     prev_ndcg = 0
#     for epochs in range(1, N_EPOCHS+1):
#         print(f"\n Epoch {epochs}:")

#         # initializing BiasedMF model
#         mf = BiasedMF(features=n_components, 
#                       damping=damping, 
#                       reg=reg, 
#                       iterations=epochs, 
#                       rng_spec=RANDOM_STATE)

#         # fitting the model
#         mf.fit(ratings_df)

#         # getting scores for each item for each user
#         episode_scores = utils.get_cf_scores(model=mf, 
#                                              items=item_list,
#                                              users=user_list,
#                                              item_mapping=show_mapping)

#         recs_dict = utils.extract_recs(scores_dict=episode_scores,
#                                        n_recs=N_RECOMMENDATIONS)

#         # computing ndcg@10
#         ndcgs = []
#         for user_id, rec_items in recs_dict.items():
#             gain_dict = completion_rates_dict[user_id]
#             optimal_items = sorted(gain_dict, key=lambda x: gain_dict[x], reverse=True)[:N_RECOMMENDATIONS]
#             dcg = utils.compute_dcg(rec_items, gain_dict)
#             dcg_star = utils.compute_dcg(optimal_items, gain_dict)
#             ndcg_user = dcg / dcg_star 
#             ndcgs.append(ndcg_user)

#         ndcg = np.mean(ndcgs)
#         print(f"ndcg@10: {ndcg:.10f}")

#         # stopping if ndcg@10 decreases compared to previous epoch 
#         if ndcg <= prev_ndcg:
#             print("ndcg@10 has decreased or is unchanged.")
#             print("Stopping early.")
#             print(f"Saving experiment results to {EXPERIMENTS_CF_PATH}.")

#             # writing row to csv
#             row = [n_components, damping, reg, prev_ndcg]
#             with open(EXPERIMENTS_CF_PATH, mode="a", newline="") as file:
#                 writer = csv.writer(file)
#                 writer.writerow(row)

#             break

#         prev_ndcg = ndcg


# enhanced hyperparameter values (v3) to test
n_components_values = [700, 800, 900]
damping_values = [80, 100, 120]
reg_values = [0.0001, 0.00025, 0.0005]

# performing enhanced hyperparameter experiments (v3) for cf recommender
print("Performing enhanced hyperparameter experiments (v3).")
print(f"Testing values: n_components={n_components_values}, damping={damping_values}, reg={reg_values}")

for n_components, damping, reg in tqdm(list(product(n_components_values, damping_values, reg_values))):
    print(f"\nTesting combination: features={n_components}, damping={damping}, reg={reg}")
    prev_ndcg = 0
    for epochs in range(1, N_EPOCHS+1):
        print(f"\n Epoch {epochs}:")

        # initializing BiasedMF model
        mf = BiasedMF(features=n_components, 
                      damping=damping, 
                      reg=reg, 
                      iterations=epochs, 
                      rng_spec=RANDOM_STATE)

        # fitting the model
        mf.fit(ratings_df)

        # getting scores for each item for each user
        episode_scores = utils.get_cf_scores(model=mf, 
                                             items=item_list,
                                             users=user_list,
                                             item_mapping=show_mapping)

        recs_dict = utils.extract_recs(scores_dict=episode_scores,
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
            print(f"Saving experiment results to {EXPERIMENTS_CF_PATH}.")

            # writing row to csv
            row = [n_components, damping, reg, prev_ndcg]
            with open(EXPERIMENTS_CF_PATH, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(row)

            break

        prev_ndcg = ndcg
