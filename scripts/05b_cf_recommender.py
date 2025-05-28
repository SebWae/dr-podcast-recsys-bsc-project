import csv
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
    N_RECOMMENDATIONS,
    N_COMPONENTS,
    REG,
    RANDOM_STATE,
    OPTIMAL_CF_PATH,
    SCORES_PATH_CF,
    SCORES_PATH_CF_INCL_VAL,
    RECOMMENDATIONS_KEY_CF,
    RECOMMENDATIONS_PATH,
)
import utils as utils


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

# generating recommendations from mf model using optimal hyperparameters
prev_ndcg = 0

for epochs in tqdm(range(1, N_EPOCHS+1)):
    print(f"\n Epoch {epochs}:")

    # initializing BiasedMF model
    mf = BiasedMF(features=N_COMPONENTS, 
                  iterations=epochs,
                  reg=REG, 
                  bias=False,
                  rng_spec=RANDOM_STATE)

    # fitting the model
    mf.fit(ratings_df)

    # for evaluation (considers interactions in validation period)
    episode_scores_incl_val = utils.get_cf_scores(model=mf, 
                                                  items=item_list,
                                                  users=user_list,
                                                  item_mapping=show_mapping,
                                                  incl_val_interactions=True)
    
    # for validation (only considers training interactions)
    episode_scores = utils.get_cf_scores(model=mf, 
                                         items=item_list,
                                         users=user_list,
                                         item_mapping=show_mapping,
                                         incl_val_interactions=False)

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

    # saving result
    row = [epochs, ndcg]
    with open(OPTIMAL_CF_PATH, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(row)

    # stopping if ndcg@10 decreases compared to previous epoch 
    if ndcg <= prev_ndcg:
        print("ndcg@10 has decreased or is unchanged.")
        print("Stopping early.")

        # saving scores
        print(f"Saving scores to {SCORES_PATH_CF} and {SCORES_PATH_CF_INCL_VAL}.")
        scores_df = pd.DataFrame(prev_scores)
        scores_df.to_parquet(SCORES_PATH_CF)
        scores_incl_val_df = pd.DataFrame(prev_scores_incl_val)
        scores_incl_val_df.to_parquet(SCORES_PATH_CF_INCL_VAL)

        # saving recommendations
        print(f"Saving recommendations to {RECOMMENDATIONS_PATH}.")
        final_recs = utils.extract_recs(scores_dict=prev_scores_incl_val,
                                        n_recs=N_RECOMMENDATIONS)
        recs_dict_key = {RECOMMENDATIONS_KEY_CF: final_recs}
        utils.save_dict_to_json(data_dict=recs_dict_key, 
                                file_path=RECOMMENDATIONS_PATH)
        break

    prev_ndcg = ndcg
    prev_scores_incl_val = episode_scores_incl_val
    prev_scores = episode_scores
