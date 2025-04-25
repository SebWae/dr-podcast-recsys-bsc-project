import os
import sys

import pandas as pd

# adding the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))

from config import (
    TRANSFORMED_DATA_PATH,
    TRAIN_DATA_PATH,
    METADATA_PATH,
    UTILS_PATH,
)
import utils.utils as utils


# loading transformed, train and metadata
transformed_df = pd.read_parquet(TRANSFORMED_DATA_PATH)
train_df = pd.read_parquet(TRAIN_DATA_PATH)
meta_df = pd.read_parquet(METADATA_PATH)

# creating dictionary containing episodes in publication order per show
shows = set(meta_df["series_title"])
show_episodes = {show: meta_df[meta_df["series_title"] == show]
                 .sort_values(by="pub_date", ascending=True)["prd_number"]
                 .tolist() for show in shows
                 }

# saving show_episodes_dict to json
show_episodes_final = {"show_episodes_dict": show_episodes}
utils.save_dict_to_json(data_dict=show_episodes_final,
                        file_path=UTILS_PATH)

# left joining the metadata onto the train data
train_w_meta = pd.merge(train_df, meta_df, on="prd_number", how="left")

# grouping by user_id and series_title, and getting the prd_number with the most recent pub_date
most_recent_prd = (train_w_meta.sort_values(by="pub_date", ascending=False)
                   .groupby(["user_id", "series_title"])
                   .first()["prd_number"]
                   .reset_index()
                   )

# creating a dictionary to retrieve the most recent episode per show per user
most_recent_episodes = {}
for _, row in most_recent_prd.iterrows():
    user_id = row["user_id"]
    series_title = row["series_title"]
    prd_number = row["prd_number"]
    
    if user_id not in most_recent_episodes:
        most_recent_episodes[user_id] = {}
    most_recent_episodes[user_id][series_title] = prd_number

# save most_recent_episode_dict to json
most_recent_episode_final = {"most_recent_episodes": most_recent_episodes}
utils.save_dict_to_json(data_dict=most_recent_episode_final,
                        file_path=UTILS_PATH)