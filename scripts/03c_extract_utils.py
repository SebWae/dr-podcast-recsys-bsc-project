from collections import defaultdict
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
show_episodes_final = {"show_episodes": show_episodes}
utils.save_dict_to_json(data_dict=show_episodes_final,
                        file_path=UTILS_PATH)

# left joining the metadata onto the train data
train_w_meta = pd.merge(train_df, meta_df, on="prd_number", how="left")

# sorting the train_w_meta dataframe by pub_date 
train_w_meta_sorted = train_w_meta.sort_values(by="pub_date", ascending=True)

# initializing the user_show_episodes dictionary
user_show_episodes = defaultdict(lambda: defaultdict(list))

# iterating through the train_w_meta_sorted dataframe and generating the dictionary
for _, row in train_w_meta_sorted.iterrows():
    user_id = row["user_id"]
    series_title = row["series_title"]
    prd_number = row["prd_number"]
    user_show_episodes[user_id][series_title].append(prd_number)

# save most_recent_episode_dict to json
user_show_episodes_final = {"user_show_episodes": user_show_episodes}
utils.save_dict_to_json(data_dict=user_show_episodes_final,
                        file_path=UTILS_PATH)
