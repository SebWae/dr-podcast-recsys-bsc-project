from collections import defaultdict
import os
import sys

import pandas as pd
from tqdm import tqdm

# adding the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))

from config import (
    TRANSFORMED_DATA_PATH,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    METADATA_PATH,
    SPLIT_DATE_TRAIN_VAL,
    UTILS_PATH,
)
import utils.utils as utils


# loading transformed, train, validation and metadata
print("Loading data.")
transformed_df = pd.read_parquet(TRANSFORMED_DATA_PATH)
train_df = pd.read_parquet(TRAIN_DATA_PATH)
val_df = pd.read_parquet(VAL_DATA_PATH)
meta_df = pd.read_parquet(METADATA_PATH)

# creating dictionary containing episodes in publication order per show
print("Generating dictionary containing episodes in publication order per show.")
shows = set(meta_df["series_title"])
show_episodes = {show: meta_df[meta_df["series_title"] == show]
                 .sort_values(by="pub_date", ascending=True)["prd_number"]
                 .tolist() for show in shows
                 }

# saving show_episodes_dict to json
print("Saving show_episodes dictionary.")
show_episodes_final = {"show_episodes": show_episodes}
utils.save_dict_to_json(data_dict=show_episodes_final,
                        file_path=UTILS_PATH)

# dropping the days_since column from train_df and concatenating train and validation data
print("Concatenating the train and validation data.")
train_df.drop(columns="days_since", inplace=True)
train_val_df = pd.concat([train_df, val_df], ignore_index=True)

# left joining the metadata onto the combined df
train_val_w_meta = pd.merge(train_val_df, meta_df, on="prd_number", how="left")

# sorting the train_val_w_meta dataframe by pub_date 
train_val_w_meta_sorted = train_val_w_meta.sort_values(by="pub_date", ascending=True)

# initializing the user_show_episodes dictionary
print("Generating user_show_episodes dictionaries.")
user_show_episodes = defaultdict(lambda: defaultdict(list))
user_show_episodes_val = defaultdict(lambda: defaultdict(list))

# iterating through the train_w_meta_sorted dataframe and generating the dictionary
for _, row in tqdm(train_val_w_meta_sorted.iterrows(), total=len(train_val_w_meta_sorted)):
    user_id = row["user_id"]
    series_title = row["series_title"]
    prd_number = row["prd_number"]
    date = row["date"]
    
    # only appending to user_show_episodes if interaction was in train period
    if date < pd.to_datetime(SPLIT_DATE_TRAIN_VAL):
        user_show_episodes[user_id][series_title].append(prd_number)

    # always append to user_show_episodes_val
    user_show_episodes_val[user_id][series_title].append(prd_number)

# save most_recent_episode_dicts to json
print("Saving user_show_episodes dictionaries.")
user_show_episodes_final = {"user_show_episodes": user_show_episodes}
user_show_episodes_val_final = {"user_show_episodes_val": user_show_episodes_val}

utils.save_dict_to_json(data_dict=user_show_episodes_final,
                        file_path=UTILS_PATH)
utils.save_dict_to_json(data_dict=user_show_episodes_val_final,
                        file_path=UTILS_PATH)
