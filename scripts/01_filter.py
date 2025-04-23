import os
import pandas as pd
import sys

# adding the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))

from config import (
    RAW_DATA_PATH,
    VAR_RENAME_DICT,
    START_DATE, 
    END_DATE, 
    MIN_CONTENT_TIME_SPENT,
    SPLIT_DATE_TRAIN_VAL,
    SPLIT_DATE_VAL_TEST,
    MIN_PLAYS_PER_USER,
    FILTERED_DATA_PATH,
    MIN_USERS_PER_SHOW,
)


# loading the raw data 
print("Loading raw data")
df = pd.read_parquet(RAW_DATA_PATH)

# renaming the columns 
df.rename(columns=VAR_RENAME_DICT, inplace=True)

# changing the datatype for the content time spent column
df["content_time_spent"] = df["content_time_spent"].astype(int)

# user_id and prd_number must exist and content_type must be vod
print("Filtering on existing user_id, prd_number and content_type (filtering task 1/7)")
filtered_df = df[(df["user_id"].notna()) & 
                 (df["prd_number"].notna()) & 
                 (df["content_type"] == "vod")
                 ]

# dropping the content_type attribute
filtered_df.drop(columns=["content_type"], inplace=True)

# only including relevant mobile device types
print("Filtering on device_type (filtering task 2/7)")
main_devices = {"Mobile Phone", "Other", "Tablet"}
filtered_df = filtered_df[filtered_df["device_type"].isin(main_devices)]

# excluding some combinations of platform and mobile device type
print("Filtering on combination of platform and device_type (filtering task 3/7)")
exclude_combinations = {("mobile web", "Other"), ("web", "Mobile Phone"), ("web", "Tablet")}
filtered_df = filtered_df.loc[~df[["platform", "device_type"]]
                              .apply(tuple, axis=1)
                              .isin(exclude_combinations)]

# formatting the date_time column
filtered_df["date_time"] = pd.to_datetime(filtered_df["date_time"], format="%d:%m:%Y|%H:%M")

# filter out interactions outside of desired date range
print("Filtering on start and end date (filtering task 4/7)")
filtered_df = filtered_df[(filtered_df["date_time"] >= START_DATE) & 
                          (filtered_df["date_time"] <= END_DATE)]

# grouping rows by user_id and prd_number
print("Grouping by user_id and prd_number and filtering on content_time_spent (filtering task 5/7)")
cts_grp_df = filtered_df.groupby(["user_id", "prd_number"]).agg(
    date_time =             ("date_time",           lambda x: 
                             x.loc[df.loc[x.index, "content_time_spent"].idxmax()]),
    series_title =          ("series_title",        "first"),
    unique_title =          ("unique_title",        "first"),
    platform =              ("platform",            "first"),
    device_type =           ("device_type",         "first"),
    pub_date =              ("pub_date",            "first"),
    episode_duration =      ("episode_duration",    "first"),
    genre =                 ("genre",               "first"),
    branding_channel =      ("branding_channel",    "first"),
    mother_channel =        ("mother_channel",      "first"),
    category =              ("category",            "first"),
    content_time_spent =    ("content_time_spent",  "sum")
).reset_index()

# filtering on content time spent
filtered_df = cts_grp_df[(cts_grp_df["content_time_spent"] > MIN_CONTENT_TIME_SPENT)]

# formatting split dates as datetime
print("Filtering on train, val and test period (filtering task 6/7)")
train_val_datetime = SPLIT_DATE_TRAIN_VAL + " 00:00:00"
val_test_datetime = SPLIT_DATE_VAL_TEST + " 00:00:00"

# users in each of the three periods
train_users = filtered_df[filtered_df["date_time"] < train_val_datetime]
val_users = filtered_df[(filtered_df["date_time"] >= train_val_datetime) & 
                        (filtered_df["date_time"] < val_test_datetime)]["user_id"]
test_users = filtered_df[filtered_df["date_time"] >= train_val_datetime]["user_id"]

# filtering away users below threshold for number of plays per user
grp_train_users = train_users.groupby('user_id')['prd_number'].count()
filtered_train_users = train_users[train_users['user_id']
                                   .isin(grp_train_users[grp_train_users >= MIN_PLAYS_PER_USER].index)]
filtered_train_users_set = set(filtered_train_users["user_id"])

# common users across all three sets 
common_users = filtered_train_users_set.intersection(val_users, test_users)

# Filter the dataframe to include only these users
filtered_df = filtered_df[filtered_df["user_id"].isin(common_users)]

# counting the number of unique users for each podcast show
print("Filtering out infrequent shows (filtering task 7/7)")
show_grp_df = cts_grp_df.groupby('series_title')['user_id'].nunique()

# filtering away episodes below threshold fo number of plays per episode
filtered_df = filtered_df[filtered_df['series_title']
                          .isin(show_grp_df[show_grp_df >= MIN_USERS_PER_SHOW].index)]

# saving the filtered data as parquet file
print("All filtering tasks have been completed! \n Saving filtered data to parquet.")
filtered_df.to_parquet(FILTERED_DATA_PATH, index=False)
