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
    FILTERED_DATA_PATH,
)


# loading the raw data 
df = pd.read_parquet(RAW_DATA_PATH)

# renaming the columns 
df = df.rename(columns=VAR_RENAME_DICT)

# changing the datatype for the content time spent column
df["content_time_spent"] = df["content_time_spent"].astype(int)

# user_id and prd_number must exist and content_type must be vod
filtered_df = df[(df["user_id"].notna()) & 
                 (df["prd_number"].notna()) & 
                 (df["content_type"] == "vod")
                 ]

# dropping the content_type attribute
filtered_df.drop(columns=["content_type"], inplace=True)

# only including relevant mobile device types
main_devices = {"Mobile Phone", "Other", "Tablet"}
filtered_df = filtered_df[filtered_df["device_type"].isin(main_devices)]

# excluding some combinations of platform and mobile device type
exclude_combinations = {("mobile web", "Other"), ("web", "Mobile Phone"), ("web", "Tablet")}
filtered_df = filtered_df.loc[~df[["platform", "device_type"]].apply(tuple, axis=1).isin(exclude_combinations)]

# grouping rows by user_id and prd_number
filtered_df["date_time"] = pd.to_datetime(filtered_df["date_time"], format="%d:%m:%Y|%H:%M")
cts_grp_df = filtered_df.groupby(["user_id", "prd_number"]).agg(
    date_time =             ("date_time",           lambda x: x.loc[df.loc[x.index, "content_time_spent"].idxmax()]),
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
cts_grp_df = cts_grp_df[(cts_grp_df["content_time_spent"] > MIN_CONTENT_TIME_SPENT)]

# filter out dates that are outside desired range
cts_grp_df = cts_grp_df[(cts_grp_df["date_time"] >= START_DATE) & (cts_grp_df["date_time"] <= END_DATE)]

# saving the filtered data as parquet file
cts_grp_df.to_parquet(FILTERED_DATA_PATH, index=False)