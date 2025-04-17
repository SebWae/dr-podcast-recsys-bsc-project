import numpy as np
import os
import pandas as pd
import sys

# adding the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))

from config import (
    FILTERED_DATA_PATH,
    TRANSFORMED_DATA_PATH,
)


# loading filtered data
transformed_df = pd.read_parquet(FILTERED_DATA_PATH)

# transforming episode duration to float
transformed_df["episode_duration"] = (
    transformed_df["episode_duration"]
    .astype(int) 
)

# splitting the Date:Time column into two separate columns
transformed_df["date"] = transformed_df["date_time"].dt.strftime("%Y-%m-%d")
transformed_df["time"] = transformed_df["date_time"].dt.strftime("%H:%M:%S")
transformed_df.drop(columns=["date_time"], inplace=True)

# computing completion rate
transformed_df["completion_rate"] = np.where(
    transformed_df["content_time_spent"] > transformed_df["episode_duration"],
    1,
    transformed_df["content_time_spent"] / transformed_df["episode_duration"]
)

# renaming variable values
transformed_df["platform"] = transformed_df["platform"].replace({"mobile web": "Web"})
transformed_df["device_type"] = transformed_df["device_type"].replace({"Other": "PC"})

# saving the transformed data as parquet file
transformed_df.to_parquet(TRANSFORMED_DATA_PATH, index=False)