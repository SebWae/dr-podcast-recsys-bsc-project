import os
import pandas as pd
import sys

# adding the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))

from config import (
    TRANSFORMED_DATA_PATH,
    EPISODE_DESCRIPTION_PATH,
    DESCRIPTION_VAR_RENAME_DICT,
    METADATA_COLUMNS,
    METADATA_PATH,
)


# loading the transformed data
transformed_df = pd.read_parquet(TRANSFORMED_DATA_PATH)

# loading the episode description data
descr_df = pd.read_parquet(EPISODE_DESCRIPTION_PATH)

# renaming the columns
descr_df = descr_df.rename(columns=DESCRIPTION_VAR_RENAME_DICT)

# converting the prd_number column to string type
descr_df["prd_number"] = descr_df["prd_number"].astype(str)

# left joining the descr_df onto the transformed_df on the prd_number column
transformed_df_w_descr = pd.merge(transformed_df, descr_df, on="prd_number", how="left")

# grouping by prd_number and selecting metadata columns
metadata_df = transformed_df_w_descr.groupby("prd_number").agg(METADATA_COLUMNS).reset_index()

# extracting episode title from unique title and series title
episode_titles = []

for _, row in metadata_df.iterrows():
    prd = row["prd_number"]
    series_title = row["series_title"]
    unique_title = row["unique_title"]

    series_title_v1 = f"{series_title}:"
    series_title_v2 = f"{series_title} - "

    if unique_title == f"{series_title}_{prd}":
        episode_titles.append("")
    
    else:
        prd_removed = unique_title.replace(f"_{prd}", "")

        if series_title_v1 in prd_removed:
            prd_removed = prd_removed.replace(series_title_v1, "")
        elif series_title_v2 in prd_removed:
            prd_removed = prd_removed.replace(series_title_v2, "")

        episode_titles.append(prd_removed)

# adding the episode titles as a column to metadata_df
metadata_df["episode_title"] = episode_titles

# saving metadata_df
metadata_df.to_parquet(METADATA_PATH)
