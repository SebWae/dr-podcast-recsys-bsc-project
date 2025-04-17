# TODO: split "unik titel" column to obtain episode title
# TODO: remove stopwords from episode descriptions and apply stemming
# TODO: create embeddings for various level of metadata and weighting schemes

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

print(metadata_df.head())
