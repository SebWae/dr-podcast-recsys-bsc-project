import os
import pandas as pd
import sys

# adding the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))

from config import (
    TRANSFORMED_DATA_PATH,
    SPLIT_DATE_TRAIN_VAL,
    SPLIT_DATE_VAL_TEST,
    COLUMNS_TO_KEEP,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
)


# loading the transformed data
transformed_df = pd.read_parquet(TRANSFORMED_DATA_PATH)

# applying the global user split
int_train_df = transformed_df[transformed_df["date"] < SPLIT_DATE_TRAIN_VAL]
int_val_df = transformed_df[(transformed_df["date"] >= SPLIT_DATE_TRAIN_VAL) & (transformed_df["date"] < SPLIT_DATE_VAL_TEST)]
int_test_df = transformed_df[transformed_df["date"] >= SPLIT_DATE_VAL_TEST]

# only keeping relevant columns
train_df = int_train_df[COLUMNS_TO_KEEP]
val_df = int_val_df[COLUMNS_TO_KEEP]
test_df = int_test_df[COLUMNS_TO_KEEP]

# saving train and test data to parquet files
train_df.to_parquet(TRAIN_DATA_PATH)
val_df.to_parquet(VAL_DATA_PATH)
test_df.to_parquet(TEST_DATA_PATH)