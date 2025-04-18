import os
import pandas as pd
import sys

# adding the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))

from config import (
    TRANSFORMED_DATA_PATH,
    MIN_PLAYS_PER_EPISODE,
    SPLIT_DATE,
    MIN_PLAYS_PER_USER,
    COLUMNS_TO_KEEP,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
)


# loading the transformed data
transformed_df = pd.read_parquet(TRANSFORMED_DATA_PATH)

# grouping by prd_number and counting the number of plays for each episode
prd_grp_df = transformed_df.groupby('prd_number')['user_id'].count()

# filtering away episodes below threshold fo number of plays per episode
filtered_df = transformed_df[transformed_df['prd_number'].isin(prd_grp_df[prd_grp_df >= MIN_PLAYS_PER_EPISODE].index)]

# applying the global user split
int_train_df = filtered_df[filtered_df['date'] < SPLIT_DATE]
int_test_df = filtered_df[filtered_df['date'] >= SPLIT_DATE]

# number of unique users both in the intermediary train and test data
common_users = set(int_train_df['user_id']).intersection(set(int_test_df['user_id']))

# filter df according to the common users
train_df_common = int_train_df[int_train_df['user_id'].isin(common_users)]
test_df_common = int_test_df[int_test_df['user_id'].isin(common_users)]

# grouping by user_id and counting the number of prd_numbers for each user in the train data
df_grouped_train = train_df_common.groupby('user_id')['prd_number'].count()

# filtering away users below threshold for number of plays per user
train_df = train_df_common[train_df_common['user_id'].isin(df_grouped_train[df_grouped_train >= MIN_PLAYS_PER_USER].index)]
test_df = test_df_common[test_df_common['user_id'].isin(df_grouped_train[df_grouped_train >= MIN_PLAYS_PER_USER].index)]

# only keeping relevant columns
train_df = train_df[COLUMNS_TO_KEEP]
test_df = test_df[COLUMNS_TO_KEEP]

# saving train and test data to parquet files
train_df.to_parquet(TRAIN_DATA_PATH)
test_df.to_parquet(TEST_DATA_PATH)