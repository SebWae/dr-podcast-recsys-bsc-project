import os
import pandas as pd
import sys

# adding the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))

from config import (
    TRANSFORMED_DATA_PATH,
    MIN_PLAYS_PER_EPISODE,
    SPLIT_DATE,
)


# loading the transformed data
transformed_df = pd.read_parquet(TRANSFORMED_DATA_PATH)

# grouping by prd_number and counting the number of plays for each episode
prd_grp_df = transformed_df.groupby('prd_number')['user_id'].count().sort_values(ascending=True)

# filtering away episodes below threshold fo number of plays
filtered_df = transformed_df[transformed_df['prd_number'].isin(prd_grp_df[prd_grp_df >= MIN_PLAYS_PER_EPISODE].index)]

# applying the global user split
int_train_df = filtered_df[filtered_df['date'] < SPLIT_DATE]
int_test_df = filtered_df[filtered_df['date'] >= SPLIT_DATE]



# filtering away users below threshold for number of plays in the train set
