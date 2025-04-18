import os
import pandas as pd
import sys

from lightfm import LightFM

# adding the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))

import utils
from config import (
    TRAIN_DATA_PATH,
)


# loading the train data
train_df = pd.read_parquet(TRAIN_DATA_PATH)

# preparing the interaction matrix
interaction_matrix = utils.prep_interaction_matrix(
    df=train_df,
    user_col="user_id",
    item_col="prd_number",
    rating_col="completion_rate",
)

