import os
import sys

import pandas as pd
from tqdm import tqdm

# adding the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))

import utils
from config import (
    TRAIN_DATA_PATH,
    N_COMPONENTS,
    RANDOM_STATE,
    N_RECOMMENDATIONS,
    N_EPOCHS,
    RECOMMENDATIONS_KEY_CF,
    RECOMMENDATIONS_PATH,
)


# importing training data
train_df = pd.read_parquet(TRAIN_DATA_PATH)
