import os
import sys

import pandas as pd

# adding the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))

from config import (
    SCORES_PATH_CF,
    SCORES_PATH_CB_COMBI, 
    N_RECOMMENDATIONS,
    LAMBDA_HYBRID,
    RECOMMENDATIONS_PATH,
    RECOMMENDATIONS_KEY_HYBRID,
)
import utils.utils as utils

# loading cf and cb scores
cf_scores_df = pd.read_parquet(SCORES_PATH_CF)
cb_scores_df = pd.read_parquet(SCORES_PATH_CB_COMBI)

# converting the scores dataframes to dictionaries
cf_scores = cf_scores_df.to_dict()
cb_scores = cb_scores_df.to_dict()

# all users
cf_users = set(cf_scores.keys())
cb_users = set(cb_scores.keys())
users = cf_users.union(cb_users)

# generating recommendations from hybrid recommender with optimal lambda hyperparameter value
print("Generating recommendations from hybrid recommender with optimal lambda hyperparameter value")
hybrid_scores = utils.get_hybrid_scores(scores_dict_1=cf_scores,
                                        scores_dict_2=cb_scores,
                                        users=users,
                                        _lambda=LAMBDA_HYBRID)
    
hybrid_recs = utils.extract_recs(scores_dict=hybrid_scores,
                                 n_recs=N_RECOMMENDATIONS)

# saving recommendations
print(f"Saving recommendations to {RECOMMENDATIONS_PATH}.")
recs_dict_key = {RECOMMENDATIONS_KEY_HYBRID: hybrid_recs}
utils.save_dict_to_json(data_dict=recs_dict_key, 
                        file_path=RECOMMENDATIONS_PATH)
