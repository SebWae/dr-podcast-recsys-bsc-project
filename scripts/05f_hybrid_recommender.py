import json
import os
import sys

# adding the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))

from config import (
    SCORES_PATH,
    RECOMMENDATIONS_KEY_CF,
    RECOMMENDATIONS_KEY_CB_COMBI,
    N_RECOMMENDATIONS,
    LAMBDA,
    RECOMMENDATIONS_PATH,
    RECOMMENDATIONS_KEY_HYBRID,
)
import utils.utils as utils

# loading scores from cf and cb recommender
print(f"Loading utils dictionaries from {SCORES_PATH}.")
with open(SCORES_PATH, "r") as file:
    scores_dicts = json.load(file)

cf_scores = scores_dicts[RECOMMENDATIONS_KEY_CF]
cb_scores = scores_dicts[RECOMMENDATIONS_KEY_CB_COMBI]

# all users
cf_users = set(cf_scores.keys())
cb_users = set(cb_scores.keys())
users = cf_users.union(cb_users)

# generating recommendations from hybrid recommender with optimal lambda hyperparameter value
print("Generating recommendations from hybrid recommender with optimal lambda hyperparameter value")
hybrid_scores = utils.get_hybrid_scores(scores_dict_1=cf_scores,
                                        scores_dict_2=cb_scores,
                                        users=users,
                                        _lambda=LAMBDA)
    
hybrid_recs = utils.extract_recs(scores_dict=hybrid_scores,
                                 n_recs=N_RECOMMENDATIONS)

# saving recommendations
print(f"Saving recommendations to {RECOMMENDATIONS_PATH}.")
recs_dict_key = {RECOMMENDATIONS_KEY_HYBRID: hybrid_recs}
utils.save_dict_to_json(data_dict=recs_dict_key, 
                        file_path=RECOMMENDATIONS_PATH)
