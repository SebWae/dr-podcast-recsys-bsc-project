import os
import sys

from lightfm import LightFM
import pandas as pd
from tqdm import tqdm

# adding the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))

from config import (
    TRAIN_DATA_PATH,
    N_COMPONENTS,
    RANDOM_STATE,
    N_RECOMMENDATIONS,
    N_EPOCHS,
    EPSILON,
    RECOMMENDATIONS_KEY_CF,
    RECOMMENDATIONS_PATH,
)
import utils


# loading the train data
train_df = pd.read_parquet(TRAIN_DATA_PATH)

# preparing the interaction matrix
interaction_matrix = utils.prep_interaction_matrix(df=train_df, 
                                                   user_col="user_id", 
                                                   item_col="prd_number", 
                                                   rating_col="completion_rate")

# list of users and items
user_list = sorted(train_df['user_id'].unique().tolist())
n_users = len(user_list)
item_list = sorted(train_df['prd_number'].unique().tolist())

# user and item mappings
user_mapping = {user: i for i, user in enumerate(user_list)}
item_mapping = {i: item for i, item in enumerate(item_list)}

# LightFM model
cf_model = LightFM(loss="logistic", 
                   no_components=N_COMPONENTS, 
                   random_state=RANDOM_STATE)

# initializing recommendations
prev_recommendations = ["0" for _ in range(n_users * N_RECOMMENDATIONS)]

for epoch in tqdm(range(N_EPOCHS)):
    print("\n Epoch", epoch + 1)

    # fitting the model
    cf_model.fit_partial(interaction_matrix)

    # getting the top N recommendations for all users
    recommendations = utils.get_top_n_recommendations_all_users(model=cf_model, 
                                                                interaction_matrix=interaction_matrix, 
                                                                user_list=user_list, 
                                                                item_mapping=item_mapping, 
                                                                n=N_RECOMMENDATIONS)
    
    # computing the proportion of changed recommendations
    diff_percentage = utils.compare_lists(prev_recommendations, recommendations)
    print(f"{diff_percentage*100:.2f}% of the recommendations changed.", )

    # stopping if less than <EPSILON> of the recommendations are changing
    if diff_percentage < EPSILON:
        print("Stopping early")
        print("Extracting recommendations")
        recs_dict = utils.extract_recommendations(recommendations=recommendations,
                                                  user_mapping=user_mapping,
                                                  n_recs=N_RECOMMENDATIONS,
                                                  recommendations_key=RECOMMENDATIONS_KEY_CF)
        print("Saving recommendations")
        utils.save_dict_to_json(data_dict=recs_dict,
                                file_path=RECOMMENDATIONS_PATH)
        break

    prev_recommendations = recommendations
