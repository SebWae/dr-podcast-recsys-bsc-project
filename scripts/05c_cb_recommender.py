# TODO: remove stopwords from episode descriptions and apply stemming
# TODO: create embeddings for various level of metadata and weighting schemes

import os
import sys

from lightfm import LightFM
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from tqdm import tqdm

# adding the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))

from config import (
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    N_COMPONENTS,
    RANDOM_STATE,
    N_RECOMMENDATIONS,
    N_EPOCHS,
    EPSILON,
    RECOMMENDATIONS_PATH,
    EMBEDDINGS_TITLE_PATH,
    EMBEDDINGS_DESCR_PATH,
    EMBEDDINGS_COMBI_PATH,
    RECOMMENDATIONS_KEY_CB_COMBI,
    RECOMMENDATIONS_KEY_CB_DESCR,
    RECOMMENDATIONS_KEY_CB_TITLE
)
import utils.utils as utils

# loading train data
train_df = pd.read_parquet(TRAIN_DATA_PATH)

# preparing the interaction matrix
interaction_matrix = utils.prep_interaction_matrix(df=train_df,
                                                   user_col="user_id",
                                                   item_col="prd_number",
                                                   rating_col="completion_rate",
)

# list of users
user_list = sorted(train_df['user_id'].unique().tolist())
n_users = len(user_list)

# loading test data
test_df = pd.read_parquet(TEST_DATA_PATH)

# finding new items
train_items = set(train_df["prd_number"])
test_items = set(test_df["prd_number"])
new_items = test_items.difference(train_items)
all_items = train_items.union(test_items)
item_list = sorted(list(all_items))

# extra matrix of zeros for new items
zero_matrix = np.zeros((n_users, len(new_items)))
interaction_matrix_w_zeros = hstack([interaction_matrix, zero_matrix]).tocsr()

# user and item mappings
user_mapping = {user: i for i, user in enumerate(user_list)}
item_mapping = {i: item for i, item in enumerate(item_list)}

# LightFM model
cb_model = LightFM(loss="logistic", 
                   no_components=N_COMPONENTS, 
                   random_state=RANDOM_STATE)

# paths to embedding locations
emb_paths = [EMBEDDINGS_TITLE_PATH, EMBEDDINGS_DESCR_PATH, EMBEDDINGS_COMBI_PATH]
cb_keys = [RECOMMENDATIONS_KEY_CB_TITLE, RECOMMENDATIONS_KEY_CB_DESCR, RECOMMENDATIONS_KEY_CB_COMBI]

for path, key in zip(emb_paths, cb_keys):
    print(f"\n Generating recommendations for {key}:")
    # initializing recommendations
    prev_recommendations = ["0" for _ in range(n_users * N_RECOMMENDATIONS)]
    prev_diff = 0

    # loading embeddings
    emb_df = pd.read_parquet(path)

    # formatting the emb_df 
    emb_df = emb_df.rename(columns={"episode": "prd_number"})
    emb_dict = emb_df.to_dict(orient="list")
    emb_df_formatted = pd.DataFrame(emb_dict)

    # turning the embedding dataframe into an csr matrix
    item_matrix = emb_df_formatted.drop(columns='prd_number').values
    item_matrix_csr = csr_matrix(item_matrix)

    for epoch in tqdm(range(N_EPOCHS)):
        print("\n Epoch", epoch + 1)

        # fitting the model
        cb_model.fit_partial(interactions=interaction_matrix, item_features=item_matrix_csr)

        # getting the top N recommendations for all users
        recommendations = utils.get_top_n_recommendations_all_users(model=cb_model, 
                                                                    interaction_matrix=interaction_matrix, 
                                                                    user_list=user_list, 
                                                                    item_mapping=item_mapping, 
                                                                    n=N_RECOMMENDATIONS,
                                                                    item_matrix=item_matrix_csr)
        
        # computing the proportion of changed recommendations
        diff_percentage = utils.compare_lists(prev_recommendations, recommendations)
        print(f"{diff_percentage*100:.2f}% of the recommendations changed.", )

        # comparing the diff_percentage with the previous epoch
        change_diff_percentage = abs(diff_percentage - prev_diff)

        # stopping if change_diff_percentage is less than EPSILON
        if change_diff_percentage < EPSILON:
            print("Stopping early")
            print("Extracting recommendations")
            recs_dict = utils.extract_recommendations(recommendations=recommendations,
                                                    user_mapping=user_mapping,
                                                    n_recs=N_RECOMMENDATIONS,
                                                    recommendations_key=key)
            print("Saving recommendations")
            utils.save_dict_to_json(data_dict=recs_dict,
                                    file_path=RECOMMENDATIONS_PATH)
            break
        
        prev_recommendations = recommendations
