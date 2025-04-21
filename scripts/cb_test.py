from lightfm import LightFM
from scipy.sparse import csr_matrix, hstack, identity
import pandas as pd
from tqdm import tqdm
import sys 
import os

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

N_RECOMMENDATIONS = 1

# Dummy data
df = pd.DataFrame({
    'user_id': ["bent", "bent", "bent", "bent", "ove", "ove", "ove", "ove"],
    'prd_number': ["101", "102", "103", "104", "101", "102", "103", "104"],
    'played': [0.9, 0, 0.1, 0, 0, 0, 0, 0]
})

item_features = pd.DataFrame({
    'prd_number': ["101", "102", "103", "104"],
    'feature1': [0.5, 0.45, 0.41, 0.9],
    'feature2': [0.2, 0.25, 0.89, 0.95],
    'feature3': [0.6, 0.6, 0.7, 0.85],
})

item_matrix = item_features.drop(columns='prd_number').values
item_matrix_csr = csr_matrix(item_matrix)

# Number of items
n_items = item_matrix_csr.shape[0]

# Identity matrix for item IDs
item_identities = identity(n_items, format='csr')

# Combine identity matrix with feature matrix to get hybrid features
item_features_hybrid = hstack([item_identities, item_matrix_csr]).tocsr()

# loading training data
# df = pd.read_parquet(TRAIN_DATA_PATH)
interaction_matrix = utils.prep_interaction_matrix(df=df, 
                                                   user_col="user_id", 
                                                   item_col="prd_number", 
                                                   rating_col="played")

# list of users and items
user_list = sorted(df['user_id'].unique().tolist())
n_users = len(user_list)
item_list = sorted(df['prd_number'].unique().tolist())

# user and item mappings
user_mapping = {user: i for i, user in enumerate(user_list)}
item_mapping = {i: item for i, item in enumerate(item_list)}
print(user_mapping)
# LightFM model
cb_model = LightFM(loss="logistic", 
                   no_components=N_COMPONENTS, 
                   random_state=RANDOM_STATE)

prev_recommendations = ["0" for _ in range(n_users * N_RECOMMENDATIONS)]

user_idx = user_mapping["ove"]
for epoch in tqdm(range(N_EPOCHS)):
    print("\n Epoch", epoch + 1)

    cb_model.fit_partial(interactions=interaction_matrix, item_features=item_features_hybrid)

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

    # stopping if less than <EPSILON> of the recommendations are changing
    if diff_percentage < EPSILON:
        print("Stopping early")
        print(recommendations)
        break
    
    prev_recommendations = recommendations



