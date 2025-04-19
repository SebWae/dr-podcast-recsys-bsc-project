from lightfm import LightFM
import numpy as np
from scipy.sparse import coo_matrix
import pandas as pd
from tqdm import tqdm
import sys 
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))

import utils

n_epochs = 1000
n_recommendations = 2

# Dummy data
df = pd.DataFrame({
    'user_id': ["bent", "bent", "ove", "ove", "kirsten", "kirsten"],
    'prd_number': ["101", "103", "101", "103", "102", "104"],
    'completion_rate': [0.5, 0.8, 0.6, 0.7, 0.9, 0.4]
})


# loading training data
# df = pd.read_parquet("data/podcast_data_train.parquet")
interaction_matrix = utils.prep_interaction_matrix(df=df, 
                                                   user_col="user_id", 
                                                   item_col="prd_number", 
                                                   rating_col="completion_rate")
item_list = sorted(df['prd_number'].unique().tolist())
item_mapping = {i: item for i, item in enumerate(item_list)}
user_list = sorted(df['user_id'].unique().tolist())
user_mapping = {user: i for i, user in enumerate(user_list)}
print("User mapping:", user_mapping)
n_users = len(user_list)

# LightFM model
model = LightFM(loss="logistic", no_components=40, random_state=250500)

prev_recommendations = ["0" for _ in range(n_users * n_recommendations)]
prev_diff = 1.1

for epoch in tqdm(range(n_epochs)):
    print("Epoch", epoch + 1)
    model.fit_partial(interaction_matrix)
    recommendations = utils.get_top_n_recommendations_all_users(model, interaction_matrix, user_list, item_mapping, n=n_recommendations)
    print("Recommendations:", recommendations)
    diff_percentage = utils.compare_lists(prev_recommendations, recommendations)
    print("% of recommendations changed:", diff_percentage)
    if diff_percentage >= prev_diff:
        print("Stopping early")
        break
    prev_recommendations = recommendations
    prev_diff = diff_percentage

