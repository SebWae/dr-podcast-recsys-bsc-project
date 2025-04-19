import os
import sys
import time

from lightfm import LightFM
import pandas as pd
import tqdm

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

# user and item lists
user_list = interaction_matrix.index.to_list()
item_list = interaction_matrix.columns.to_list()

# user and item mappings
user_mapping = {user: i for i, user in enumerate(user_list)}
item_mapping = {i: item for i, item in enumerate(item_list)}

# initializing LightFM model
model = LightFM(loss="logistic", no_components=10)

# Time the fitting process
start_time = time.time()

# Use tqdm to show progress bar during the fitting task
for epoch in tqdm(range(n_epochs), desc="Fitting the model", unit="epoch"):
    model.fit(interaction_matrix, epochs=1, num_threads=1)

end_time = time.time()

# Calculate the time taken
fitting_time = end_time - start_time
print(f"Model fitting took {fitting_time:.2f} seconds")


# Function to get top N recommendations for a user (excluding already rated items)
def get_top_n_recommendations(model, interaction_matrix, user_id, n):
    # obtain index for user_id
    user_idx = user_mapping[user_id]
    
    # Predict scores for all items for the user
    scores = model.predict(user_idx, np.arange(interaction_matrix.shape[1]))
    
    # Get the user's already rated items (i.e., items with non-zero ratings)
    rated_items = interaction_matrix[user_idx].toarray().flatten()
    print(rated_items)
    # Mask out already rated items by setting their scores to 0
    scores[rated_items > 0] = 0
    print(scores)
    # Get the indices of the top N items (excluding rated items)
    top_items = scores.argsort()[-n:][::-1]
    print(top_items)
    # Map the item indices to the actual product numbers
    top_items_prd = [item_mapping[i] for i in top_items]
    
    return top_items_prd

# Get top 3 recommendations for user 1
top_recommendations = get_top_n_recommendations(model, interaction_matrix, user_id="ove", n=2)
print("Top 3 recommendations for user 1:", top_recommendations)