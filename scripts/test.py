from lightfm import LightFM
import numpy as np
from scipy.sparse import coo_matrix
import pandas as pd
import time
from tqdm import tqdm  

n_epochs = 10

# Dummy data
# df = pd.DataFrame({
#     'user_id': ["1", "1", "2", "2", "3", "3"],
#     'prd_number': ["101", "102", "101", "103", "102", "104"],
#     'completion_rate': [0.5, 0.8, 0.6, 0.7, 0.9, 0.4]
# })

# loading training data
df = pd.read_parquet("data/podcast_data_train.parquet")

# Pivot to interaction matrix
interaction_matrix = df.pivot(index="user_id", columns="prd_number", values="completion_rate").fillna(0)
item_mapping = interaction_matrix.columns.to_list()
print(interaction_matrix)

interaction_matrix = coo_matrix(interaction_matrix.values).tocsr()

# LightFM model
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
def get_top_n_recommendations(model, interaction_matrix, user_id, n=3):
    # Predict scores for all items for the user
    scores = model.predict(user_id, np.arange(interaction_matrix.shape[1]))
    
    # Get the user's already rated items (i.e., items with non-zero ratings)
    rated_items = interaction_matrix[user_id].toarray().flatten()
    
    # Mask out already rated items by setting their scores to 0
    scores[rated_items > 0] = 0
    print(scores)
    # Get the indices of the top N items (excluding rated items)
    top_items = scores.argsort()[-n:][::-1]

    # Map the item indices to the actual product numbers
    top_items_prd = [item_mapping[i] for i in top_items]
    
    return top_items_prd

# Get top 3 recommendations for user 1
top_recommendations = get_top_n_recommendations(model, interaction_matrix, user_id=1, n=1)
print("Top 3 recommendations for user 1:", top_recommendations)