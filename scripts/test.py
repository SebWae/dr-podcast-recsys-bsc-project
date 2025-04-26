
from lenskit.algorithms.als import BiasedMF
import pandas as pd
import numpy as np

# your data should be a DataFrame with columns: user, item, rating
ratings = pd.DataFrame({
    'user': ["1", "2", "1", "3", "2"],
    'item': [10, 10, 20, 30, 30],
    'rating': [0.9, 0.9, 0.9, 0.5, 0.9]
})

# Set up the matrix factorization model
mf = BiasedMF(features=20, iterations=50, damping=5.0)

# Train the model
mf.fit(ratings)

# Predict scores
scores = mf.predict_for_user("1", [10, 20, 30])

# normalizing the scores
norm = np.linalg.norm(scores)
scores = (np.array(scores) / norm).tolist()
print(scores)
# for item, score in scores:
#     print(item, score)
