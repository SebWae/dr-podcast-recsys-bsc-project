import json
import os
import sys
from typing import Tuple

from lightfm import LightFM
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import entropy

# adding the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))

from config import (
    RANDOM_STATE,
)


def compare_lists(list1: list, list2: list) -> float:
    """
    Compares two lists and returns the percentage of differences.

    Parameters:
    - list1:            First list to compare.
    - list2:            Second list to compare.

    Returns:
    - diff_percentage:  Percentage of differences between the two lists.
    """
    len1 = len(list1)
    len2 = len(list2)

    # lists must have the same length
    if len1 != len2:
        raise ValueError("Lists must have the same length.")
    else:
        diff_count = 0
        # counting the number of differences between the two lists
        for i in range(len1):
            if list1[i] != list2[i]:
                diff_count += 1
        
        # calculating the percentage of differences
        diff_percentage = diff_count / len1

        return diff_percentage


def compute_dcg(recommendations: list, gain_dict: dict) -> float:
    """
    Computes the Discounted Cumulative Gain (DCG) for a list of recommendations.
    
    Parameters:
    - recommendations:   List of recommended items.
    - gain_dict:        Dictionary mapping items to their gains.

    Returns:
    - dcg:              Discounted Cumulative Gain for the recommendations.
    """
    dcg = 0

    # iterating through the recommendations and calculating DCG
    for j, item in enumerate(recommendations):

        # only obtaining a discounted gain for items in the gain_dict
        if item in gain_dict:
            gain = gain_dict[item]
            discounted_gain = gain / np.log2(j + 2)
            dcg += discounted_gain

    return dcg


def extract_recommendations(recommendations: list,
                            user_mapping: dict, 
                            n_recs: int,
                            recommendations_key: str) -> dict:
    """
    Extracts recommendations for each user from the list of recommendations.

    Parameters:
    - recommendations:      List of recommended items.
    - user_mapping:         Dictionary mapping user IDs to indices.
    - n_recs:               Number of recommendations per user.
    - recommendations_key:  Key for the recommendations in the final dictionary.

    Returns:
    - final_dict:           Final dictionary containing recommendations for each user.
    """
    # initializing dictionary to hold recommendations for each user
    recommendations_dict = {user_id: [] for user_id in user_mapping.keys()}
    n_users = len(user_mapping)

    # extracting recommendations for each user
    for i in range(n_users):
        for j in range(n_recs):
            rec = recommendations[i * n_recs + j]
            user_id = list(user_mapping.keys())[i]
            recommendations_dict[user_id].append(rec)
    
    # creating final dictionary
    final_dict = {recommendations_key: recommendations_dict}

    return final_dict


def format_embedding_dict(emb_dict: dict) -> dict:
    """
    Converts a dictionary with episodes as keys to have one episode key and feature keys.

    Parameters:
    - emb_dict:         Dictionary in the format {"id1": [embedding1], "id2": [embedding2], ...}

    Returnes:
    - reshaped_dict:    Dictionary in the format {"episodes": ["id1", "id2", ...], "feature1": [x_11, x_21, ...], "feature2": [x_12, x_22, ...], ...}
    """
    # input dict as dataframe
    emb_df = pd.DataFrame(emb_dict)

    # transposing and resetting index
    reshaped_df = emb_df.T.reset_index()

    # renaming columns
    n_features = len(reshaped_df.columns) - 1
    feature_columns = [f"feature{i+1}" for i in range(n_features)]
    reshaped_df.columns = ['episode'] + feature_columns
    formatted_dict = reshaped_df.to_dict(orient="list")

    return formatted_dict

def get_top_n_recommendations_all_users(model: LightFM, 
                                        interaction_matrix: csr_matrix, 
                                        user_list: list,
                                        item_mapping: dict, 
                                        n,
                                        item_matrix: csr_matrix = None) -> list:
    """
    Retrieves the top N recommendations for all users.

    Parameters:
    - model:                Trained LightFM model.  
    - interaction_matrix:   Sparse matrix of interactions in scr format.
    - user_list:            List of user IDs.
    - item_mapping:         Mapping of item indices to actual product numbers.
    - n:                    Number of recommendations to retrieve.

    Returns:
    - recommendations:      List of recommended items for all users.
    """
    recommendations = []
    user_mapping = {user: i for i, user in enumerate(user_list)}

    for user_id in user_list:
        # retrieving the index for user_id
        user_idx = user_mapping[user_id]

        # retrieving scores for all items for the user
        if item_matrix is not None:
            n_items = item_matrix.shape[0]
            scores = model.predict(user_idx, np.arange(n_items), item_features=item_matrix)
        else:
            scores = model.predict(user_idx, np.arange(interaction_matrix.shape[1]))
        
        # setting the scores of consumed items to 0 so they won't be recommended
        consumed_items = interaction_matrix[user_idx].nonzero()[1]
        scores[consumed_items] = 0

        # getting the indices of the top n items (excluding rated items)
        top_items = scores.argsort()[-n:][::-1]
        
        # mapping the item indices to the prd_numbers
        top_items_prd = [item_mapping[i] for i in top_items]
        
        # appending to the list of recommendations
        for item in top_items_prd:
            recommendations.append(item)
    
    return recommendations


def prep_interaction_matrix(df: pd.DataFrame, 
                            user_col: str, 
                            item_col: str, 
                            rating_col: str) -> csr_matrix:
    """
    Prepares the interaction matrix from the df.

    Parameters:
    - df:                   Pandas DataFrame containing the data.
    - user_col:             Column name for users.
    - item_col:             Column name for items.
    - rating_col:           Column name for ratings.

    Returns:
    - interaction_matrix:   Sparse matrix of interactions as a csr_matrix from scipy.sparse.
    """
    # create the interaction matrix from the DataFrame
    interaction_matrix = df.pivot(index=user_col, columns=item_col, values=rating_col)

    # fill NaN values with 0 for missing user-item pairs
    interaction_matrix.fillna(0, inplace=True)
    matrix_values = interaction_matrix.values

    # convert matrix to scr format
    interaction_matrix = csr_matrix(matrix_values)
    
    return interaction_matrix


def save_dict_to_json(data_dict: dict, file_path: str) -> None:
    """
    Saves a dictionary to a JSON file.

    Parameters:
    - data_dict:    Dictionary containing the data to be saved.
    - file_path:    Path to the JSON file (can include folder or just the filename).
    """
    # directory name from the file path
    folder = os.path.dirname(file_path)
    
    # if there's a folder part in the file path, make sure it exists
    if folder:
        os.makedirs(folder, exist_ok=True)

    # loading existing data if the file exists, otherwise initialize an empty dict
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
    else:
        data = {}

    # updating the data with the new dictionary of recommendations
    data.update(data_dict)

    # writing the data back to the file
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def permutation_test(dist1: dict, 
                     dist2: dict, 
                     N=1000, 
                     n_permutations=10000) -> Tuple[float, float]:
    """
    Computes the KL Divergence between the input distributions, dist1 and dist2.
    Performs a permutation test to check if dist2 is significantly different from dist1.

    Parameters:
    - dist1:            Dictionary of category: prob, the target distribution P in the KL formula. 
    - dist1:            Dictionary of category: prob, the candidate distribution Q in the KL formula.
    - N:                Size of sample vectors (default value: 100). 
    - n_permutations:   Number of permutations used in the test (default value: 1000).

    Returns: 
    - observed_kl:      Observed KL Divergence between dist1 and dist2. 
    - p_value:          The obtained p-value from the permutation test.
    """
    # setting seed
    np.random.seed(RANDOM_STATE)
    
    # obtain all categories
    dist1_categories = set(dist1.keys())
    dist2_categories = set(dist2.keys())
    all_categories = list(dist1_categories.union(dist2_categories))

    # vectors for all categories
    vec1 = np.array([dist1.get(cat, 0.0) for cat in all_categories])
    vec2 = np.array([dist2.get(cat, 0.0) for cat in all_categories])

    # Laplace smoothing to avoid division by zero
    epsilon = 1e-10
    vec1 += epsilon
    vec2 += epsilon

    # normalizing probability vectors to make sure they sum to 1
    vec1 /= vec1.sum()
    vec2 /= vec2.sum()

    # computing observed KL divergence
    observed_kl = entropy(vec1, vec2)

    # generating synthetic samples
    sample1 = np.random.choice(all_categories, size=N, p=vec1)
    sample2 = np.random.choice(all_categories, size=N, p=vec2)

    # combining samples and initializing sample labels (0 or 1)
    combined = np.concatenate([sample1, sample2])
    labels = np.array([0]*N + [1]*N)

    # performing the permutation test
    permuted_kls = []

    for _ in range(n_permutations):
        # shuffling the labels
        np.random.shuffle(labels)
        g1 = combined[labels == 0]
        g2 = combined[labels == 1]
        
        # generating probability vectors
        p1 = np.array([np.sum(g1 == cat) for cat in all_categories], dtype=float)
        p2 = np.array([np.sum(g2 == cat) for cat in all_categories], dtype=float)

        # applying Laplace smoothing 
        p1 += epsilon
        p2 += epsilon
        
        # computing KL divergence for the current permutation
        kl = entropy(p1, p2)
        permuted_kls.append(kl)

    # computing p-value
    p_value = np.mean(np.array(permuted_kls) >= observed_kl)

    return observed_kl, p_value
    
