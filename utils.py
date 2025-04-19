import json
import os

from lightfm import LightFM
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


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

    if len1 != len2:
        raise ValueError("Lists must have the same length.")
    else:
        diff_count = 0
        for i in range(len1):
            if list1[i] != list2[i]:
                diff_count += 1
        diff_percentage = diff_count / len1

        return diff_percentage


def get_top_n_recommendations_all_users(model: LightFM, 
                                        interaction_matrix: csr_matrix, 
                                        user_list: list,
                                        item_mapping: dict, 
                                        n) -> list:
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


def save_recommendations(user_mapping: dict, 
                         n_recs: int, 
                         recommendations: list, 
                         recommendations_key: str,
                         file_path: str) -> None:
    """
    Saves the recommendations to a JSON file.

    Parameters:
    - user_mapping:     Dictionary mapping user IDs to indices.
    - n_recs:           Number of recommendations per user.
    - recommendations:  Dictionary of recommendations to save.
    - file_path:        Path to the JSON file (can include folder or just the filename).
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
    data.update(final_dict)

    # writing the data back to the file
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

