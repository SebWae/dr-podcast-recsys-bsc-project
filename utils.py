from collections import defaultdict
import json
import os
import sys
from typing import Iterable

from lenskit.algorithms.als import BiasedMF
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# adding the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))

from config import (
    UTILS_PATH,
    UTILS_INTERACTIONS_PATH,
)


def compute_dcg(recommendations: list, gain_dict: dict) -> float:
    """
    Computes the Discounted Cumulative Gain (DCG) for a list of recommendations.
    
    Parameters:
    - recommendations:  List of recommended items.
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


def compute_diversity(recommendations: list, 
                      emb_dict: dict,
                      weights_dict: dict) -> float:
    """
    Computes the diversity as the average intra-list distance (AILD) of a recommendations list.
    Cosine similarity is chosen as the distance measure and is transformed to be in the range [0,1].

    Parameters:
    - recommendations:  List of recommended items.
    - emb_dict:         Dictionary whose keys are item_ids and values and the corresponding item embedding.
    - weights_dict:     Dictionary containing weights for each item pair.

    Returns:
    - total_diversity:  Diversity metric in the range [0,1], 0 indicates similar items, 1 indicates diverse items.
    """
    # retrieving embeddings for items in the recommendations list
    embs = [emb_dict[item] for item in recommendations]
    n_embs = len(embs)

    total_diversity = 0
    # iterating through pair of items in the recommendations list
    for i in range(n_embs):
        emb_1 = embs[i]

        for j in range(i + 1, n_embs):
            emb_2 = embs[j]

            # computing cosine similarity and transforming to diversity metric
            sim = cosine_similarity([emb_1], [emb_2])[0][0]
            diversity = 1 - (sim + 1) / 2

            # retrieving and applying weight
            weight = weights_dict[(i, j)]
            diversity_weighted = diversity * weight
            total_diversity += diversity_weighted

    return total_diversity


def extract_recs(scores_dict: dict,
                 n_recs: int) -> dict:
    """
    Extracts recommendations for each user from a dictionary of scores.

    Parameters:
    - scores_dict:  Dictionary of scores for each item for each user.
    - n_recs:       Number of recommendations per user.

    Returns:
    - recs_dict:    Final dictionary containing recommendations for each user.
    """
    users = scores_dict.keys()
    recs_dict = {}

    for user in users:
        scores = scores_dict[user]

        # sorting score dict by the scores in descending order
        sorted_dict = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
        
        # retrieving top n_recs items
        recs = list(sorted_dict.keys())[:n_recs]

        # saving recs to recs_dict with user as key
        recs_dict[user] = recs

    return recs_dict


def format_embedding_dict(emb_dict: dict) -> dict:
    """
    Converts a dictionary with episodes as keys to have one episode key and feature keys.

    Parameters:
    - emb_dict:         Dictionary in the format {"id1": [embedding1], "id2": [embedding2], ...}

    Returns:
    - formatted_dict:   Dictionary in the format {"episodes": ["id1", "id2", ...], "feature1": [x_11, x_21, ...], "feature2": [x_12, x_22, ...], ...}
    """
    # input dict as dataframe
    emb_df = pd.DataFrame(emb_dict)

    # transposing and resetting index
    reshaped_df = emb_df.T.reset_index()

    # renaming columns
    n_features = len(reshaped_df.columns) - 1
    feature_columns = [f"feature{i+1}" for i in range(n_features)]
    reshaped_df.columns = ["episode"] + feature_columns
    formatted_dict = reshaped_df.to_dict(orient="list")

    return formatted_dict


def get_cb_scores(user: str, 
                  show_episodes: dict, 
                  user_profile: np.ndarray, 
                  item_embeddings: np.ndarray,
                  items: Iterable) -> dict:
    """
    Returns a dictionary of normalized scores for all items possible to recommend for a given user.

    Parameters:
    - user:                     ID of user to generate scores for.
    - show_episodes:            Dictionary containing the shows and episodes listened by every user, {"user_id": {"show_id": ["episode_x", "episode_y"]}}
    - user_profile:             Numpy array containing vector corresponding to the user profile.
    - item_embeddings:          Numpy array containing all item embeddings.
    - items:                    Iterable containing all item_ids.

    Returns:
    - normalized_user_scores:   Dictionary of normalized scores for the given user.
    """
    # items consumed by the user
    user_show_episodes_dict = show_episodes[user]
    user_items = {item for sublist in user_show_episodes_dict.values() for item in sublist}

    # computing all cosine similarities at once for all items
    cos_sim = cosine_similarity(user_profile, item_embeddings).flatten()

    # filtering out items already consumed by the user
    user_scores = {}
    for idx, item in enumerate(items):
        if item not in user_items:
            user_scores[item] = cos_sim[idx]

    # normalizing the user scores
    values = np.array(list(user_scores.values()))
    norm = np.linalg.norm(values)
    normalized_user_scores = {key: value / norm for key, value in user_scores.items()}

    return normalized_user_scores


def get_cf_scores(model: BiasedMF, 
                  items: list,
                  users: Iterable,
                  item_mapping: dict,
                  incl_val_interactions: bool) -> dict:
    """
    Retrieves a dictionary containing scores for relevant items for each user.

    Parameters:
    - model:                    Fitted BiasedMF model from lenskit.algorithms.als.  
    - items:                    List of items available for recommendation.
    - users:                    List of users to generate recommendations for. 
    - item_mapping:             Mapping of item indices to item IDs.
    - incl_val_interactions:    Boolean parameter deciding whether to include interactions from validation data or not.

    Returns:
    - scores_dict:              Dictionary of scores for relevant episodes for each user.
    """
    scores_dict = defaultdict(dict)

    # loading utils dictionaries
    with open(UTILS_PATH, "r") as file:
        utils_dicts = json.load(file)

    with open(UTILS_INTERACTIONS_PATH, "r") as file:
        train_val_interactions = json.load(file)

    # extracting dictionaries
    show_episodes_dict = utils_dicts["show_episodes"]
    if incl_val_interactions:
        user_show_episodes_dict = train_val_interactions["user_show_episodes_val"]
    else:
        user_show_episodes_dict = utils_dicts["user_show_episodes"]

    for user in users:
        # retrieving scores for user
        scores = model.predict_for_user(user, items)
       
        # normalizing the scores
        norm = np.linalg.norm(scores)
        scores = (np.array(scores) / norm).tolist()

        # dictionary of episodes user has listened to for each show
        user_dict = user_show_episodes_dict[user]

        for i, score in enumerate(scores):
            # name of show and its episodes in publication order
            item_id = item_mapping[i]
            episodes = show_episodes_dict[item_id]

            # checking if user has listened to some episodes in show
            if item_id in user_dict:
                listened_episodes = user_dict[item_id]
                last_episode_user = listened_episodes[-1]
                last_episode_show = episodes[-1]

                # checking if user has listened to the most recent episode in show
                if last_episode_user == last_episode_show:
                    episodes_not_listened = sorted(set(episodes) - set(listened_episodes))
                    n_episodes_not_listened = len(episodes_not_listened)

                    # recommending the newest episode the user has not listened to
                    # keep the score at 0 for all episodes in show if user has listened to all episodes published
                    if n_episodes_not_listened > 0:
                        prd_to_recommend = episodes_not_listened[-1]

                # recommending the next episode if new episodes are available
                else:
                    most_recent_episode_index = episodes.index(last_episode_user)
                    prd_to_recommend = episodes[most_recent_episode_index + 1]
            
            # recommending the first episode of show if user has not listened to any episodes in show
            else:
                prd_to_recommend = episodes[0]
            
            scores_dict[user][prd_to_recommend] = score

    return scores_dict


def get_pair_weights(n: int) -> dict:
    """
    Constructs a dictionary of weights for each possible pair of elements from an iterable.

    Parameters:
    - n:                Lenght of list whose pairs to compute a weight for.

    Returns:
    - weights_dict:     Dictionary of weights for each pair of item indices.
    """
    # initializing weights_dict
    weights_dict = {}

    # initial values for each item pair
    for i in range(n):
        for j in range(i+1, n):
            weights_dict[(i, j)] = i + j

    # taking the inverse of each value
    weights_dict = {pair: 1 / v for pair, v in weights_dict.items()}

    # normalizing 
    total_inv_value = sum(weights_dict.values())
    weights_dict = {pair: v / total_inv_value for pair, v in weights_dict.items()}

    return weights_dict


def get_ratings_dict(data: pd.DataFrame,
                     user_col: str,
                     item_col: str,
                     ratings_col: str) -> dict:
    """
    Constructs a dictionary of completion rates for every user and the items they have consumed.

    Parameters:
    - data: Dataframe containing the rated user-item interactions.
    - user_col:         Name of the user column in the dataframe.
    - item_col:         Name of the item column in the dataframe.
    - ratings_col:      Name of the ratings column in the dataframe.

    Returns:
    - ratings_dict:     Dictionary containing ratings for rated items for each user.
    """
    ratings_dict = defaultdict(dict)

    # iterating through the rows of the data to build the dictionary
    for _, row in data.iterrows():
        user = row[user_col]
        prd = row[item_col]
        completion_rate = row[ratings_col]
        
        # adding the rating to the dictionary
        ratings_dict[user][prd] = completion_rate
    
    return ratings_dict


def get_user_profile(emb_size: int, 
                     user_int: pd.DataFrame, 
                     time_col: str, 
                     item_col: str,
                     emb_dict: dict,
                     wght_scheme="inverse") -> np.ndarray:
    """
    Function to build user profiles as a numpy array for a content-based recommender.

    Parameter:
    emb_size:       Dimension of embeddings and the user profile. 
    user_int:       Dataframe containing the interactions of the user to build a profile for.
    time_col:       Name of the column containing the temporal data indicating how long time since an interaction took place.
    item_col:       Name of the column containing item IDs.
    emb_df:         Dictionary containing the embeddings for any item. 
    wght_scheme:    Method to weight the interactions, must be one of 'inverse' or 'linear'.

    Returns:
    user_profile:   The resulting user profile as a numpy array. 
    """
    # initialize user profile (embedding)
    user_profile = np.zeros(emb_size)

    # time column to numpy array
    time_list = user_int[time_col].to_numpy() 
    
    # initial weights for each weighting method
    if wght_scheme == "inverse":
        weights = 1 / time_list
    elif wght_scheme == "linear":
        max_time = max(time_list)
        weights = max_time - time_list + 1
    else:
        raise ValueError("The wght_method should be one of 'inverse' or 'linear'.")

    # normalizing the weights
    weight_total = np.sum(weights)

    # retrieving embeddings for items consumed by user
    item_ids = user_int[item_col].to_numpy()
    embeddings = np.array([emb_dict[item_id] for item_id in item_ids], dtype=np.float64)

    # applying weights to embeddings and accumulate to user profile
    user_profile += np.sum(embeddings * (weights[:, np.newaxis] / weight_total), axis=0)

    # reshaping the user profile to a 2D numpy array
    user_profile = user_profile.reshape(1, -1)

    return user_profile


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
