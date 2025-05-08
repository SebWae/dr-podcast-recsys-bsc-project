from collections import defaultdict
import itertools
import json
import os
import sys
from typing import Iterable, Tuple

from lenskit.algorithms.als import BiasedMF
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity

# adding the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))

from config import (
    RANDOM_STATE,
    UTILS_PATH,
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
                      emb_dict: dict) -> float:
    """
    Computes the diversity as the average intra-list distance (AILD) of a recommendations list.
    Cosine similarity is chosen as the distance measure and is transformed to be in the range [0,1].

    Parameters:
    - recommendations:  List of recommended items.
    - emb_dict:         Dictionary whose keys are item_ids and values and the corresponding item embedding.

    Returns:
    - diversity:        Diversity metric in the range [0,1], 0 indicates similar items, 1 indicates diverse items.
    """
    # retrieving embeddings for items in the recommendations list
    embs = [emb_dict[item] for item in recommendations]
    n_embs = len(embs)

    diversities = []

    # iterating through pair of items in the recommendations list
    for i in range(n_embs):
        emb_1 = embs[i]
        
        for j in range(i + 1, n_embs):
            emb_2 = embs[j]
            sim = cosine_similarity([emb_1], [emb_2])[0][0]
            diversity = 1 - (sim + 1) / 2
            diversities.append(diversity)

    return np.mean(diversities)


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


def get_cf_scores(model: BiasedMF, 
                  items: list,
                  users: Iterable,
                  item_mapping: dict) -> dict:
    """
    Retrieves a dictionary containing scores for relevant items for each user.

    Parameters:
    - model:            Fitted BiasedMF model from lenskit.algorithms.als.  
    - items:            List of items available for recommendation.
    - users:            List of users to generate recommendations for. 
    - item_mapping:     Mapping of item indices to item IDs.

    Returns:
    - scores_dict:      Dictionary of scores for relevant episodes for each user.
    """
    scores_dict = defaultdict(dict)

    # loading utils dictionaries
    with open(UTILS_PATH, "r") as file:
        utils_dicts = json.load(file)

    # extracting dictionaries
    show_episodes_dict = utils_dicts["show_episodes"]
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


def get_hybrid_scores(scores_dict_1: dict,
                      scores_dict_2: dict,
                      users: Iterable,
                      _lambda: int) -> dict:
    """
    Retrieves a dictionary containing scores for relevant items for each user.
    The scores are computed as a weighted average of the scores from scores_dict_1 and scores_dict_2.

    Parameters:
    - scores_dict_1: Dictionary of scores for each item for each user from the first recommender.
    - scores_dict_2: Dictionary of scores for each item for each user from the second recommender.
    - users: Iterable containing all users.
    - _lambda: Parameter controlling the weighting of scores with weights _lambda and (1 - _lambda).

    Returns:
    - scores_dict: Dictionary of scores for each item for each user. 
    """
    # initializing scores_dict
    scores_dict = defaultdict(dict)

    for user in users:
        # initializing dictionary to store scores for each user
        user_scores = {}

        # user scores from cf and cb recommenders
        user_cf_scores = scores_dict_1[user]
        user_cb_scores = scores_dict_2[user]

        # keys from each of the two dictionaries
        cf_items = user_cf_scores.keys()
        cb_items = user_cb_scores.keys()

        # apply weighting by lambda if item both has a cf and a cb score
        for item in cf_items & cb_items: 
            user_scores[item] = _lambda * user_cf_scores[item] + (1 - _lambda) * user_cb_scores[item]

        # retieve cf score if item only has a cf score
        for item in cf_items - cb_items: 
            user_scores[item] = user_cf_scores[item]

        # retieve cb score if item only has a cb score
        for item in cb_items - cf_items:  
            user_scores[item] = user_cb_scores[item]
        
        # adding user_scores to scores_dict
        scores_dict[user] = user_scores

    return scores_dict


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

    return user_profile


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
