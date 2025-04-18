import pandas as pd
from scipy.sparse import coo_matrix

def prep_interaction_matrix(df: pd.DataFrame, 
                            user_col: str, 
                            item_col: str, 
                            rating_col: str) -> coo_matrix:
    """
    Prepares the interaction matrix from the df.

    Parameters:
    - df:           Pandas DataFrame containing the data.
    - user_col:     Column name for users.
    - item_col:     Column name for items.
    - rating_col:   Column name for ratings.

    Returns:
    - interaction_matrix: Sparse matrix of interactions as a coo_matrix from scipy.sparse.
    """
    
    # create the interaction matrix from the DataFrame
    interaction_matrix = df.pivot(index=user_col, columns=item_col, values=rating_col)

    # fill NaN values with 0 for missing user-item pairs
    interaction_matrix.fillna(0, inplace=True)
    matrix_values = interaction_matrix.values

    # convert matrix to COO format
    interaction_matrix = coo_matrix(matrix_values).tocsr()
    
    return interaction_matrix
