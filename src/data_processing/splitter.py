import pandas as pd
import numpy as np

def group_aware_split(df, target_col, test_size=0.25, random_state=666):
    """
    Splits the DataFrame into training and testing sets, ensuring that
    all samples belonging to the same 'gwgroupid' are kept together
    in either the train or test set.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_col (str): The name of the target column.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Seed for random number generation for reproducibility.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The training DataFrame.
            - pd.DataFrame: The testing DataFrame.
            Returns empty DataFrames if a valid split cannot be achieved.
    """
    unique_groups = df['gwgroupid'].unique()
    np.random.seed(random_state)

    num_test_groups = int(len(unique_groups) * test_size)
    test_groups_selected = np.random.choice(unique_groups, size=num_test_groups, replace=False)

    train_mask = ~df['gwgroupid'].isin(test_groups_selected)
    train_df = df[train_mask].copy()
    test_df = df[~train_mask].copy()

    # Retry a few times if the initial split results in unusable sets
    if train_df.empty or test_df.empty or train_df[target_col].nunique() < 2 or test_df[target_col].nunique() < 2:
        for _attempt in range(3):
            new_seed = random_state + _attempt + 1
            np.random.seed(new_seed)
            test_groups_selected = np.random.choice(unique_groups, size=num_test_groups, replace=False)
            train_mask = ~df['gwgroupid'].isin(test_groups_selected)
            train_df = df[train_mask].copy()
            test_df = df[~train_mask].copy()
            if not train_df.empty and not test_df.empty and train_df[target_col].nunique() >= 2 and test_df[target_col].nunique() >= 2:
                break
        else:
            # If still invalid after retries, return empty
            return pd.DataFrame(), pd.DataFrame()

    return train_df, test_df
