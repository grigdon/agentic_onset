import pandas as pd
import os

def load_and_preprocess_data(file_path):
    """
    Loads the main data file and applies initial filtering.

    Args:
        file_path (str): Path to the raw data CSV file.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    df = pd.read_csv(file_path)
    df_filtered = df[(df['isrelevant'] == 1) & (df['exclacc'] == 0)].copy()
    return df_filtered

def prepare_stage_data(df, stage, theoretical_models, stage1_target, stage2_target):
    """
    Prepares data for a specific stage (onset or escalation) by selecting
    relevant features and dropping rows with missing values.

    Args:
        df (pd.DataFrame): The base DataFrame.
        stage (str): The current stage ('onset' or 'escalation').
        theoretical_models (dict): Dictionary of theoretical models and their features.
        stage1_target (str): Column name for the onset target.
        stage2_target (str): Column name for the escalation target.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: DataFrame prepared for the specified stage.
            - str: The target column name for the stage.
    """
    target = stage1_target if stage == "onset" else stage2_target

    all_possible_features = set()
    for theory_features in theoretical_models.values():
        all_possible_features.update(theory_features[stage])

    cols_to_check_for_na = list(all_possible_features) + [target, 'gwgroupid']
    existing_cols_to_check = [col for col in cols_to_check_for_na if col in df.columns]

    df_stage = df[existing_cols_to_check].copy()
    df_stage.dropna(subset=existing_cols_to_check, inplace=True)

    return df_stage, target
