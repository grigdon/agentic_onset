import pandas as pd

def load_and_preprocess_data(file_path):

    df = pd.read_csv(file_path)

    df_filtered = df[(df['isrelevant'] == 1) & (df['exclacc'] == 0)].copy()
    return df_filtered

def prepare_stage_data(df, stage, theoretical_models, stage1_target, stage2_target):
    target = stage1_target if stage == "onset" else stage2_target

    all_possible_features = set()
    for theory_features in theoretical_models.values():
        all_possible_features.update(theory_features[stage])

    cols_to_check_for_na = list(all_possible_features) + [target, 'gwgroupid']
    existing_cols_to_check = [col for col in cols_to_check_for_na if col in df.columns]

    df_stage = df[existing_cols_to_check].copy()
    df_stage.dropna(subset=existing_cols_to_check, inplace=True)

    return df_stage, target
