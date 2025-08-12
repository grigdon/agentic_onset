import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
# Updated imports to include RandomizedSearchCV and randint for distributions
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, GroupShuffleSplit
from scipy.stats import randint
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

from src.data_processing.preprocessor import create_preprocessor

def train_human_baseline_replication(X_train_full, y_train_full, groups_train_full, features, fixed_params, numeric_features, categorical_features):
    """
    Trains a Random Forest model replicating the human-tuned baseline from the R script.
    This includes downsampling the majority class and performing GridSearchCV for max_features (mtry) tuning.

    Args:
        X_train_full (pd.DataFrame): Full training features.
        y_train_full (pd.Series): Full training target labels.
        groups_train_full (pd.Series): Group IDs for the training data.
        features (list): List of features used for this specific model.
        fixed_params (dict): Dictionary of fixed Random Forest hyperparameters.
        numeric_features (list): Global list of all numeric features.
        categorical_features (list): Global list of all categorical features.

    Returns:
        sklearn.pipeline.Pipeline: The best estimator (trained model) found by GridSearchCV.
    """
    preprocessor = create_preprocessor(features, numeric_features, categorical_features)

    # Downsample the training dataset to handle class imbalance
    class_counts = y_train_full.value_counts()
    if y_train_full.nunique() < 2:
        # If only one class, skip downsampling
        X_train_downsampled = X_train_full
        y_train_downsampled = y_train_full
        groups_train_downsampled = groups_train_full
    else:
        minority_class = y_train_full.value_counts().idxmin()
        majority_class = y_train_full.value_counts().idxmax()
        minority_count = class_counts[minority_class]

        majority_indices = y_train_full[y_train_full == majority_class].index
        minority_indices = y_train_full[y_train_full == minority_class].index

        np.random.seed(fixed_params['random_state'])
        downsampled_majority_indices = np.random.choice(
            majority_indices, minority_count, replace=False
        )

        balanced_indices = np.concatenate([downsampled_majority_indices, minority_indices])

        X_train_downsampled = X_train_full.loc[balanced_indices].copy()
        y_train_downsampled = y_train_full.loc[balanced_indices].copy()
        groups_train_downsampled = groups_train_full.loc[balanced_indices].copy()

    human_rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(**fixed_params))
    ])

    # Tune `max_features` (mtry) for the human baseline
    param_grid_human = {
        'classifier__max_features': ['sqrt', 'log2', 1.0]
    }

    # Set up GroupShuffleSplit for cross-validation, keeping groups together
    human_cv = GroupShuffleSplit(n_splits=10, test_size=0.2, random_state=fixed_params['random_state'])

    grid_search_human = GridSearchCV(
        estimator=human_rf_pipeline,
        param_grid=param_grid_human,
        cv=human_cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0,
        refit=True,
        return_train_score=False
    )

    grid_search_human.fit(X_train_downsampled, y_train_downsampled, groups=groups_train_downsampled)

    return grid_search_human.best_estimator_

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ NEW FUNCTION TO TRAIN THE RANDOMIZED SEARCH BASELINE +++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def train_randomized_search_baseline(X_train_full, y_train_full, groups_train_full, features, numeric_features, categorical_features, random_state=42):
    """
    Trains a Random Forest model using RandomizedSearchCV for hyperparameter tuning.
    Uses the same downsampling and GroupShuffleSplit CV as the human baseline for a fair comparison.
    
    Args:
        (Same as train_human_baseline_replication, plus random_state)

    Returns:
        sklearn.pipeline.Pipeline: The best estimator (trained model) found by RandomizedSearchCV.
    """
    preprocessor = create_preprocessor(features, numeric_features, categorical_features)

    # Perform the exact same downsampling as the human baseline for a fair comparison
    class_counts = y_train_full.value_counts()
    if y_train_full.nunique() < 2:
        X_train_downsampled, y_train_downsampled, groups_train_downsampled = X_train_full, y_train_full, groups_train_full
    else:
        minority_class, majority_class = class_counts.idxmin(), class_counts.idxmax()
        minority_count = class_counts[minority_class]
        majority_indices = y_train_full[y_train_full == majority_class].index
        minority_indices = y_train_full[y_train_full == minority_class].index
        np.random.seed(random_state)
        downsampled_majority_indices = np.random.choice(majority_indices, minority_count, replace=False)
        balanced_indices = np.concatenate([downsampled_majority_indices, minority_indices])
        X_train_downsampled = X_train_full.loc[balanced_indices].copy()
        y_train_downsampled = y_train_full.loc[balanced_indices].copy()
        groups_train_downsampled = groups_train_full.loc[balanced_indices].copy()

    # Define a comprehensive search space for Randomized Search
    param_dist = {
        'classifier__n_estimators': randint(100, 1500),
        'classifier__max_features': ['sqrt', 'log2', 0.2, 0.4, 0.6, 0.8],
        'classifier__max_depth': [None, 10, 20, 30, 40, 50],
        'classifier__min_samples_split': randint(2, 20),
        'classifier__min_samples_leaf': randint(1, 20),
        'classifier__bootstrap': [True, False],
        'classifier__class_weight': ['balanced', 'balanced_subsample', None]
    }

    # Base pipeline with a default classifier
    rs_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=random_state))
    ])

    # Use the same CV strategy as the human baseline
    rs_cv = GroupShuffleSplit(n_splits=10, test_size=0.2, random_state=random_state)

    random_search = RandomizedSearchCV(
        estimator=rs_pipeline,
        param_distributions=param_dist,
        n_iter=100,  # Number of parameter settings that are sampled
        cv=rs_cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1, # Set to 1 or 2 to see progress
        random_state=random_state,
        refit=True
    )

    random_search.fit(X_train_downsampled, y_train_downsampled, groups=groups_train_downsampled)

    print(f"Best Randomized Search AUC (CV): {random_search.best_score_:.4f}")
    print(f"Best Randomized Search Params: {random_search.best_params_}")

    return random_search.best_estimator_

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def evaluate_model(model, X_test, y_test, model_type_name):
    # ... (This function remains unchanged)
    if X_test.empty or y_test.empty or y_test.nunique() < 2:
        return {
            'auc': 0.5,
            'predictions': np.array([]),
            'true_labels': np.array([]),
            'feature_importance': pd.DataFrame(),
            'model_name': model_type_name,
            'classification_report': "Not enough data for evaluation.",
            'confusion_matrix': []
        }

    test_preds_proba = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_preds_proba)
    test_preds_binary = (test_preds_proba > 0.5).astype(int)
    report = classification_report(y_test, test_preds_binary, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, test_preds_binary)
    classifier = model.named_steps['classifier']
    preprocessor = model.named_steps['preprocessor']
    feature_names = []
    numeric_features_in_model = []
    categorical_feature_names_out = []
    for name, transformer, features_in_transformer in preprocessor.transformers_:
        if name == 'num_pipeline':
            numeric_features_in_model.extend(features_in_transformer)
        elif name == 'cat_pipeline':
            ohe = transformer.named_steps['onehot']
            categorical_feature_names_out.extend(list(ohe.get_feature_names_out(features_in_transformer)))
    feature_names = numeric_features_in_model + categorical_feature_names_out
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': classifier.feature_importances_
    }).sort_values('importance', ascending=False)
    return {
        'auc': test_auc,
        'predictions': test_preds_proba,
        'true_labels': y_test.values,
        'feature_importance': importance_df,
        'model_name': model_type_name,
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }