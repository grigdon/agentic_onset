from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def create_preprocessor(features, numeric_features_list, categorical_features_list):
    numeric_features = [f for f in features if f in numeric_features_list]
    categorical_features = [f for f in features if f in categorical_features_list]

    transformers = []

    if numeric_features:
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median'))
        ])
        transformers.append(('num_pipeline', numeric_transformer, numeric_features))

    if categorical_features:
        categorical_transformer = Pipeline([
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
        ])
        transformers.append(('cat_pipeline', categorical_transformer, categorical_features))

    if not transformers:
        return 'passthrough'

    return ColumnTransformer(transformers=transformers, remainder='drop')
