from .data_preprocessing import pipeline_preprocessor

import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


def create_XGBoost() -> Pipeline:
    """
    Create a machine learning model pipeline.

    Returns:
        Pipeline: A scikit-learn Pipeline object with preprocessing and model.
    """

    column_transformer = pipeline_preprocessor()

    model = Pipeline([
        ("preprocessor", column_transformer),
        ("clf", XGBClassifier())
    ])

    return model

