from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from xgboost import XGBClassifier, XGBRegressor

CLASSIFICATION_MODELS = {
    "logistic_regression": lambda: LogisticRegression(
        max_iter=1000, random_state=42, class_weight="balanced"
    ),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced"
    ),
    "xgboost": lambda: XGBClassifier(
        n_estimators=100, random_state=42, eval_metric="logloss",
        n_jobs=-1, verbosity=0
    ),
    "lightgbm": lambda: LGBMClassifier(
        n_estimators=100, random_state=42, verbose=-1,
        n_jobs=-1, class_weight="balanced"
    ),
    "extra_trees": lambda: ExtraTreesClassifier(
        n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced"
    ),
}

REGRESSION_MODELS = {
    "ridge": lambda: Ridge(random_state=42),
    "random_forest": lambda: RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
    "xgboost": lambda: XGBRegressor(
        n_estimators=100, random_state=42, n_jobs=-1, verbosity=0
    ),
    "lightgbm": lambda: LGBMRegressor(
        n_estimators=100, random_state=42, verbose=-1, n_jobs=-1
    ),
    "extra_trees": lambda: ExtraTreesRegressor(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
}
