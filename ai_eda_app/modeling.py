from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split


def _default_models(task: str) -> dict[str, object]:
    if task == "regression":
        return {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(),
        }
    elif task == "classification":
        return {
            "Random Forest Classifier": RandomForestClassifier(),
        }
    else:
        raise ValueError("task must be 'regression' or 'classification'")


def train_and_evaluate(
    X, y, task, *, test_size=0.20, random_state=42
):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    models = _default_models(task)
    scores, best_name, best_score = {}, None, -np.inf
    best_preds, best_model = None, None

    for name, mdl in models.items():
        mdl.fit(X_tr, y_tr)
        preds = mdl.predict(X_te)

        score = (
            -mean_squared_error(y_te, preds)
            if task == "regression" else accuracy_score(y_te, preds)
        )
        scores[name] = score

        if score > best_score:
            best_name, best_score = name, score
            best_preds, best_model = preds, mdl      # keep the fitted estimator

    return scores, best_name, y_te, best_preds, best_model