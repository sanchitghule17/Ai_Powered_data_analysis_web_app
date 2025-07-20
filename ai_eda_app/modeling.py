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
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    *,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[dict[str, float], str, np.ndarray, np.ndarray]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    models = _default_models(task)
    scores: dict[str, float] = {}
    best_name: str | None = None
    best_score: float = -np.inf
    best_preds: np.ndarray | None = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        if task == "regression":
            score = -mean_squared_error(y_test, preds)
        else:
            score = accuracy_score(y_test, preds)

        scores[name] = score

        if score > best_score:
            best_name = name
            best_score = score
            best_preds = preds

    assert best_name is not None and best_preds is not None
    return scores, best_name, y_test, best_preds
