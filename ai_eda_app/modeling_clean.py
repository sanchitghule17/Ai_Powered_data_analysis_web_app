import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split


def _default_models(task):
    if task == "regression":
        return {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor()
        }
    elif task == "classification":
        return {
            "Random Forest Classifier": RandomForestClassifier()
        }
    else:
        raise ValueError("Invalid task type")


def train_and_evaluate(X, y, task, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    models = _default_models(task)

    best_score = -np.inf
    best_model = None
    scores = {}
    best_preds = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        if task == "regression":
            score = -mean_squared_error(y_test, preds)
        else:
            score = accuracy_score(y_test, preds)

        scores[name] = score
        if score > best_score:
            best_score = score
            best_model = name
            best_preds = preds

    return scores, best_model, y_test, best_preds
