import numpy as np
from ai_eda_app.modeling import train_and_evaluate

def test_train_and_evaluate_classification():
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    scores, best_name, y_test, best_preds = train_and_evaluate(X, y, task="classification")

    assert isinstance(scores, dict)
    assert isinstance(best_name, str)
    assert len(y_test) == len(best_preds)

def test_train_and_evaluate_regression():
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    scores, best_name, y_test, best_preds = train_and_evaluate(X, y, task="regression")

    assert isinstance(scores, dict)
    assert isinstance(best_name, str)
    assert len(y_test) == len(best_preds)
