import numpy as np
from ai_eda_app.tuning import optimise_rf

def test_optimise_rf_runs():
    X = np.random.rand(30, 4)
    y = np.random.randint(0, 2, 30)
    params, score = optimise_rf(X, y, "classification", n_trials=5)
    assert "n_estimators" in params and isinstance(score, float)
