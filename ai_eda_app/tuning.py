import optuna
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def optimise_rf(X, y, task: str, n_trials: int = 30, cv: int = 3, seed: int = 42):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "random_state": seed,
        }
        Model = RandomForestRegressor if task == "regression" else RandomForestClassifier
        model = Model(**params)
        score = cross_val_score(model, X, y, cv=cv,
                                scoring="neg_mean_squared_error" if task=="regression" else "accuracy").mean()
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_trial.params, study.best_value
