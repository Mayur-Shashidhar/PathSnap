
import json
import os
import time
import optuna
import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

DATA_PATH = "data/candidates_train.csv.gz"
OUTPUT_PATH = "tuning/lightgbm_best_params.json"

FEATURES = [
    "dist", "rank", "dist_minus_min", "dist_ratio",
    "n_cands", "proj_t", "heading_diff",
    "heading_diff_undirected", "heading_valid",
    "speed", "dt_prev", "dt_next", "arc_length",
    "out_degree", "in_degree", "dist_prev_pt",
    "dist_next_pt", "in_prev_cands", "in_next_cands"
]

print("Loading data...")
df = pd.read_csv(DATA_PATH, dtype={"trip_id": str})
df = df[df["reliable"] == 1].copy()
df = df[df["label"].isin([0, 1])].copy()

X = df[FEATURES].astype(np.float32)
y = df["label"].astype(int)
groups = df["trip_id"]

# Three grouped folds for efficient hyperparameter search.
cv = GroupKFold(n_splits=3)


def observation_top1(validation, probabilities):
    scored = validation[
        ["trip_id", "point_idx", "arc_id", "rank", "label"]
    ].copy()

    scored["probability"] = probabilities

    selected = (
        scored.sort_values(
            ["trip_id", "point_idx", "probability", "rank"],
            ascending=[True, True, False, True]
        )
        .drop_duplicates(["trip_id", "point_idx"])
    )

    return selected["label"].mean()


def objective(trial):

    params = {
        "objective": "binary",
        "n_estimators": trial.suggest_int(
            "n_estimators", 200, 700, step=100
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.02, 0.12, log=True
        ),
        "num_leaves": trial.suggest_int(
            "num_leaves", 15, 63
        ),
        "max_depth": trial.suggest_int(
            "max_depth", 4, 12
        ),
        "min_child_samples": trial.suggest_int(
            "min_child_samples", 15, 80
        ),
        "subsample": trial.suggest_float(
            "subsample", 0.7, 1.0
        ),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.7, 1.0
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha", 1e-4, 5.0, log=True
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", 1e-4, 10.0, log=True
        ),
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1
    }

    scores = []

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X, y, groups), start=1
    ):
        model = LGBMClassifier(**params)

        model.fit(X.iloc[train_idx], y.iloc[train_idx])

        probabilities = model.predict_proba(
            X.iloc[val_idx]
        )[:, 1]

        score = observation_top1(
            df.iloc[val_idx], probabilities
        )

        scores.append(score)

    mean_score = float(np.mean(scores))

    trial.set_user_attr("fold_scores", scores)

    return mean_score


if __name__ == "__main__":

    os.makedirs("tuning", exist_ok=True)

    study = optuna.create_study(
        direction="maximize",
        study_name="pathsnap_lightgbm_top1"
    )

    start = time.time()

    # Start with 15 trials. Increase later if time permits.
    study.optimize(objective, n_trials=15)

    print("\n========== OPTUNA RESULTS ==========")
    print("Best Top-1 CV score:", study.best_value)
    print("Best parameters:")
    print(study.best_params)

    output = {
        "best_top1_cv": study.best_value,
        "best_params": study.best_params
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=4)

    print("\nSaved:", OUTPUT_PATH)
    print("Total tuning time:", round(time.time() - start, 2), "seconds")
