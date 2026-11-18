
import os
import json
import time
import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss
)

DATA_PATH = "data/candidates_train.csv.gz"
PARAM_PATH = "tuning/lightgbm_best_params.json"

OUTPUT_DIR = "tuning/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURES = [
    "dist", "rank", "dist_minus_min", "dist_ratio",
    "n_cands", "proj_t", "heading_diff",
    "heading_diff_undirected", "heading_valid",
    "speed", "dt_prev", "dt_next", "arc_length",
    "out_degree", "in_degree", "dist_prev_pt",
    "dist_next_pt", "in_prev_cands", "in_next_cands"
]

BASELINE_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 30,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1
}


def observation_top1(frame, probabilities):
    scored = frame[
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


print("Loading reliable training data...")

df = pd.read_csv(
    DATA_PATH,
    dtype={"trip_id": str}
)

df = df[
    (df["reliable"] == 1) &
    (df["label"].isin([0, 1]))
].copy()

X = df[FEATURES].astype(np.float32)
y = df["label"].astype(int)
groups = df["trip_id"]

with open(PARAM_PATH, "r") as f:
    tuning = json.load(f)

tuned_params = {
    **tuning["best_params"],
    "objective": "binary",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1
}

models = {
    "LightGBM Baseline": BASELINE_PARAMS,
    "LightGBM Tuned": tuned_params
}

cv = GroupKFold(n_splits=5)

all_results = []
all_predictions = []

for model_name, params in models.items():

    print("\n================================")
    print(model_name)
    print("================================")

    oof = np.zeros(len(df), dtype=float)
    fold_times = []

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X, y, groups), start=1
    ):

        model = LGBMClassifier(**params)

        start = time.time()

        model.fit(
            X.iloc[train_idx],
            y.iloc[train_idx]
        )

        probabilities = model.predict_proba(
            X.iloc[val_idx]
        )[:, 1]

        elapsed = time.time() - start
        fold_times.append(elapsed)

        oof[val_idx] = probabilities

        fold_top1 = observation_top1(
            df.iloc[val_idx],
            probabilities
        )

        print(
            f"Fold {fold}: "
            f"Top-1={fold_top1:.4f}, "
            f"Time={elapsed:.2f}s"
        )

    predictions = (oof >= 0.5).astype(int)

    candidate_accuracy = accuracy_score(y, predictions)

    macro_f1 = f1_score(
        y, predictions,
        average="macro"
    )

    loss = log_loss(y, oof)

    top1 = observation_top1(df, oof)

    result = {
        "Model": model_name,
        "Candidate Accuracy": candidate_accuracy,
        "Macro F1": macro_f1,
        "Log Loss": loss,
        "Observation Top-1": top1,
        "Mean Fold Training Time (s)": np.mean(fold_times)
    }

    all_results.append(result)

    pred_frame = df[
        ["trip_id", "point_idx", "arc_id", "label"]
    ].copy()

    pred_frame["probability"] = oof
    pred_frame["model"] = model_name

    all_predictions.append(pred_frame)

    print("\nPooled OOF metrics:")
    for key, value in result.items():
        if isinstance(value, (float, np.floating)):
            print(f"{key}: {value:.5f}")

results = pd.DataFrame(all_results)

results.to_csv(
    f"{OUTPUT_DIR}/baseline_vs_tuned.csv",
    index=False
)

pd.concat(
    all_predictions,
    ignore_index=True
).to_csv(
    f"{OUTPUT_DIR}/baseline_vs_tuned_oof.csv",
    index=False
)

print("\n================================")
print("FINAL COMPARISON")
print("================================")
print(results.to_string(index=False))

print("\nSaved comparison files in:", OUTPUT_DIR)
