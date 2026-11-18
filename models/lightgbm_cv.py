
import os
import time
import joblib
import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    classification_report
)

DATA_PATH = "data/candidates_train.csv.gz"
MODEL_PATH = "models/lightgbm_baseline.joblib"
PREDICTION_PATH = "models/lightgbm_oof_predictions.csv"

N_SPLITS = 5
RANDOM_STATE = 42

FEATURES = [
    "dist", "rank", "dist_minus_min", "dist_ratio",
    "n_cands", "proj_t", "heading_diff",
    "heading_diff_undirected", "heading_valid",
    "speed", "dt_prev", "dt_next", "arc_length",
    "out_degree", "in_degree", "dist_prev_pt",
    "dist_next_pt", "in_prev_cands", "in_next_cands"
]


def create_model():
    return LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=30,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1
    )


print("\nLoading training data...")

df = pd.read_csv(
    DATA_PATH,
    dtype={"trip_id": str}
)

df = df[df["reliable"] == 1].copy()
df = df[df["label"].isin([0, 1])].copy()

X = df[FEATURES].astype(np.float32)
y = df["label"].astype(int)
groups = df["trip_id"]

print("Training rows:", len(df))
print("Training trips:", groups.nunique())
print("Features:", len(FEATURES))

cv = GroupKFold(n_splits=N_SPLITS)

oof_probabilities = np.zeros(len(df), dtype=np.float32)
fold_results = []

print("\nStarting LightGBM 5-Fold GroupKFold CV...")

for fold, (train_idx, val_idx) in enumerate(
    cv.split(X, y, groups), start=1
):

    print(f"\nFOLD {fold}/{N_SPLITS}")

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]
    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    model = create_model()

    start = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start

    start = time.time()
    probabilities = model.predict_proba(X_val)[:, 1]
    inference_time = time.time() - start

    oof_probabilities[val_idx] = probabilities

    predictions = (probabilities >= 0.5).astype(int)

    candidate_accuracy = accuracy_score(y_val, predictions)
    macro_f1 = f1_score(
        y_val, predictions, average="macro", zero_division=0
    )
    fold_log_loss = log_loss(y_val, probabilities)

    # Observation-level Top-1
    val_data = df.iloc[val_idx][
        ["trip_id", "point_idx", "arc_id", "rank", "label"]
    ].copy()

    val_data["probability"] = probabilities

    selected = (
        val_data.sort_values(
            ["trip_id", "point_idx", "probability", "rank"],
            ascending=[True, True, False, True]
        )
        .drop_duplicates(["trip_id", "point_idx"])
    )

    top1_all = selected["label"].mean()

    positive_counts = val_data.groupby(
        ["trip_id", "point_idx"]
    )["label"].sum()

    hit_keys = positive_counts[positive_counts > 0].index

    selected_hit = selected.set_index(
        ["trip_id", "point_idx"]
    ).loc[hit_keys]

    top1_hit = selected_hit["label"].mean()

    print(f"Candidate Accuracy : {candidate_accuracy:.4f}")
    print(f"Macro F1           : {macro_f1:.4f}")
    print(f"Log Loss           : {fold_log_loss:.4f}")
    print(f"Top-1 all          : {top1_all:.4f}")
    print(f"Top-1 hit points   : {top1_hit:.4f}")
    print(f"Training time      : {training_time:.2f}s")
    print(f"Inference time     : {inference_time:.2f}s")

    fold_results.append({
        "fold": fold,
        "candidate_accuracy": candidate_accuracy,
        "macro_f1": macro_f1,
        "log_loss": fold_log_loss,
        "top1_all": top1_all,
        "top1_hit": top1_hit,
        "training_time": training_time,
        "inference_time": inference_time
    })


print("\n========== OVERALL LIGHTGBM RESULTS ==========")

oof_predictions = (oof_probabilities >= 0.5).astype(int)

print("Candidate Accuracy:",
      accuracy_score(y, oof_predictions))

print("Macro F1:",
      f1_score(y, oof_predictions, average="macro", zero_division=0))

print("Log Loss:", log_loss(y, oof_probabilities))

print("\nClassification Report:")
print(classification_report(
    y,
    oof_predictions,
    target_names=["Negative", "Positive"],
    zero_division=0
))

oof_data = df[
    ["trip_id", "point_idx", "arc_id", "rank", "label"]
].copy()

oof_data["probability"] = oof_probabilities

selected = (
    oof_data.sort_values(
        ["trip_id", "point_idx", "probability", "rank"],
        ascending=[True, True, False, True]
    )
    .drop_duplicates(["trip_id", "point_idx"])
)

top1_all = selected["label"].mean()

positive_counts = oof_data.groupby(
    ["trip_id", "point_idx"]
)["label"].sum()

hit_keys = positive_counts[positive_counts > 0].index

selected_hit = selected.set_index(
    ["trip_id", "point_idx"]
).loc[hit_keys]

top1_hit = selected_hit["label"].mean()

print(f"\nTop-1 all reliable observations: {top1_all:.4f}")
print(f"Top-1 candidate-present observations: {top1_hit:.4f}")

print("\nFold summary:")
print(pd.DataFrame(fold_results).to_string(index=False))

os.makedirs("models", exist_ok=True)

oof_data[
    ["trip_id", "point_idx", "arc_id", "label", "probability"]
].to_csv(PREDICTION_PATH, index=False)

print("\nOOF predictions saved:", PREDICTION_PATH)

print("\nTraining final LightGBM model...")

final_model = create_model()

start = time.time()
final_model.fit(X, y)
final_training_time = time.time() - start

joblib.dump(final_model, MODEL_PATH)

print("Final model saved:", MODEL_PATH)
print(f"Final training time: {final_training_time:.2f}s")
print("\nLIGHTGBM COMPLETE!")
