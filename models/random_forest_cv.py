
import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    classification_report
)

# =====================================================
# 1. CONFIGURATION
# =====================================================

DATA_PATH = "data/candidates_train.csv.gz"
MODEL_PATH = "models/random_forest_baseline.joblib"
PREDICTION_PATH = "models/random_forest_oof_predictions.csv"

RANDOM_STATE = 42
N_SPLITS = 5

FEATURES = [
    "dist",
    "rank",
    "dist_minus_min",
    "dist_ratio",
    "n_cands",
    "proj_t",
    "heading_diff",
    "heading_diff_undirected",
    "heading_valid",
    "speed",
    "dt_prev",
    "dt_next",
    "arc_length",
    "out_degree",
    "in_degree",
    "dist_prev_pt",
    "dist_next_pt",
    "in_prev_cands",
    "in_next_cands"
]


# =====================================================
# 2. LOAD DATA
# =====================================================

print("\nLoading training dataset...")

df = pd.read_csv(
    DATA_PATH,
    dtype={"trip_id": str}
)

print("Original dataset shape:", df.shape)

# Keep only reliable candidate rows.
df = df[df["reliable"] == 1].copy()

# Keep valid binary labels.
df = df[df["label"].isin([0, 1])].copy()

print("Reliable dataset shape:", df.shape)

X = df[FEATURES].astype(np.float32)
y = df["label"].astype(int)

# Group by trip so that candidates from the same trip
# cannot appear in both training and validation folds.
groups = df["trip_id"]

print("Number of trips:", groups.nunique())
print("Number of features:", len(FEATURES))

print("\nClass distribution:")
print(y.value_counts(normalize=True) * 100)


# =====================================================
# 3. MODEL DEFINITION
# =====================================================

def create_model():

    return RandomForestClassifier(
        n_estimators=100,
        max_depth=18,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE
    )


# =====================================================
# 4. CROSS-VALIDATION
# =====================================================

cv = GroupKFold(n_splits=N_SPLITS)

oof_probabilities = np.zeros(len(df), dtype=np.float32)

fold_results = []

print("\nStarting 5-Fold GroupKFold Cross-Validation...")
print("=" * 60)

for fold, (train_idx, val_idx) in enumerate(
    cv.split(X, y, groups),
    start=1
):

    print(f"\nFOLD {fold}/{N_SPLITS}")

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]

    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    print("Training rows:", len(train_idx))
    print("Validation rows:", len(val_idx))

    print(
        "Training trips:",
        groups.iloc[train_idx].nunique()
    )

    print(
        "Validation trips:",
        groups.iloc[val_idx].nunique()
    )

    model = create_model()

    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    start_time = time.time()

    probabilities = model.predict_proba(X_val)[:, 1]

    inference_time = time.time() - start_time

    oof_probabilities[val_idx] = probabilities

    # Candidate-level metrics
    predictions = (probabilities >= 0.5).astype(int)

    fold_accuracy = accuracy_score(y_val, predictions)

    fold_f1 = f1_score(
        y_val,
        predictions,
        average="macro",
        zero_division=0
    )

    fold_log_loss = log_loss(y_val, probabilities)

    # Observation-level Top-1 accuracy
    validation_data = df.iloc[val_idx][
        ["trip_id", "point_idx", "arc_id", "rank", "label"]
    ].copy()

    validation_data["probability"] = probabilities

    # Choose the candidate with the highest predicted probability.
    # If probabilities tie, prefer the candidate with the better rank.
    selected = (
        validation_data.sort_values(
            ["trip_id", "point_idx", "probability", "rank"],
            ascending=[True, True, False, True]
        )
        .drop_duplicates(["trip_id", "point_idx"])
    )

    top1_all = selected["label"].mean()

    # Evaluate separately on observations where a correct
    # candidate actually exists.
    observations_with_positive = (
        validation_data.groupby(["trip_id", "point_idx"])["label"]
        .sum()
    )

    hit_keys = observations_with_positive[
        observations_with_positive > 0
    ].index

    selected_hit = selected.set_index(
        ["trip_id", "point_idx"]
    ).loc[hit_keys]

    top1_hit = selected_hit["label"].mean()

    print("\nFold results:")
    print(f"Candidate Accuracy : {fold_accuracy:.4f}")
    print(f"Macro F1           : {fold_f1:.4f}")
    print(f"Log Loss           : {fold_log_loss:.4f}")
    print(f"Top-1 Accuracy     : {top1_all:.4f}")
    print(f"Top-1 on hit points: {top1_hit:.4f}")
    print(f"Training time      : {training_time:.2f} seconds")
    print(f"Inference time     : {inference_time:.2f} seconds")

    fold_results.append({
        "fold": fold,
        "candidate_accuracy": fold_accuracy,
        "macro_f1": fold_f1,
        "log_loss": fold_log_loss,
        "top1_all": top1_all,
        "top1_hit": top1_hit,
        "training_time": training_time,
        "inference_time": inference_time
    })


# =====================================================
# 5. OVERALL OUT-OF-FOLD EVALUATION
# =====================================================

print("\n" + "=" * 60)
print("OVERALL CROSS-VALIDATION RESULTS")
print("=" * 60)

oof_predictions = (oof_probabilities >= 0.5).astype(int)

print(
    "Candidate Accuracy:",
    accuracy_score(y, oof_predictions)
)

print(
    "Macro F1:",
    f1_score(
        y,
        oof_predictions,
        average="macro",
        zero_division=0
    )
)

print(
    "Log Loss:",
    log_loss(y, oof_probabilities)
)

print("\nClassification Report:")
print(
    classification_report(
        y,
        oof_predictions,
        target_names=["Negative", "Positive"],
        zero_division=0
    )
)

# Calculate observation-level OOF Top-1 accuracy.
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

positive_per_observation = (
    oof_data.groupby(["trip_id", "point_idx"])["label"]
    .sum()
)

hit_keys = positive_per_observation[
    positive_per_observation > 0
].index

selected_hit = selected.set_index(
    ["trip_id", "point_idx"]
).loc[hit_keys]

top1_hit = selected_hit["label"].mean()

print("\nObservation-level metrics:")
print(f"Top-1 Accuracy (all reliable observations): {top1_all:.4f}")
print(f"Top-1 Accuracy (candidate-present observations): {top1_hit:.4f}")

print("\nFold-wise summary:")
print(pd.DataFrame(fold_results).to_string(index=False))


# =====================================================
# 6. SAVE OUT-OF-FOLD PREDICTIONS
# =====================================================

os.makedirs("models", exist_ok=True)

oof_data[
    ["trip_id", "point_idx", "arc_id", "label", "probability"]
].to_csv(PREDICTION_PATH, index=False)

print("\nOOF predictions saved to:", PREDICTION_PATH)


# =====================================================
# 7. TRAIN FINAL RANDOM FOREST ON ALL RELIABLE DATA
# =====================================================

print("\nTraining final Random Forest on all reliable data...")

final_model = create_model()

start_time = time.time()

final_model.fit(X, y)

final_training_time = time.time() - start_time

joblib.dump(final_model, MODEL_PATH)

print("Final model saved to:", MODEL_PATH)
print(f"Final training time: {final_training_time:.2f} seconds")

print("\nRANDOM FOREST TRAINING COMPLETE!")
