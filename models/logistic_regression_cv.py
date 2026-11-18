
import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, log_loss

DATA_PATH = "data/candidates_train.csv.gz"
MODEL_PATH = "models/logistic_regression_baseline.joblib"
PREDICTION_PATH = "models/logistic_regression_oof_predictions.csv"

FEATURES = [
    "dist", "rank", "dist_minus_min", "dist_ratio",
    "n_cands", "proj_t", "heading_diff",
    "heading_diff_undirected", "heading_valid",
    "speed", "dt_prev", "dt_next", "arc_length",
    "out_degree", "in_degree", "dist_prev_pt",
    "dist_next_pt", "in_prev_cands", "in_next_cands"
]

def create_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=500,
            solver="lbfgs",
            random_state=42
        ))
    ])


print("Loading data...")

df = pd.read_csv(DATA_PATH, dtype={"trip_id": str})
df = df[df["reliable"] == 1].copy()

X = df[FEATURES].astype(np.float32)
y = df["label"].astype(int)
groups = df["trip_id"]

cv = GroupKFold(n_splits=5)
oof_probabilities = np.zeros(len(df), dtype=np.float32)

for fold, (train_idx, val_idx) in enumerate(
    cv.split(X, y, groups), start=1
):

    print(f"\nFOLD {fold}/5")

    model = create_model()

    start = time.time()
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    train_time = time.time() - start

    start = time.time()
    probabilities = model.predict_proba(X.iloc[val_idx])[:, 1]
    inference_time = time.time() - start

    oof_probabilities[val_idx] = probabilities

    y_val = y.iloc[val_idx]
    predictions = (probabilities >= 0.5).astype(int)

    print("Candidate Accuracy:",
          accuracy_score(y_val, predictions))

    print("Macro F1:",
          f1_score(y_val, predictions, average="macro"))

    print("Log Loss:", log_loss(y_val, probabilities))

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

    print("Observation Top-1:", selected["label"].mean())

    positive_counts = val_data.groupby(
        ["trip_id", "point_idx"]
    )["label"].sum()

    hit_keys = positive_counts[positive_counts > 0].index

    selected_hit = selected.set_index(
        ["trip_id", "point_idx"]
    ).loc[hit_keys]

    print("Top-1 on candidate-present points:",
          selected_hit["label"].mean())

    print("Training time:", round(train_time, 2), "seconds")
    print("Inference time:", round(inference_time, 2), "seconds")


# Overall OOF evaluation

predictions = (oof_probabilities >= 0.5).astype(int)

print("\n========== LOGISTIC REGRESSION RESULTS ==========")

print("Candidate Accuracy:", accuracy_score(y, predictions))
print("Macro F1:", f1_score(y, predictions, average="macro"))
print("Log Loss:", log_loss(y, oof_probabilities))

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

print("Observation Top-1:", selected["label"].mean())

positive_counts = oof_data.groupby(
    ["trip_id", "point_idx"]
)["label"].sum()

hit_keys = positive_counts[positive_counts > 0].index

selected_hit = selected.set_index(
    ["trip_id", "point_idx"]
).loc[hit_keys]

print("Top-1 on candidate-present points:",
      selected_hit["label"].mean())


# Save predictions

os.makedirs("models", exist_ok=True)

oof_data[
    ["trip_id", "point_idx", "arc_id", "label", "probability"]
].to_csv(PREDICTION_PATH, index=False)

print("Saved:", PREDICTION_PATH)


# Train final model

print("\nTraining final Logistic Regression...")

final_model = create_model()
final_model.fit(X, y)

joblib.dump(final_model, MODEL_PATH)

print("Saved:", MODEL_PATH)
print("\nLOGISTIC REGRESSION COMPLETE!")
