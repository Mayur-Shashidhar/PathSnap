
import os
import time
import numpy as np
import pandas as pd
import joblib

from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, log_loss

DATA_PATH = "data/candidates_train.csv.gz"
PREDICTION_PATH = "models/knn_oof_predictions.csv"

FEATURES = [
    "dist", "rank", "dist_minus_min", "dist_ratio",
    "n_cands", "proj_t", "heading_diff",
    "heading_diff_undirected", "heading_valid",
    "speed", "dt_prev", "dt_next", "arc_length",
    "out_degree", "in_degree", "dist_prev_pt",
    "dist_next_pt", "in_prev_cands", "in_next_cands"
]

# Keep the prototype count manageable.
PROTOTYPES_PER_CLASS = 3000


def create_prototypes(X, y):

    all_centers = []
    all_labels = []

    for label in [0, 1]:

        X_class = X[y == label]

        n_clusters = min(
            PROTOTYPES_PER_CLASS,
            len(X_class)
        )

        print(f"Compressing class {label}: {len(X_class)} rows")

        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=4096,
            random_state=42,
            n_init=2,
            max_iter=50
        )

        kmeans.fit(X_class)

        all_centers.append(kmeans.cluster_centers_)
        all_labels.extend([label] * n_clusters)

    return (
        np.vstack(all_centers).astype(np.float32),
        np.array(all_labels)
    )


print("Loading dataset...")

df = pd.read_csv(
    DATA_PATH,
    dtype={"trip_id": str}
)

df = df[df["reliable"] == 1].copy()

X = df[FEATURES].to_numpy(dtype=np.float32)
y = df["label"].to_numpy(dtype=int)
groups = df["trip_id"].to_numpy()

cv = GroupKFold(n_splits=5)

oof_probabilities = np.zeros(len(df), dtype=np.float32)

fold_results = []

for fold, (train_idx, val_idx) in enumerate(
    cv.split(X, y, groups), start=1
):

    print(f"\n========== FOLD {fold}/5 ==========")

    X_train = X[train_idx]
    X_val = X[val_idx]

    y_train = y[train_idx]
    y_val = y[val_idx]

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)

    start = time.time()

    X_proto, y_proto = create_prototypes(
        X_train, y_train
    )

    knn = KNeighborsClassifier(
        n_neighbors=15,
        weights="distance",
        metric="euclidean",
        algorithm="brute",
        n_jobs=-1
    )

    knn.fit(X_proto, y_proto)

    training_time = time.time() - start

    start = time.time()

    probabilities = knn.predict_proba(X_val)[:, 1]

    inference_time = time.time() - start

    oof_probabilities[val_idx] = probabilities

    predictions = (probabilities >= 0.5).astype(int)

    print("Candidate Accuracy:",
          accuracy_score(y_val, predictions))

    print("Macro F1:",
          f1_score(y_val, predictions, average="macro"))

    print("Log Loss:",
          log_loss(y_val, probabilities))

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

    print("Observation Top-1:",
          selected["label"].mean())

    positive_counts = val_data.groupby(
        ["trip_id", "point_idx"]
    )["label"].sum()

    hit_keys = positive_counts[
        positive_counts > 0
    ].index

    selected_hit = selected.set_index(
        ["trip_id", "point_idx"]
    ).loc[hit_keys]

    print("Top-1 on candidate-present points:",
          selected_hit["label"].mean())

    print("Training time:", round(training_time, 2), "seconds")
    print("Inference time:", round(inference_time, 2), "seconds")

    fold_results.append({
        "fold": fold,
        "candidate_accuracy": accuracy_score(y_val, predictions),
        "macro_f1": f1_score(y_val, predictions, average="macro"),
        "log_loss": log_loss(y_val, probabilities),
        "top1": selected["label"].mean(),
        "training_time": training_time,
        "inference_time": inference_time
    })


print("\n========== OVERALL KNN RESULTS ==========")

predictions = (oof_probabilities >= 0.5).astype(int)

print("Candidate Accuracy:",
      accuracy_score(y, predictions))

print("Macro F1:",
      f1_score(y, predictions, average="macro"))

print("Log Loss:",
      log_loss(y, oof_probabilities))

oof_data = df[
    ["trip_id", "point_idx", "arc_id", "label"]
].copy()

oof_data["probability"] = oof_probabilities

selected = (
    oof_data.sort_values(
        ["trip_id", "point_idx", "probability", "arc_id"],
        ascending=[True, True, False, True]
    )
    .drop_duplicates(["trip_id", "point_idx"])
)

print("Observation Top-1:",
      selected["label"].mean())

positive_counts = oof_data.groupby(
    ["trip_id", "point_idx"]
)["label"].sum()

hit_keys = positive_counts[
    positive_counts > 0
].index

selected_hit = selected.set_index(
    ["trip_id", "point_idx"]
).loc[hit_keys]

print("Top-1 on candidate-present points:",
      selected_hit["label"].mean())

print("\nFold summary:")
print(pd.DataFrame(fold_results).to_string(index=False))

os.makedirs("models", exist_ok=True)

oof_data.to_csv(
    PREDICTION_PATH,
    index=False
)

print("\nOOF predictions saved:", PREDICTION_PATH)
print("\nKNN COMPLETE!")
