
import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss
)

# =====================================================
# 1. MODEL PREDICTION FILES
# =====================================================

FILES = {
    "Logistic Regression":
        "models/logistic_regression_oof_predictions.csv",

    "KNN":
        "models/knn_oof_predictions.csv",

    "Random Forest":
        "models/random_forest_oof_predictions.csv",

    "LightGBM":
        "models/lightgbm_oof_predictions.csv",

    "MLP":
        "models/mlp_oof_predictions.csv"
}

REQUIRED_COLUMNS = [
    "trip_id", "point_idx", "arc_id",
    "label", "probability"
]

KEY_COLUMNS = [
    "trip_id", "point_idx", "arc_id"
]


# =====================================================
# 2. EVALUATION FUNCTION
# =====================================================

def evaluate_model(name, path, reference=None):

    print(f"\nEvaluating {name}...")

    df = pd.read_csv(
        path,
        dtype={"trip_id": str}
    )

    # Check required columns
    missing = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(
            f"{name} is missing columns: {missing}"
        )

    # Check duplicate candidate rows
    if df.duplicated(KEY_COLUMNS).any():
        raise ValueError(
            f"{name} contains duplicate candidate rows."
        )

    # Check probability validity
    if not np.isfinite(df["probability"]).all():
        raise ValueError(
            f"{name} contains invalid probabilities."
        )

    if not df["probability"].between(0, 1).all():
        raise ValueError(
            f"{name} probabilities must be between 0 and 1."
        )

    # Sort for consistent comparison
    df = df.sort_values(
        KEY_COLUMNS
    ).reset_index(drop=True)

    # Check candidate rows and labels match
    keys = df[
        KEY_COLUMNS + ["label"]
    ]

    if reference is not None and not keys.equals(reference):
        raise ValueError(
            f"{name} does not have the same candidate rows "
            "and labels as the reference model."
        )

    y = df["label"].to_numpy(dtype=int)

    probabilities = df["probability"].to_numpy()

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    # Candidate-level metrics
    accuracy = accuracy_score(y, predictions)

    precision = precision_score(
        y, predictions, zero_division=0
    )

    recall = recall_score(
        y, predictions, zero_division=0
    )

    macro_f1 = f1_score(
        y,
        predictions,
        average="macro",
        zero_division=0
    )

    loss = log_loss(y, probabilities)

    # =================================================
    # OBSERVATION-LEVEL TOP-1
    # =================================================

    selected = (
        df.assign(predicted_probability=probabilities)
        .sort_values(
            [
                "trip_id",
                "point_idx",
                "predicted_probability",
                "arc_id"
            ],
            ascending=[True, True, False, True]
        )
        .drop_duplicates(
            ["trip_id", "point_idx"]
        )
    )

    top1_all = selected["label"].mean()

    # Observations where a correct candidate exists
    positive_counts = df.groupby(
        ["trip_id", "point_idx"]
    )["label"].sum()

    hit_keys = positive_counts[
        positive_counts > 0
    ].index

    selected_hit = selected.set_index(
        ["trip_id", "point_idx"]
    ).loc[hit_keys]

    top1_hit = selected_hit["label"].mean()

    return {
        "Model": name,
        "Candidate Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Macro F1": macro_f1,
        "Log Loss": loss,
        "Observation Top-1": top1_all,
        "Top-1 on Hit Points": top1_hit,
        "Candidate Rows": len(df),
        "GPS Observations": len(selected)
    }, keys


# =====================================================
# 3. EVALUATE ALL FIVE MODELS
# =====================================================

results = []
reference = None

for name, path in FILES.items():

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Prediction file not found: {path}"
        )

    result, keys = evaluate_model(
        name,
        path,
        reference
    )

    if reference is None:
        reference = keys.copy()

    results.append(result)


# =====================================================
# 4. CREATE COMPARISON TABLE
# =====================================================

comparison = pd.DataFrame(results)

comparison = comparison.sort_values(
    "Observation Top-1",
    ascending=False
)

print("\n")
print("=" * 90)
print("PATHSNAP - FIVE MODEL PERFORMANCE COMPARISON")
print("=" * 90)

print(
    comparison.to_string(
        index=False,
        formatters={
            "Candidate Accuracy": "{:.4%}".format,
            "Precision": "{:.4%}".format,
            "Recall": "{:.4%}".format,
            "Macro F1": "{:.4%}".format,
            "Log Loss": "{:.5f}".format,
            "Observation Top-1": "{:.4%}".format,
            "Top-1 on Hit Points": "{:.4%}".format
        }
    )
)


# =====================================================
# 5. SAVE RESULTS
# =====================================================

os.makedirs("models", exist_ok=True)

comparison.to_csv(
    "models/model_comparison.csv",
    index=False
)

print("\nSaved comparison to:")
print("models/model_comparison.csv")


# =====================================================
# 6. IDENTIFY BEST MODEL
# =====================================================

best_model = comparison.iloc[0]

print("\n========== BEST MODEL ==========")

print("Model:", best_model["Model"])

print(
    "Observation Top-1:",
    f"{best_model['Observation Top-1']:.4%}"
)

print(
    "Candidate Accuracy:",
    f"{best_model['Candidate Accuracy']:.4%}"
)

print("\nCOMPARISON COMPLETE!")
