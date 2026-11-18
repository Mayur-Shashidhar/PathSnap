
import pandas as pd

print("Loading training dataset...")
train = pd.read_csv(
    "data/candidates_train.csv.gz",
    dtype={"trip_id": str}
)

print("Loading test dataset...")
test = pd.read_csv(
    "data/candidates_test.csv",
    dtype={"trip_id": str}
)

print("\n========== DATASET SHAPES ==========")
print("TRAIN:", train.shape)
print("TEST:", test.shape)

print("\n========== COLUMN NAMES ==========")
print(train.columns.tolist())

print("\n========== FIRST 5 ROWS ==========")
print(train.head().to_string())

print("\n========== DATA TYPES ==========")
print(train.dtypes)

print("\n========== MISSING VALUES ==========")
print(train.isnull().sum())

print("\n========== LABEL DISTRIBUTION ==========")
print(train["label"].value_counts(dropna=False))

print("\n========== RELIABILITY DISTRIBUTION ==========")
print(train["reliable"].value_counts(dropna=False))

print("\n========== UNIQUE TRIPS ==========")
print("Training trips:", train["trip_id"].nunique())
print("Test trips:", test["trip_id"].nunique())

print("\n========== OBSERVATIONS ==========")
print("Training GPS points:", train["point_idx"].nunique())
print("Test GPS points:", test["point_idx"].nunique())
