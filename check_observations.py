
import pandas as pd

train = pd.read_csv(
    "data/candidates_train.csv.gz",
    dtype={"trip_id": str}
)

test = pd.read_csv(
    "data/candidates_test.csv",
    dtype={"trip_id": str}
)

GROUP = ["trip_id", "point_idx"]

print("\n========== OBSERVATION COUNTS ==========")

print("Training GPS observations:",
      train[GROUP].drop_duplicates().shape[0])

print("Test GPS observations:",
      test[GROUP].drop_duplicates().shape[0])

print("\n========== RELIABLE DATA ==========")

reliable = train[train["reliable"] == 1].copy()

print("Reliable candidate rows:", len(reliable))
print("Excluded candidate rows:", len(train) - len(reliable))

print("Reliable GPS observations:",
      reliable[GROUP].drop_duplicates().shape[0])

print("Excluded GPS observations:",
      train[train["reliable"] == 0][GROUP].drop_duplicates().shape[0])

print("\n========== POSITIVE CANDIDATES ==========")

positive_counts = (
    reliable.groupby(GROUP)["label"]
    .sum()
)

print("Observations with a positive candidate:",
      (positive_counts > 0).sum())

print("Observations without a positive candidate:",
      (positive_counts == 0).sum())

print("Observations with multiple positive candidates:",
      (positive_counts > 1).sum())

print("\n========== CANDIDATES PER OBSERVATION ==========")

candidate_counts = reliable.groupby(GROUP).size()

print(candidate_counts.describe())

print("\n========== LABEL DISTRIBUTION AFTER FILTERING ==========")

print(reliable["label"].value_counts())
print(reliable["label"].value_counts(normalize=True) * 100)

print("\n========== TRIP DISTRIBUTION ==========")

print("Number of training trips:",
      reliable["trip_id"].nunique())

print("\n========== CHECK COMPLETE ==========")

