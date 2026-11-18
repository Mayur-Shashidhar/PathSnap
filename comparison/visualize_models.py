
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------
# PATHS
# ------------------------------------------------

INPUT = "models/model_comparison.csv"
OUTPUT = "comparison/plots"

os.makedirs(OUTPUT, exist_ok=True)

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------

df = pd.read_csv(INPUT)

print("\nAvailable columns:")
print(df.columns.tolist())

# Normalize column names for flexible matching.
def normalize(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())

column_lookup = {
    normalize(col): col for col in df.columns
}

def find_column(*aliases):
    for alias in aliases:
        key = normalize(alias)
        if key in column_lookup:
            return column_lookup[key]
    raise ValueError(
        f"Could not find any of these columns: {aliases}"
    )

model_col = find_column("Model", "model_name")

accuracy_col = find_column(
    "Candidate Accuracy", "candidate_accuracy",
    "accuracy"
)

f1_col = find_column(
    "Macro F1", "macro_f1", "macro_f1_score"
)

top1_col = find_column(
    "Observation Top-1", "observation_top1",
    "top1_accuracy"
)

loss_col = find_column(
    "Log Loss", "log_loss"
)

# Convert metrics to numeric.
metrics = [accuracy_col, f1_col, top1_col, loss_col]

for col in metrics:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=[model_col] + metrics)

if df.empty:
    raise ValueError("No valid model results found.")

models = df[model_col].astype(str)

# Detect whether scores are stored as fractions or percentages.
percentage_columns = [accuracy_col, f1_col, top1_col]

for col in percentage_columns:
    if df[col].max() <= 1.0:
        df[col] *= 100

# ------------------------------------------------
# STYLE
# ------------------------------------------------

plt.rcParams.update({
    "figure.figsize": (11, 6),
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "figure.dpi": 120,
    "savefig.dpi": 300
})

# ------------------------------------------------
# GRAPH 1: PERFORMANCE COMPARISON
# ------------------------------------------------

x = np.arange(len(df))
width = 0.24

fig, ax = plt.subplots(figsize=(12, 7))

ax.bar(
    x - width,
    df[accuracy_col],
    width,
    label="Candidate Accuracy"
)

ax.bar(
    x,
    df[f1_col],
    width,
    label="Macro F1"
)

ax.bar(
    x + width,
    df[top1_col],
    width,
    label="Observation Top-1"
)

ax.set_title("PathSnap: ML Model Performance Comparison")
ax.set_ylabel("Score (%)")
ax.set_xlabel("Machine Learning Model")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15, ha="right")
ax.set_ylim(0, 105)
ax.legend()
ax.grid(axis="y", alpha=0.25)

plt.tight_layout()
plt.savefig(f"{OUTPUT}/01_performance_comparison.png")
plt.close()

# ------------------------------------------------
# GRAPH 2: LOG LOSS
# ------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(models, df[loss_col])

ax.set_title("PathSnap: Model Log Loss Comparison")
ax.set_ylabel("Log Loss (lower is better)")
ax.set_xlabel("Machine Learning Model")
ax.tick_params(axis="x", rotation=15)
ax.grid(axis="y", alpha=0.25)

for bar, value in zip(bars, df[loss_col]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.4f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig(f"{OUTPUT}/02_log_loss_comparison.png")
plt.close()

# ------------------------------------------------
# GRAPH 3: PERFORMANCE HEATMAP
# ------------------------------------------------

heatmap_data = df[
    [accuracy_col, f1_col, top1_col]
].copy()

heatmap_data.index = models

# Add Log Loss as a reversed relative score.
# Lower log loss receives a higher normalized score.
loss = df[loss_col].to_numpy()
if loss.max() != loss.min():
    loss_score = 100 * (
        loss.max() - loss
    ) / (loss.max() - loss.min())
else:
    loss_score = np.full(len(loss), 100.0)

heatmap_data["Log Loss (relative)"] = loss_score

# Normalize percentage metrics if necessary.
for col in [accuracy_col, f1_col, top1_col]:
    if heatmap_data[col].max() <= 1.0:
        heatmap_data[col] *= 100

fig, ax = plt.subplots(figsize=(11, 6))

image = ax.imshow(
    heatmap_data.to_numpy(),
    aspect="auto"
)

ax.set_title("PathSnap: Model Performance Heatmap")
ax.set_yticks(range(len(heatmap_data)))
ax.set_yticklabels(heatmap_data.index)
ax.set_xticks(range(len(heatmap_data.columns)))
ax.set_xticklabels(
    ["Candidate Accuracy", "Macro F1",
     "Observation Top-1", "Log Loss (relative)"],
    rotation=20,
    ha="right"
)

for i in range(heatmap_data.shape[0]):
    for j in range(heatmap_data.shape[1]):
        ax.text(
            j, i,
            f"{heatmap_data.iloc[i, j]:.1f}",
            ha="center",
            va="center"
        )

fig.colorbar(image, ax=ax, label="Relative score")
plt.tight_layout()
plt.savefig(f"{OUTPUT}/03_performance_heatmap.png")
plt.close()

# ------------------------------------------------
# GRAPH 4: OBSERVATION TOP-1 RANKING
# ------------------------------------------------

ranked = df.sort_values(
    top1_col,
    ascending=True
)

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.barh(
    ranked[model_col],
    ranked[top1_col]
)

ax.set_title("PathSnap: Observation-Level Top-1 Ranking")
ax.set_xlabel("Observation Top-1 Accuracy (%)")
ax.set_xlim(0, 100)
ax.grid(axis="x", alpha=0.25)

for bar, value in zip(bars, ranked[top1_col]):
    ax.text(
        value + 0.3,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.2f}%",
        va="center"
    )

plt.tight_layout()
plt.savefig(f"{OUTPUT}/04_observation_top1_ranking.png")
plt.close()

# ------------------------------------------------
# EXPORT CLEAN TABLE
# ------------------------------------------------

summary = df[
    [model_col, accuracy_col, f1_col, loss_col, top1_col]
].copy()

summary.to_csv(
    f"{OUTPUT}/model_comparison_clean.csv",
    index=False
)

print("\n====================================")
print("PATHSNAP VISUALISATION COMPLETE")
print("====================================")

print("\nGenerated files:")
for filename in sorted(os.listdir(OUTPUT)):
    print(filename)

print("\nGraphs saved in:", OUTPUT)
