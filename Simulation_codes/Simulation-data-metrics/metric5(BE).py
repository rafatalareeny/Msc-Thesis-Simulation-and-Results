"""
Bayesian-network intent classifier (W & B predictors) + evaluation utilities
--------------------------------------------------------------------------
• Loads trial_features4.csv
• Binarises:
      W : corridor width ≤ 2.0 m → 1 (“narrow”) else 0
      B : body angle     ≤ 20 °  → 1 (“facing”) else 0
• Trains network   W → B,  W → I,  B → I
• Prints overall and per-corridor-width metrics
• Saves confusion matrices (PNG + CSV) in OUT_DIR
"""

# ------------------------------------------------------------------
# 1) Imports
# ------------------------------------------------------------------
import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
)

# ------------------------------------------------------------------
# 2) Configuration
# ------------------------------------------------------------------
CSV_PATH = Path("trial_features6.csv")          # change if needed
OUT_DIR  = Path(os.getcwd()) / "outputs_WB"     # separate folder
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# 3) Load & preprocess
# ------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

df["W"] = (df.width      < 2.0 ).astype(int)   # environment: narrow?
df["B"] = (df.body_angle <= 20.0).astype(int)   # body orientation
df["I"] = df.ground_truth                        # intent (0/1)

# ------------------------------------------------------------------
# 4) Train / test split
# ------------------------------------------------------------------
train, test = train_test_split(
    df, test_size=0.30, stratify=df.I, random_state=42
)

# ------------------------------------------------------------------
# 5) Define & fit Bayesian network
# ------------------------------------------------------------------
model = DiscreteBayesianNetwork(
    [("W", "B"), ("W", "I"), ("B", "I")]         # W influences B & I; B influences I
)

model.fit(
    train[["W", "B", "I"]],
    estimator=BayesianEstimator,
    prior_type="BDeu",
    equivalent_sample_size=10,
)

# ------------------------------------------------------------------
# 6) Inference
# ------------------------------------------------------------------
infer = VariableElimination(model)

preds = []
for _, row in test.iterrows():
    evidence = {
        "W": int(row.W),
        "B": int(row.B),
    }
    q = infer.query(["I"], evidence=evidence)
    preds.append(int(q.values.argmax()))

test["pred"] = preds

# ------------------------------------------------------------------
# 7) Metrics (overall & per width)
# ------------------------------------------------------------------
print("\nOverall classification report:")
print(classification_report(test.I, test.pred, target_names=["avoid", "obstruct"]))
print("Overall accuracy:", accuracy_score(test.I, test.pred))

print("\nMetrics by corridor width:")
for w_val, group in test.groupby("width"):
    print(f"\n--- width = {w_val:.1f} m ---")
    print(classification_report(group.I, group.pred, target_names=["avoid", "obstruct"]))
    print("accuracy:", accuracy_score(group.I, group.pred))

# ------------------------------------------------------------------
# 8) Confusion-matrix helpers
# ------------------------------------------------------------------
def save_confmat(y_true, y_pred, labels, stem):
    """Save confusion matrix as PNG + CSV."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # CSV
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(
        OUT_DIR / f"{stem}.csv", index=True
    )

    # PNG heat-map
    fig, ax = plt.subplots()
    ax.imshow(cm, interpolation="nearest")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, str(val), ha="center", va="center")
    plt.title(stem.replace("_", " ").title())
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=300)
    plt.close(fig)

# ------------------------------------------------------------------
# 9) Generate & save confusion matrices
# ------------------------------------------------------------------
LABELS = ["avoid", "obstruct"]

# 9-a) Overall
save_confmat(test.I, test.pred, LABELS, stem="cm_overall")

# 9-b) Per corridor width
for w_val, group in test.groupby("width"):
    save_confmat(group.I, group.pred, LABELS, stem=f"cm_width_{w_val:.1f}m")

print(f"\nConfusion matrices saved in: {OUT_DIR.resolve()}")

print("\n--- Conditional-Probability Tables (CPDs) ---")
for cpd in model.get_cpds():
    print(f"\nCPD of {cpd.variable}:")
    print(cpd) 