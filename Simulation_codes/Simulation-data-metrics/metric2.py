"""
Bayesian-network intent classifier + evaluation utilities
---------------------------------------------------------
• Loads trial_features4.csv
• Binarises body & gaze angles (≤ 20 deg)
• Trains a simple B → I ← G Bayesian net
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
CSV_PATH = Path("trial_features6.csv")   # change if the CSV lives elsewhere
OUT_DIR  = Path(os.getcwd()) / "outputs_human" # folder for confusion-matrix files
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# 3) Load & preprocess
# ------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
df["B"] = (df.body_angle <= 20.0).astype(int)
df["G"] = (df.gaze_angle <= 15.0).astype(int)
df["I"] = df.ground_truth

# ------------------------------------------------------------------
# 4) Train / test split
# ------------------------------------------------------------------
train, test = train_test_split(
    df, test_size=0.3, stratify=df.I, random_state=42
)

# ------------------------------------------------------------------
# 5) Define & fit Bayesian network
# ------------------------------------------------------------------
model = DiscreteBayesianNetwork([("B", "I"), ("G", "I")])
model.fit(
    train[["B", "G", "I"]],
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
    evidence = {"B": int(row.B), "G": int(row.G)}
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
for w, group in test.groupby("width"):
    print(f"\n--- width = {w:.1f} m ---")
    print(classification_report(group.I, group.pred, target_names=["avoid", "obstruct"]))
    print("accuracy:", accuracy_score(group.I, group.pred))

# ------------------------------------------------------------------
# 8) Confusion-matrix helpers
# ------------------------------------------------------------------
def save_confmat(y_true, y_pred, labels, stem):
    """
    Save a confusion matrix as both PNG heat-map and CSV table.

    Parameters
    ----------
    y_true, y_pred : iterable
        Ground-truth and predicted labels (0/1).
    labels : list[str]
        Label names, e.g. ['avoid', 'obstruct'].
    stem : str
        Base filename without extension, e.g. 'cm_overall'.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # CSV version
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    df_cm.to_csv(OUT_DIR / f"{stem}.csv", index=True)

    # PNG heat-map
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f"{val}", ha="center", va="center")
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
for w, group in test.groupby("width"):
    save_confmat(
        group.I,
        group.pred,
        LABELS,
        stem=f"cm_width_{w:.1f}m",
    )

print(f"\nConfusion matrices saved in: {OUT_DIR.resolve()}")

print("\n--- Conditional-Probability Tables (CPDs) ---")
for cpd in model.get_cpds():
    print(f"\nCPD of {cpd.variable}:")
    print(cpd) 

from sklearn.metrics import log_loss, brier_score_loss

# predicted P(I=1) for every test row
probs = []
for _, row in test.iterrows():
    evidence = {"B": int(row.B), "G": int(row.G)}  # adapt per model
    q = infer.query(["I"], evidence=evidence)
    probs.append(q.values[1])          # P(I=1)

test["prob"] = probs

print("Log-loss  (lower is better):", log_loss(test.I, test.prob))
print("Brier score (lower is better):", brier_score_loss(test.I, test.prob))

