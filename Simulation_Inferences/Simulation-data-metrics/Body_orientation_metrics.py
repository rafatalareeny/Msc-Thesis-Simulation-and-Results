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

CSV_PATH = Path("Simulation_dataset.csv")          
OUT_DIR  = Path(os.getcwd()) / "outputs_mono_orientation"  
OUT_DIR.mkdir(parents=True, exist_ok=True)

#load the dataset (make sure to change this part with the real-life"don't mess up the results again"-------------)
df = pd.read_csv(CSV_PATH)

# Discretizing the dataset for the bayesian network 
df["B"] = (df.body_angle <= 20.0).astype(int)   
df["I"] = df.ground_truth

#train test slpit the dataset
train, test = train_test_split(
    df, test_size=0.30, stratify=df.I, random_state=42
)

#Define & fit Bayesian network
model = DiscreteBayesianNetwork([("B", "I")])   

model.fit(
    train[["B", "I"]],
    estimator=BayesianEstimator,
    prior_type="BDeu",
    equivalent_sample_size=10,
)

#Inferences part using the constructed bayesian network 
infer = VariableElimination(model)

preds = []
for _, row in test.iterrows():
    evidence = {"B": int(row.B)}
    q = infer.query(["I"], evidence=evidence)
    preds.append(int(q.values.argmax()))

test["pred"] = preds

#Metrics (overall & per width)
print("\n//////////Overall classification report:////////////")
print(classification_report(test.I, test.pred, target_names=["avoid", "obstruct"]))
print("Overall accuracy:", accuracy_score(test.I, test.pred))

print("\n////////////Metrics by corridor width:////////////////")
for w_val, group in test.groupby("width"):
    print(f"\n--- width = {w_val:.1f} m ---")
    print(classification_report(group.I, group.pred, target_names=["avoid", "obstruct"]))
    print("accuracy:", accuracy_score(group.I, group.pred))

#confusion-matrix generation
def save_confmat(y_true, y_pred, labels, stem):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(
        OUT_DIR / f"{stem}.csv", index=True
    )
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

#confusion matrix generation
LABELS = ["avoid", "obstruct"]

#Overall matrix
save_confmat(test.I, test.pred, LABELS, stem="cm_overall")

#Per corridor width matrix
for w_val, group in test.groupby("width"):
    save_confmat(group.I, group.pred, LABELS, stem=f"cm_width_{w_val:.1f}m")

print(f"\nConfusion matrices saved in: {OUT_DIR.resolve()}")

print("\n--- Conditional-Probability Tables (CPDs) ---")
for cpd in model.get_cpds():
    print(f"\nCPD of {cpd.variable}:")
    print(cpd) 