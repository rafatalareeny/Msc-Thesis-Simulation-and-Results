import os, itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    log_loss,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.calibration import calibration_curve
from scipy.stats import ttest_rel

plt.rcParams["figure.autolayout"] = True

CSV_PATH  = Path("Simulation_dataset.csv")
OUT_DIR   = Path.cwd() / "outputs_fused_environment_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED      = 42
ANGLE_CUT = 20.0      
WIDTH_CUT = 2.0       
EPS       = 1e-12     

#load the dataset (make sure to change this part with the real-life"don't mess up the results again"-------------)
df = pd.read_csv(CSV_PATH)

# Discretizing the dataset for the bayesian network 
df["W"] = (df.width      < WIDTH_CUT).astype(int)    
df["B"] = (df.body_angle <= 20).astype(int)   
df["G"] = (df.gaze_angle <= 15).astype(int)   
df["I"] = df.ground_truth                            

#train test slpit the dataset
train, test = train_test_split(df, test_size=0.30, stratify=df.I, random_state=SEED)

#Define & fit Bayesian network
def fit_bn(edges, cols, name):
    m = DiscreteBayesianNetwork(edges)
    m.fit(train[cols], estimator=BayesianEstimator,
          prior_type="BDeu", equivalent_sample_size=10)
    print(f"\n--- CPDs for {name} ---")
    for cpd in m.get_cpds():
        print(f"\nCPD of {cpd.variable}:\n{cpd}")
    return m, VariableElimination(m)

#With environment width
edges_env  = [("W","B"), ("W","G"), ("W","I"), ("B","I"), ("G","I")]
model_env, infer_env = fit_bn(edges_env, ["W","B","G","I"], "WITH-W")

#Without the environment width
edges_base = [("B","I"), ("G","I")]
model_base, infer_base = fit_bn(edges_base, ["B","G","I"], "NO-W")

#Inferences part using the constructed bayesian network 
probs_env, probs_base, preds_env, preds_base = [], [], [], []
for _, r in test.iterrows():
    ev_base = {"B": int(r.B), "G": int(r.G)}
    ev_env  = {"W": int(r.W), **ev_base}

    q_env  = infer_env .query(["I"], evidence=ev_env )
    q_base = infer_base.query(["I"], evidence=ev_base)

    probs_env .append(q_env .values[1])           # P(I=1)
    probs_base.append(q_base.values[1])
    preds_env .append(int(q_env .values.argmax()))
    preds_base.append(int(q_base.values.argmax()))

test = test.copy()
test["prob_env"]   = np.clip(probs_env,  EPS, 1-EPS)
test["prob_base"]  = np.clip(probs_base, EPS, 1-EPS)
test["pred_env"]   = preds_env
test["pred_base"]  = preds_base

#Hard prediction metrics and the confusion matrices for both models (the with environment and without the environment)
def save_confmat(y_true, y_pred, labels, stem):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(OUT_DIR / f"{stem}.csv")
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for (i,j), val in np.ndenumerate(cm):
        ax.text(j, i, str(val), ha="center", va="center")
    ax.set_title(stem.replace("_"," ").title())
    plt.tight_layout(); fig.savefig(OUT_DIR / f"{stem}.png", dpi=300); plt.close(fig)

print("\n///////////Hard Metrics//////////////////")
for tag, pred in [("WITH-W", test.pred_env), ("NO-W", test.pred_base)]:
    print(f"\n[{tag}]")
    print(classification_report(test.I, pred, target_names=["avoid","obstruct"]))
    print("Accuracy:", accuracy_score(test.I, pred))
    save_confmat(test.I, pred, ["avoid","obstruct"], stem=f"cm_{tag.replace('-','')}")

#Probabilistic metrics with one-sided T-tests
def scores(y, p):
    return (log_loss(y, p), brier_score_loss(y, p), roc_auc_score(y, p))

ll_e, br_e, auc_e = scores(test.I, test.prob_env)
ll_b, br_b, auc_b = scores(test.I, test.prob_base)

print("\n//////////Probabilistic Metrics//////////")
print(f"{'':15}  Log-loss   Brier    AUC")
print(f"WITH-W        : {ll_e:8.4f} {br_e:8.4f} {auc_e:6.3f}")
print(f"NO-W          : {ll_b:8.4f} {br_b:8.4f} {auc_b:6.3f}")

#Per-sample losses and paired diffs (with environment and without environment)
p_e = test.prob_env.values
p_b = test.prob_base.values
y   = test.I.values
ll_env   = -(y*np.log(p_e) + (1-y)*np.log(1-p_e))
ll_base  = -(y*np.log(p_b) + (1-y)*np.log(1-p_b))
d_ll     = ll_base - ll_env            
d_brier  = (p_b - y)**2 - (p_e - y)**2 

t_ll, p_ll   = ttest_rel(d_ll,   np.zeros_like(d_ll),   alternative="greater")
t_br, p_br   = ttest_rel(d_brier,np.zeros_like(d_brier),alternative="greater")

print(f"\n(delta) Log-loss mean  (NO-W − WITH-W): {d_ll.mean():.4f}   p = {p_ll:.3g}")
print(f"(delta) Brier   mean  (NO-W − WITH-W): {d_brier.mean():.4f}   p = {p_br:.3g}")

#Save global metrics
pd.DataFrame({
    "model":["WITH-W","NO-W"],
    "logloss":[ll_e,ll_b],
    "brier":[br_e,br_b],
    "auc":[auc_e,auc_b]
}).to_csv(OUT_DIR/"metrics_global.csv", index=False)

#Per width probability metrics (both models)
def per_width_metrics(df_probs, label_col="I", prob_col="prob"):
    pass

rows = []
for w_val, g in test.groupby("width"):
    ll_e_w = log_loss(g.I, g.prob_env)
    ll_b_w = log_loss(g.I, g.prob_base)
    br_e_w = brier_score_loss(g.I, g.prob_env)
    br_b_w = brier_score_loss(g.I, g.prob_base)
    auc_e_w = roc_auc_score(g.I, g.prob_env)
    auc_b_w = roc_auc_score(g.I, g.prob_base)
    rows.append({
        "width": w_val, "n": len(g),
        "logloss_withW": ll_e_w, "logloss_noW": ll_b_w,
        "brier_withW": br_e_w,   "brier_noW": br_b_w,
        "auc_withW": auc_e_w,    "auc_noW": auc_b_w
    })
per_width = pd.DataFrame(rows).sort_values("width")
per_width.to_csv(OUT_DIR/"metrics_per_width.csv", index=False)
print("\n//////////Probabilistic metrics by corridor width (saved to metrics_per_width.csv)///////")
print(per_width.to_string(index=False, float_format="{:.4f}".format))

# ROC overlay for the AUC comparison
fpr_e, tpr_e, _ = roc_curve(test.I, test.prob_env)
fpr_b, tpr_b, _ = roc_curve(test.I, test.prob_base)
plt.figure()
plt.plot(fpr_e, tpr_e, label=f"WITH-W  (AUC={auc_e:.3f})")
plt.plot(fpr_b, tpr_b, label=f"NO-W    (AUC={auc_b:.3f})", linestyle="--")
plt.plot([0,1],[0,1],color="grey",linestyle=":")
plt.xlabel("False-positive rate"); plt.ylabel("True-positive rate")
plt.title("ROC curve"); plt.legend()
plt.savefig(OUT_DIR/"ROC.png", dpi=300); plt.close()


print(f"\nAll figures & CSVs saved in: {OUT_DIR.resolve()}")
