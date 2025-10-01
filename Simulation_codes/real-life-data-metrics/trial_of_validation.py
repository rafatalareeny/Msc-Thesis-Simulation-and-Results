#!/usr/bin/env python3
"""
End-to-end pipeline:
- Train Bayesian networks on simulated data (with & without environment width)
- Evaluate hard & probabilistic metrics on SIM-TEST
- Validate on REAL-LIFE dataset (no refit) using identical evaluation
- Outputs include CPDs, hard metrics (overall + per width W), probabilistic metrics (overall),
  ROC curves (if valid), probability-shift histograms, and a global comparison CSV.
"""

# ------------------------------------------------------------------
# 1) Imports
# ------------------------------------------------------------------
import itertools
from pathlib import Path
from typing import Optional

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
from sklearn.calibration import calibration_curve  # (not used for plotting now)
from scipy.stats import ttest_rel

plt.rcParams["figure.autolayout"] = True

# ------------------------------------------------------------------
# 2) Configuration
# ------------------------------------------------------------------
# Simulated dataset (training + test split)
CSV_PATH_SIM  = Path("trial_features6.csv")
OUT_DIR_SIM   = Path.cwd() / "outputs_full"
OUT_DIR_SIM.mkdir(parents=True, exist_ok=True)

# Real-life dataset (evaluation only; no refit)
CSV_PATH_REAL = Path("real-life-dataset-logs.csv")
OUT_DIR_REAL  = Path.cwd() / "outputs_full_real-life"
OUT_DIR_REAL.mkdir(parents=True, exist_ok=True)

SEED      = 42
# Simulation discretization thresholds
ANGLE_CUT = 20.0      # deg, for sim: body_angle <= 20 => facing
GAZE_CUT  = 15.0      # deg, for sim: gaze_angle <= 15 => looking

# Real-life discretization thresholds
WIDTH_CUT = 2.0       # m (narrow if width < 2.0)
B_MIN, B_MAX = 160.0, 200.0   # deg, for real-life: body facing range
GAZE_SCORE_CUT = 0.80         # for real-life: gaze_score >= 0.8 => looking

EPS       = 1e-12     # numeric safety for logs

# ------------------------------------------------------------------
# 3) Load & preprocess (SIMULATION)
# ------------------------------------------------------------------
df_sim = pd.read_csv(CSV_PATH_SIM)
# Discretize as per your original simulation code
df_sim["W"] = (df_sim.width < WIDTH_CUT).astype(int)    # 1 = narrow
df_sim["B"] = (df_sim.body_angle <= ANGLE_CUT).astype(int)
df_sim["G"] = (df_sim.gaze_angle <= GAZE_CUT).astype(int)   # gaze direction
df_sim["I"] = df_sim.ground_truth.astype(int)               # 0 avoid / 1 obstruct

# Split
train, test = train_test_split(df_sim, test_size=0.30, stratify=df_sim.I, random_state=SEED)

# ------------------------------------------------------------------
# 4) Fit two networks (ON SIMULATION TRAIN)
# ------------------------------------------------------------------
def fit_bn(edges, cols, name):
    m = DiscreteBayesianNetwork(edges)
    m.fit(train[cols], estimator=BayesianEstimator,
          prior_type="BDeu", equivalent_sample_size=10)
    print(f"\n--- CPDs for {name} ---")
    for cpd in m.get_cpds():
        print(f"\nCPD of {cpd.variable}:\n{cpd}")
    return m, VariableElimination(m)

# WITH environment width
edges_env  = [("W","B"), ("W","G"), ("W","I"), ("B","I"), ("G","I")]
model_env, infer_env = fit_bn(edges_env, ["W","B","G","I"], "WITH-W")

# Baseline (NO W)
edges_base = [("B","I"), ("G","I")]
model_base, infer_base = fit_bn(edges_base, ["B","G","I"], "NO-W")

# ------------------------------------------------------------------
# 4.5) Helpers for evaluation on ANY dataset (no refit)
# ------------------------------------------------------------------
def _safe_auc(y, p) -> float:
    # Avoid errors when only one class appears.
    try:
        return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan
    except Exception:
        return np.nan

def _can_plot_roc(y) -> bool:
    return len(np.unique(y)) == 2

def save_confmat(y_true, y_pred, labels, stem, out_dir: Path):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(out_dir / f"{stem}.csv", index=True)
    fig, ax = plt.subplots()
    ax.imshow(cm, interpolation="nearest")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for (i,j), val in np.ndenumerate(cm):
        ax.text(j, i, str(val), ha="center", va="center")
    ax.set_title(stem.replace("_"," ").title())
    plt.tight_layout(); fig.savefig(out_dir / f"{stem}.png", dpi=300); plt.close(fig)

def _save_text(text: str, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def evaluate_dataset(
    df_eval: pd.DataFrame,
    name: str,
    out_dir: Path,
    infer_env: VariableElimination,
    infer_base: VariableElimination,
):
    """
    Runs evaluation on df_eval (no fitting).
    Writes artifacts into out_dir.
    Returns:
        metrics_global (DataFrame): WITH-W vs NO-W (logloss/brier/auc) with dataset column
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n===== EVALUATING DATASET: {name} =====")

    # Inference on df_eval
    probs_env, probs_base, preds_env, preds_base = [], [], [], []
    for _, r in df_eval.iterrows():
        ev_base = {"B": int(r["B"]), "G": int(r["G"])}
        ev_env  = {"W": int(r["W"]), **ev_base}
        q_env   = infer_env.query(["I"], evidence=ev_env)
        q_base  = infer_base.query(["I"], evidence=ev_base)
        probs_env.append(q_env.values[1])
        probs_base.append(q_base.values[1])
        preds_env.append(int(q_env.values.argmax()))
        preds_base.append(int(q_base.values.argmax()))

    eval_df = df_eval.copy()
    eval_df["prob_env"]  = np.clip(probs_env,  EPS, 1-EPS)
    eval_df["prob_base"] = np.clip(probs_base, EPS, 1-EPS)
    eval_df["pred_env"]  = preds_env
    eval_df["pred_base"] = preds_base

    # Save raw predictions
    cols_to_keep = [c for c in ["width","W","B","G","I","prob_env","prob_base","pred_env","pred_base"] if c in eval_df.columns]
    eval_df[cols_to_keep].to_csv(out_dir / "predictions.csv", index=False)

    # ---------------------------
    # Hard metrics (OVERALL)
    # ---------------------------
    print("\n===== HARD METRICS (OVERALL) =====")
    for tag, pred in [("WITH-W", eval_df["pred_env"]), ("NO-W", eval_df["pred_base"])]:
        print(f"\n[{name} | {tag}]")
        try:
            cr = classification_report(eval_df["I"], pred, target_names=["avoid","obstruct"])
        except Exception:
            cr = classification_report(eval_df["I"], pred)
        print(cr)
        acc = accuracy_score(eval_df["I"], pred)
        print("Accuracy:", acc)

        # Save confusion matrix + classification report
        save_confmat(eval_df["I"], pred, ["avoid","obstruct"], stem=f"cm_{tag.replace('-','')}_{name}", out_dir=out_dir)
        _save_text(cr + f"\nAccuracy: {acc:.6f}\n", out_dir / f"classification_report_{tag.replace('-','')}_{name}.txt")

    # ---------------------------
    # Hard metrics (PER WIDTH W)
    # ---------------------------
    print("\n===== HARD METRICS PER CORRIDOR WIDTH (W: 0=wide, 1=narrow) =====")
    per_w_rows = []
    for w_val in [0, 1]:
        gw = eval_df[eval_df["W"] == w_val]
        if gw.empty:
            print(f"\n[W={w_val}] No samples.")
            continue
        print(f"\n[W={w_val}] n={len(gw)}")
        for tag, pred_col in [("WITH-W", "pred_env"), ("NO-W", "pred_base")]:
            y_true = gw["I"]
            y_pred = gw[pred_col]
            try:
                cr_w = classification_report(y_true, y_pred, target_names=["avoid","obstruct"])
            except Exception:
                cr_w = classification_report(y_true, y_pred)
            acc_w = accuracy_score(y_true, y_pred)
            print(f"\n[{name} | {tag} | W={w_val}]")
            print(cr_w)
            print("Accuracy:", acc_w)

            # Save per-width confusion matrix + report
            save_confmat(y_true, y_pred, ["avoid","obstruct"],
                         stem=f"cm_{tag.replace('-','')}_{name}_W{w_val}", out_dir=out_dir)
            _save_text(cr_w + f"\nAccuracy: {acc_w:.6f}\n",
                       out_dir / f"classification_report_{tag.replace('-','')}_{name}_W{w_val}.txt")

            per_w_rows.append({"dataset": name, "W": w_val, "model": tag, "n": len(gw), "accuracy": acc_w})

    if per_w_rows:
        pd.DataFrame(per_w_rows).to_csv(out_dir / "hard_metrics_per_width_W.csv", index=False)

    # ---------------------------
    # Probabilistic metrics (GLOBAL ONLY)
    # ---------------------------
    y   = eval_df["I"].values
    p_e = eval_df["prob_env"].values
    p_b = eval_df["prob_base"].values

    ll_e  = log_loss(y, p_e, labels=[0,1])
    ll_b  = log_loss(y, p_b, labels=[0,1])
    br_e  = brier_score_loss(y, p_e)
    br_b  = brier_score_loss(y, p_b)
    auc_e = _safe_auc(y, p_e)
    auc_b = _safe_auc(y, p_b)

    print("\n===== PROBABILISTIC METRICS (GLOBAL) =====")
    header = f"{'':15}  Log-loss   Brier    AUC"
    row_e  = f"WITH-W ({name}): {ll_e:8.4f} {br_e:8.4f} {auc_e:6.3f}"
    row_b  = f"NO-W   ({name}): {ll_b:8.4f} {br_b:8.4f} {auc_b:6.3f}"
    print(header); print(row_e); print(row_b)

    # paired diffs (NO-W − WITH-W) -> >0 means WITH-W better
    ll_env  = -(y*np.log(p_e) + (1-y)*np.log(1-p_e))
    ll_base = -(y*np.log(p_b) + (1-y)*np.log(1-p_b))
    d_ll    = ll_base - ll_env
    d_br    = (p_b - y)**2 - (p_e - y)**2

    t_ll, p_ll = ttest_rel(d_ll, np.zeros_like(d_ll), alternative="greater")
    t_br, p_br = ttest_rel(d_br, np.zeros_like(d_br), alternative="greater")

    print(f"\nΔLog-loss mean (NO-W − WITH-W): {d_ll.mean():.4f}   p = {p_ll:.3g}")
    print(f"ΔBrier   mean (NO-W − WITH-W): {d_br.mean():.4f}   p = {p_br:.3g}")

    # Save global metrics
    metrics_global = pd.DataFrame({
        "dataset":[name, name],
        "model":["WITH-W","NO-W"],
        "logloss":[ll_e,ll_b],
        "brier":[br_e,br_b],
        "auc":[auc_e,auc_b]
    })
    metrics_global.to_csv(out_dir / "metrics_global.csv", index=False)

    # ---------------------------
    # Plots (ROC if possible, Histogram of probability shift)
    # ---------------------------
    if _can_plot_roc(y):
        fpr_e, tpr_e, _ = roc_curve(y, p_e)
        fpr_b, tpr_b, _ = roc_curve(y, p_b)
        plt.figure()
        plt.plot(fpr_e, tpr_e, label=f"WITH-W  (AUC={auc_e:.3f})")
        plt.plot(fpr_b, tpr_b, label=f"NO-W    (AUC={auc_b:.3f})", linestyle="--")
        plt.plot([0,1],[0,1], color="grey", linestyle=":")
        plt.xlabel("False-positive rate"); plt.ylabel("True-positive rate")
        plt.title(f"ROC curve — {name}"); plt.legend()
        plt.savefig(out_dir / "ROC.png", dpi=300); plt.close()
    else:
        print("ROC curve skipped (only one class present).")

    plt.figure()
    plt.hist(p_e - p_b, bins=30)
    plt.xlabel("P_withW − P_noW"); plt.ylabel("Count")
    plt.title(f"Shift in predicted probability due to environment — {name}")
    plt.savefig(out_dir / "prob_shift_hist.png", dpi=300); plt.close()

    print(f"\nAll figures & CSVs saved in: {out_dir.resolve()}")
    return metrics_global

# ------------------------------------------------------------------
# 5) Evaluate on SIM-TEST using the same function (DRY)
# ------------------------------------------------------------------
metrics_sim = evaluate_dataset(
    df_eval=test.copy(),
    name="SIM-TEST",
    out_dir=OUT_DIR_SIM,
    infer_env=infer_env,
    infer_base=infer_base,
)

# ------------------------------------------------------------------
# 6) REAL-LIFE validation (load, discretize, evaluate — no refit)
# ------------------------------------------------------------------
# Load & discretize REAL-LIFE
df_real_raw = pd.read_csv(CSV_PATH_REAL, sep=";", decimal=",", engine="python")
print("Real-life columns:", df_real_raw.columns.tolist())

df_real = df_real_raw.copy()
df_real["W"] = (df_real["width"] < WIDTH_CUT).astype(int)  # 1 = narrow
df_real["B"] = ((df_real["body_angle"] >= B_MIN) & (df_real["body_angle"] <= B_MAX)).astype(int)
df_real["G"] = (df_real["gaze_score"] >= GAZE_SCORE_CUT).astype(int)  # 1 = looking
df_real["I"] = df_real["ground_truth"].astype(int)

metrics_real = evaluate_dataset(
    df_eval=df_real,
    name="REAL-LIFE",
    out_dir=OUT_DIR_REAL,
    infer_env=infer_env,
    infer_base=infer_base,
)

# ------------------------------------------------------------------
# 7) Combined global metric table (SIM vs REAL)
# ------------------------------------------------------------------
metrics_both = pd.concat([metrics_sim, metrics_real], ignore_index=True)
metrics_both.to_csv(Path.cwd() / "metrics_global__sim_vs_real.csv", index=False)
print("\n=== Global metric comparison saved to metrics_global__sim_vs_real.csv ===")
print(metrics_both.to_string(index=False, float_format="{:.4f}".format))

print(f"\nAll figures & CSVs are in:\n- {OUT_DIR_SIM.resolve()}\n- {OUT_DIR_REAL.resolve()}\n"
      f"Plus combined CSV at project root.")
