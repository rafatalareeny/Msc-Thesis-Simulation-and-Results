#!/usr/bin/env python3
"""
End-to-end pipeline (no body orientation, hard metrics only):
- Train Bayesian networks on simulated data (WITH-W vs NO-W; variables: W, G, I)
- Evaluate HARD METRICS ONLY (overall + per width W) on SIM-TEST
- Validate on REAL-LIFE dataset (no refit) with identical evaluation
- Outputs:
  * CPDs (printed) from SIM training
  * Overall hard metrics & confusion matrices (SIM-TEST and REAL-LIFE) for WITH-W and NO-W
  * Hard metrics per W (W=0 wide, W=1 narrow) with confusion matrices
  * Global hard-metrics comparison CSV (accuracy & F1-macro): metrics_global__sim_vs_real_noB.csv
"""

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
)

plt.rcParams["figure.autolayout"] = True

# ------------------------------------------------------------------
# 1) Configuration
# ------------------------------------------------------------------
# Simulated dataset (training + test split)
CSV_PATH_SIM  = Path("trial_features6.csv")
OUT_DIR_SIM   = Path.cwd() / "outputs_full_noB"
OUT_DIR_SIM.mkdir(parents=True, exist_ok=True)

# Real-life dataset (evaluation only; no refit)
CSV_PATH_REAL = Path("real-life-dataset-logs.csv")
OUT_DIR_REAL  = Path.cwd() / "outputs_full_real-life_noB"
OUT_DIR_REAL.mkdir(parents=True, exist_ok=True)

SEED      = 42
# Simulation discretization thresholds
GAZE_CUT  = 15.0      # deg, for sim: gaze_angle <= 15 => looking

# Real-life discretization thresholds
WIDTH_CUT = 2.0       # m (narrow if width < 2.0)
GAZE_SCORE_CUT = 0.80 # for real-life: gaze_score >= 0.8 => looking

# ------------------------------------------------------------------
# 2) Load & preprocess (SIMULATION)
# ------------------------------------------------------------------
df_sim = pd.read_csv(CSV_PATH_SIM)
# Discretize (NO body angle used)
df_sim["W"] = (df_sim["width"] < WIDTH_CUT).astype(int)      # 1 = narrow
df_sim["G"] = (df_sim["gaze_angle"] <= GAZE_CUT).astype(int) # 1 = looking
df_sim["I"] = df_sim["ground_truth"].astype(int)             # 0 avoid / 1 obstruct

# Split
train, test = train_test_split(df_sim, test_size=0.30, stratify=df_sim.I, random_state=SEED)

# ------------------------------------------------------------------
# 3) Fit two networks (ON SIMULATION TRAIN)
#     WITH-W: W -> G, W -> I, G -> I
#     NO-W:   G -> I
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
edges_env  = [("W","G"), ("W","I"), ("G","I")]
model_env, infer_env = fit_bn(edges_env, ["W","G","I"], "WITH-W")

# Baseline (NO W)
edges_base = [("G","I")]
model_base, infer_base = fit_bn(edges_base, ["G","I"], "NO-W")

# ------------------------------------------------------------------
# 4) Helpers (hard metrics only)
# ------------------------------------------------------------------
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

def evaluate_dataset_hard_only(
    df_eval: pd.DataFrame,
    name: str,
    out_dir: Path,
    infer_env: VariableElimination,
    infer_base: VariableElimination,
):
    """
    HARD METRICS ONLY on df_eval (no fitting).
    Saves:
      - predictions.csv (W,G,I,pred_env,pred_base)
      - classification reports (+ accuracy) overall and per W
      - confusion matrices overall and per W (PNG + CSV)
      - returns a DataFrame with overall hard metrics for global comparison
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n===== EVALUATING DATASET (HARD ONLY): {name} =====")

    # Inference -> predictions only
    preds_env, preds_base = [], []
    for _, r in df_eval.iterrows():
        ev_base = {"G": int(r["G"])}
        ev_env  = {"W": int(r["W"]), **ev_base}

        q_env   = infer_env.query(["I"], evidence=ev_env)
        q_base  = infer_base.query(["I"], evidence=ev_base)

        preds_env.append(int(q_env.values.argmax()))
        preds_base.append(int(q_base.values.argmax()))

    eval_df = df_eval.copy()
    eval_df["pred_env"]  = preds_env
    eval_df["pred_base"] = preds_base

    # Save raw predictions
    cols_to_keep = [c for c in ["width","W","G","I","pred_env","pred_base"] if c in eval_df.columns]
    eval_df[cols_to_keep].to_csv(out_dir / "predictions.csv", index=False)

    # ---------------------------
    # Hard metrics (OVERALL)
    # ---------------------------
    print("\n===== HARD METRICS (OVERALL) =====")
    overall_rows = []
    for tag, pred in [("WITH-W", eval_df["pred_env"]), ("NO-W", eval_df["pred_base"])]:
        print(f"\n[{name} | {tag}]")
        try:
            cr = classification_report(eval_df["I"], pred, target_names=["avoid","obstruct"])
            cr_dict = classification_report(eval_df["I"], pred, target_names=["avoid","obstruct"], output_dict=True)
        except Exception:
            cr = classification_report(eval_df["I"], pred)
            cr_dict = classification_report(eval_df["I"], pred, output_dict=True)
        print(cr)
        acc = accuracy_score(eval_df["I"], pred)
        print("Accuracy:", acc)

        # Save confusion matrix + classification report
        save_confmat(eval_df["I"], pred, ["avoid","obstruct"], stem=f"cm_{tag.replace('-','')}_{name}", out_dir=out_dir)
        _save_text(cr + f"\nAccuracy: {acc:.6f}\n", out_dir / f"classification_report_{tag.replace('-','')}_{name}.txt")

        # Collect overall hard metrics for global CSV
        f1_macro = cr_dict.get("macro avg", {}).get("f1-score", np.nan)
        overall_rows.append({"dataset": name, "model": tag, "accuracy": acc, "f1_macro": f1_macro})

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

    # Save overall hard-metrics table and return for global merge
    overall_df = pd.DataFrame(overall_rows)
    overall_df.to_csv(out_dir / "hard_metrics_global.csv", index=False)

    print(f"\nAll figures & CSVs saved in: {out_dir.resolve()}")
    return overall_df

# ------------------------------------------------------------------
# 5) Evaluate on SIM-TEST (hard metrics only)
# ------------------------------------------------------------------
metrics_sim = evaluate_dataset_hard_only(
    df_eval=test.copy(),
    name="SIM-TEST",
    out_dir=OUT_DIR_SIM,
    infer_env=infer_env,
    infer_base=infer_base,
)

# ------------------------------------------------------------------
# 6) REAL-LIFE validation (load, discretize, evaluate — no refit)
# ------------------------------------------------------------------
# Load & discretize REAL-LIFE (no body angle used)
df_real_raw = pd.read_csv(CSV_PATH_REAL, sep=";", decimal=",", engine="python")
print("Real-life columns:", df_real_raw.columns.tolist())

df_real = df_real_raw.copy()
df_real["W"] = (df_real["width"] < WIDTH_CUT).astype(int)             # 1 = narrow
df_real["G"] = (df_real["gaze_score"] >= GAZE_SCORE_CUT).astype(int)  # 1 = looking
df_real["I"] = df_real["ground_truth"].astype(int)

metrics_real = evaluate_dataset_hard_only(
    df_eval=df_real,
    name="REAL-LIFE",
    out_dir=OUT_DIR_REAL,
    infer_env=infer_env,
    infer_base=infer_base,
)

# ------------------------------------------------------------------
# 7) Combined global HARD-metric table (SIM vs REAL)
# ------------------------------------------------------------------
metrics_both = pd.concat([metrics_sim, metrics_real], ignore_index=True)
# Keep the filename consistent with the "no body" variant
metrics_both.to_csv(Path.cwd() / "metrics_global__sim_vs_real_noB.csv", index=False)
print("\n=== Global hard-metric comparison saved to metrics_global__sim_vs_real_noB.csv ===")
print(metrics_both.to_string(index=False, float_format="{:.4f}".format))

print(f"\nAll figures & CSVs are in:\n- {OUT_DIR_SIM.resolve()}\n- {OUT_DIR_REAL.resolve()}\n"
      f"Plus combined CSV at project root.")
