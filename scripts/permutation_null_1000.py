#!/usr/bin/env python3
"""
Permutation Null — Model B only (clinical + CSF), 1000 permutations.

Speed strategy:
  - Run LassoCV once on real data per fold to get the best alpha
  - All 1000 permutations use that fixed alpha (Lasso, not LassoCV)
  - Permutations parallelized across N_JOBS cores via joblib
  - Estimated wall time with 16 CPUs: ~10-15 minutes
"""

import sys, os, json, time
import numpy as np
import pandas as pd
from datetime import datetime
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from sklearn.linear_model import LassoCV, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

sys.stdout.reconfigure(line_buffering=True)

def tlog(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "results")
DATA = os.path.join(OUT_DIR, "cdrsb_slope_dataset.csv")

N_PERMUTATIONS = 1000
N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))

# ── Step 1: run LassoCV on real data to get per-fold fixed alphas ─────────
def get_fold_alphas(X, y, n_folds=5):
    """Run LassoCV on real data once; return list of (alpha, selected_idx) per fold."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_params = []
    for fold_i, (tr, te) in enumerate(kf.split(X)):
        imp = SimpleImputer(strategy="median")
        X_tr = imp.fit_transform(X[tr])
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        lasso = LassoCV(cv=3, max_iter=10000, random_state=42, n_jobs=1)
        lasso.fit(X_tr_s, y[tr])
        alpha = lasso.alpha_
        sel = np.where(np.abs(lasso.coef_) > 1e-8)[0]
        if len(sel) < 5:
            sel = np.argsort(np.abs(lasso.coef_))[-10:]
        fold_params.append({
            "tr": tr, "te": te,
            "alpha": alpha, "sel": sel,
            "imp": imp, "sc": sc,
        })
        tlog(f"    Fold {fold_i+1}: alpha={alpha:.5f}, features selected={len(sel)}")
    return fold_params

# ── Step 2: run one permutation using fixed alphas ────────────────────────
def run_one_permutation(perm_i, X, y, fold_params):
    rng = np.random.RandomState(perm_i)
    y_perm = y.copy()
    rng.shuffle(y_perm)

    oof = np.full(len(y), np.nan)
    for fp in fold_params:
        tr, te = fp["tr"], fp["te"]
        imp, sc, alpha, sel = fp["imp"], fp["sc"], fp["alpha"], fp["sel"]

        X_tr = imp.transform(X[tr])
        X_te = imp.transform(X[te])
        X_tr_s = sc.transform(X_tr)

        # Fixed alpha — no CV needed
        lasso = Lasso(alpha=alpha, max_iter=10000, random_state=42)
        lasso.fit(X_tr_s, y_perm[tr])
        perm_sel = np.where(np.abs(lasso.coef_) > 1e-8)[0]
        if len(perm_sel) < 5:
            perm_sel = sel  # fall back to real-data selection

        gbm = GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            min_samples_leaf=10, subsample=0.8, random_state=42)
        gbm.fit(X_tr[:, perm_sel], y_perm[tr])
        oof[te] = gbm.predict(X_te[:, perm_sel])

    valid = ~np.isnan(oof)
    return r2_score(y_perm[valid], oof[valid])

# ── Observed R2 (full LassoCV) ────────────────────────────────────────────
def run_observed_cv(X, y, fold_params):
    oof = np.full(len(y), np.nan)
    for fp in fold_params:
        tr, te = fp["tr"], fp["te"]
        imp, sc, sel = fp["imp"], fp["sc"], fp["sel"]
        X_tr = imp.transform(X[tr])
        X_te = imp.transform(X[te])
        gbm = GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            min_samples_leaf=10, subsample=0.8, random_state=42)
        gbm.fit(X_tr[:, sel], y[tr])
        oof[te] = gbm.predict(X_te[:, sel])
    valid = ~np.isnan(oof)
    return r2_score(y[valid], oof[valid])

# ── Main ──────────────────────────────────────────────────────────────────
tlog(f"Permutation Null B-only ({N_PERMUTATIONS} perms, {N_JOBS} workers)")
df = pd.read_csv(DATA)
with open(os.path.join(OUT_DIR, "column_lists.json")) as f:
    col_info = json.load(f)

CLINICAL = col_info["clinical"]
CSF_COLS = col_info["csf"]
y_all = df["slope"].values
mask = df["n_visits"] >= 3
X_b = df.loc[mask, CLINICAL + CSF_COLS].values
y_b = y_all[mask]

# Pre-compute fold alphas once on real data
tlog("Step 1: Running LassoCV on real data to extract per-fold alphas ...")
fold_params = get_fold_alphas(X_b, y_b)

tlog("Step 2: Computing observed R2 with fixed-alpha pipeline ...")
obs_r2 = run_observed_cv(X_b, y_b, fold_params)
tlog(f"  Observed R2 = {obs_r2:.4f}")

# Freeze imputer/scaler transforms so workers don't refit
for fp in fold_params:
    fp["imp"] = fp["imp"]  # already fit
    fp["sc"]  = fp["sc"]   # already fit

tlog(f"Step 3: Running {N_PERMUTATIONS} permutations ({N_JOBS} parallel workers)...")
t0 = time.time()

BATCH = 50  # log every 50 perms
null_r2s = []
for batch_start in range(0, N_PERMUTATIONS, BATCH):
    batch_end = min(batch_start + BATCH, N_PERMUTATIONS)
    batch = Parallel(n_jobs=N_JOBS)(
        delayed(run_one_permutation)(i, X_b, y_b, fold_params)
        for i in range(batch_start, batch_end)
    )
    null_r2s.extend(batch)
    elapsed = time.time() - t0
    rate = elapsed / len(null_r2s)
    remaining = (N_PERMUTATIONS - len(null_r2s)) * rate
    tlog(f"  Perm {len(null_r2s)}/{N_PERMUTATIONS} | "
         f"elapsed={elapsed/60:.1f}min | "
         f"est_remaining={remaining/60:.1f}min | "
         f"null_mean={np.mean(null_r2s):.4f}  null_max={np.max(null_r2s):.4f}")

total = time.time() - t0
tlog(f"All done in {total/60:.1f} min")

# ── Save results ──────────────────────────────────────────────────────────
null = np.array(null_r2s)
p_val = (np.sum(null >= obs_r2) + 1) / (N_PERMUTATIONS + 1)

tlog("\n========== FINAL RESULTS ==========")
tlog(f"  observed={obs_r2:.4f}  null_mean={null.mean():.4f}+/-{null.std():.4f}  "
     f"null_max={null.max():.4f}  p={p_val:.6f}")

result = {
    "model": "B_clinical_csf",
    "observed_r2": obs_r2,
    "null_mean": null.mean(),
    "null_std": null.std(),
    "null_max": null.max(),
    "null_p95": float(np.percentile(null, 95)),
    "empirical_p": p_val,
    "n_permutations": N_PERMUTATIONS,
}

out_path = os.path.join(OUT_DIR, "permutation_null.csv")
if os.path.exists(out_path):
    existing = pd.read_csv(out_path)
    existing = existing[existing["model"] != "B_clinical_csf"]
    updated = pd.concat([existing, pd.DataFrame([result])], ignore_index=True)
else:
    updated = pd.DataFrame([result])
updated.to_csv(out_path, index=False)

pd.DataFrame({"null_r2": null_r2s}).to_csv(
    os.path.join(OUT_DIR, "null_distribution_B_clinical_csf.csv"), index=False)

tlog(f"Saved to {out_path}")
tlog("DONE.")
