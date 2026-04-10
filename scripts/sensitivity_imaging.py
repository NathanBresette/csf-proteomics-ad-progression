#!/usr/bin/env python3
"""
Phase 6: Sensitivity -- Imaging (PET Amyloid + FreeSurfer)
============================================================
Repeat Phases 2-4 with PET amyloid + FreeSurfer MRI instead of CSF proteomics.
Cross-platform concordance with CSF model.
Tests whether CSF proteomics adds value beyond what brain imaging provides.
"""

import sys, os, json, time
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr, binomtest

sys.stdout.reconfigure(line_buffering=True)

def tlog(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "results")
DATA = os.path.join(OUT_DIR, "cdrsb_slope_dataset.csv")

GBM_PARAM_GRID = {
    "n_estimators": [300, 500, 800],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.03, 0.05, 0.1],
    "min_samples_leaf": [5, 10, 20],
}

def train_predict_cf(X_train, y_train, X_pred, gbm_params):
    imp = SimpleImputer(strategy="median")
    X_tr = imp.fit_transform(X_train)
    X_pr = imp.transform(X_pred)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    lasso = LassoCV(cv=3, max_iter=10000, random_state=42, n_jobs=-1)
    lasso.fit(X_tr_s, y_train)
    selected = np.where(np.abs(lasso.coef_) > 1e-8)[0]
    if len(selected) < 5:
        selected = np.argsort(np.abs(lasso.coef_))[-10:]
    gbm = GradientBoostingRegressor(subsample=0.8, random_state=42, **gbm_params)
    gbm.fit(X_tr[:, selected], y_train)
    return gbm.predict(X_pr[:, selected])

# -- load -------------------------------------------------------------------
tlog("Phase 6: Sensitivity -- Imaging Cross-Platform Comparison")
df = pd.read_csv(DATA)
with open(os.path.join(OUT_DIR, "column_lists.json")) as f:
    col_info = json.load(f)

CLINICAL = col_info["clinical"]
CSF_COLS = col_info["csf"]
drug_cols = col_info["drug_classes"]

# Note: AV45 and Hippo_norm removed from CLINICAL (zero importance, inconsistent availability)
# This script tests whether adding imaging (AV45, Hippo_norm) on top of the 7-var clinical baseline adds value

tlog(f"  {len(df)} patients, {len(CSF_COLS)} CSF features")
tlog(f"  Clinical features (7): AGE PTEDUCAT APOE4 ADAS13 CDRSB MMSE FAQ")

y = df["slope"].values

# ============================================================================
# Test: CSF-only (no clinical) vs Clinical-only vs Combined
# ============================================================================
tlog("\n========== CSF-ONLY MODEL (no clinical features) ==========")

FEATURE_SETS_SENS = {
    "D_csf_only": CSF_COLS,
}

mask = df["n_visits"] >= 3
y_sub = y[mask]
rids = df.loc[mask, "RID"].values

for fs_name, feats in FEATURE_SETS_SENS.items():
    tlog(f"\n  Model: {fs_name} | {len(feats)} features")
    X_all = df.loc[mask, feats].values

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_pred = np.full(len(y_sub), np.nan)
    fold_r2s = []

    for fold_i, (tr_idx, te_idx) in enumerate(kf.split(np.arange(len(y_sub)))):
        X_tr, X_te = X_all[tr_idx], X_all[te_idx]
        y_tr, y_te = y_sub[tr_idx], y_sub[te_idx]

        imp_obj = SimpleImputer(strategy="median")
        X_tr_imp = imp_obj.fit_transform(X_tr)
        X_te_imp = imp_obj.transform(X_te)

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_imp)

        lasso = LassoCV(cv=3, max_iter=10000, random_state=42, n_jobs=-1)
        lasso.fit(X_tr_s, y_tr)
        selected = np.where(np.abs(lasso.coef_) > 1e-8)[0]
        if len(selected) < 5:
            selected = np.argsort(np.abs(lasso.coef_))[-10:]

        grid = GridSearchCV(
            GradientBoostingRegressor(subsample=0.8, random_state=42),
            GBM_PARAM_GRID, cv=KFold(3, shuffle=True, random_state=42),
            scoring="r2", n_jobs=-1, refit=True
        )
        grid.fit(X_tr_imp[:, selected], y_tr)
        pred = grid.predict(X_te_imp[:, selected])
        oof_pred[te_idx] = pred
        fold_r2s.append(r2_score(y_te, pred))

    valid = ~np.isnan(oof_pred)
    oof_r2 = r2_score(y_sub[valid], oof_pred[valid])
    tlog(f"  {fs_name} OOF R2={oof_r2:.4f} (fold mean={np.mean(fold_r2s):.4f} +/- {np.std(fold_r2s):.4f})")

    # Save CSF-only predictions
    pd.DataFrame({"RID": rids, "y_true": y_sub, "y_pred": oof_pred}).to_csv(
        os.path.join(OUT_DIR, f"cv_predictions_{fs_name}.csv"), index=False)

# ============================================================================
# Responder direction with CSF-only model
# ============================================================================
tlog("\n\n========== RESPONDER DIRECTION (CSF-only) ==========")

# Use simple consensus HP
csf_only_hp = {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.05, "min_samples_leaf": 10}

strat_rows = []
for drug in drug_cols:
    n_users = int(df[drug].sum())
    if n_users < 30:
        continue

    users = df[df[drug] == 1].copy()
    nonusers = df[df[drug] == 0]

    pred_cf = train_predict_cf(nonusers[CSF_COLS].values, nonusers["slope"].values,
                                users[CSF_COLS].values, csf_only_hp)
    te = users["slope"].values - pred_cf

    q25, q75 = np.percentile(te, [25, 75])
    top_q = users.iloc[np.where(te <= q25)[0]]
    bot_q = users.iloc[np.where(te >= q75)[0]]
    direction = top_q["slope"].mean() < bot_q["slope"].mean()
    strat_rows.append({"drug": drug, "direction_correct_csf_only": direction})
    tlog(f"  {drug:25s}  dir={'OK' if direction else 'WRONG'}")

strat_csf = pd.DataFrame(strat_rows)
n_correct = int(strat_csf["direction_correct_csf_only"].sum())
n_total = len(strat_csf)
binom_p = binomtest(n_correct, n_total, 0.5, alternative="greater").pvalue
tlog(f"\n  CSF-only direction: {n_correct}/{n_total} (binomial p={binom_p:.4f})")

# ============================================================================
# Cross-platform concordance: CSF model vs clinical model
# ============================================================================
tlog("\n\n========== CROSS-PLATFORM CONCORDANCE ==========")

csf_file = os.path.join(OUT_DIR, "cv_predictions_B_clinical_csf.csv")
clin_file = os.path.join(OUT_DIR, "cv_predictions_A_clinical.csv")
csf_only_file = os.path.join(OUT_DIR, "cv_predictions_D_csf_only.csv")

files = {"clinical": clin_file, "clinical_csf": csf_file, "csf_only": csf_only_file}
loaded = {}
for name, fpath in files.items():
    if os.path.exists(fpath):
        loaded[name] = pd.read_csv(fpath).dropna(subset=["y_pred"])
        tlog(f"  Loaded {name}: {len(loaded[name])} predictions")

# Pairwise concordance
pairs = [("clinical", "clinical_csf"), ("clinical", "csf_only"), ("csf_only", "clinical_csf")]
for a, b in pairs:
    if a in loaded and b in loaded:
        merged = loaded[a].merge(loaded[b], on="RID", suffixes=(f"_{a}", f"_{b}"))
        if len(merged) > 30:
            r_pred, p_pred = spearmanr(merged[f"y_pred_{a}"], merged[f"y_pred_{b}"])
            tlog(f"  {a} vs {b}: Spearman rho={r_pred:.4f} (p={p_pred:.2e}, n={len(merged)})")

# ============================================================================
# Summary comparison
# ============================================================================
tlog("\n\n========== PLATFORM COMPARISON ==========")
perf_file = os.path.join(OUT_DIR, "model_performance.csv")
if os.path.exists(perf_file):
    perf = pd.read_csv(perf_file)
    for _, row in perf.iterrows():
        tlog(f"  {row['model']:30s}  R2={row['oof_r2']:.4f}")

tlog(f"  {'D_csf_only':30s}  R2={oof_r2:.4f}")

# Direction comparison
nmr_dir_file = os.path.join(OUT_DIR, "responder_stratification.csv")
if os.path.exists(nmr_dir_file):
    main_strat = pd.read_csv(nmr_dir_file)
    n_main = int(main_strat["direction_correct"].sum())
    tlog(f"\n  Direction accuracy:")
    tlog(f"    Clinical+CSF: {n_main}/{len(main_strat)}")
    tlog(f"    CSF-only:     {n_correct}/{n_total}")

# Save
strat_csf.to_csv(os.path.join(OUT_DIR, "responder_direction_csf_only.csv"), index=False)
tlog("Phase 6 DONE.")
