#!/usr/bin/env python3
"""
Non-Circular Drug Responder Stratification (CSF Proteomics)
============================================================
All tests avoid the circularity of TE = actual - predicted then checking
if top-TE patients have lower actual slopes (which is tautological).

Tests (all non-circular):
  A. OOF counterfactual predictions (train on non-users in training folds,
     predict users in test folds — no leakage)
  B. Population drug effect: mean(actual) vs mean(predicted_cf) per drug.
     predicted_cf is trained ONLY on non-users, so comparing to users' actual
     outcomes is an honest out-of-sample test.
  C. Calibration slope: regress actual ~ predicted_cf separately for users
     and non-users. If drug works, users' slope < non-users' slope (drug
     attenuates the predicted natural history).
  D. Risk-stratified benefit: stratify users by predicted_cf (function of
     baseline X only, independent of actual outcome). Check if high-risk
     stratum shows larger gap between predicted and actual.
  E. Independent outcome validation: predicted_cf from CDRSB model correlates
     with ADAS13/MMSE slopes (outcomes the model never saw).
"""

import sys, os, json, time
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, ttest_1samp, binomtest

sys.stdout.reconfigure(line_buffering=True)

def tlog(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "results")
DATA = os.path.join(OUT_DIR, "cdrsb_slope_dataset.csv")

GBM_PARAMS = {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.05,
              "min_samples_leaf": 10, "subsample": 0.8, "random_state": 42}


def train_predict(X_train, y_train, X_pred):
    """LASSO feature selection + GBM, return predictions."""
    imp = SimpleImputer(strategy="median")
    X_tr = imp.fit_transform(X_train)
    X_pr = imp.transform(X_pred)
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    lasso = LassoCV(cv=3, max_iter=10000, random_state=42, n_jobs=-1)
    lasso.fit(X_tr_s, y_train)
    sel = np.where(np.abs(lasso.coef_) > 1e-8)[0]
    if len(sel) < 5:
        sel = np.argsort(np.abs(lasso.coef_))[-10:]
    gbm = GradientBoostingRegressor(**GBM_PARAMS)
    gbm.fit(X_tr[:, sel], y_train)
    return gbm.predict(X_pr[:, sel])


# -- load -------------------------------------------------------------------
tlog("Non-Circular Drug Responder Stratification")
df = pd.read_csv(DATA)
with open(os.path.join(OUT_DIR, "column_lists.json")) as f:
    col_info = json.load(f)

CLINICAL = col_info["clinical"]
CSF_COLS = col_info["csf"]
drug_cols = col_info["drug_classes"]
ALL_FEATS = CLINICAL + CSF_COLS

tlog(f"  {len(df)} patients, {len(ALL_FEATS)} features, {len(drug_cols)} drug classes")

has_adas = "adas13_slope" in df.columns
has_mmse = "mmse_slope" in df.columns
tlog(f"  Independent outcomes: ADAS13 slope={'yes' if has_adas else 'no'}, "
     f"MMSE slope={'yes' if has_mmse else 'no'}")

# ============================================================================
# Per-drug analysis
# ============================================================================
drug_results = []

for drug in drug_cols:
    n_users = int(df[drug].sum())
    if n_users < 20:
        tlog(f"\n  {drug}: skipped (n={n_users} < 20)")
        continue

    t0 = time.time()
    users_mask = df[drug] == 1
    nonusers_mask = df[drug] == 0
    users_idx = np.where(users_mask)[0]
    nonusers_idx = np.where(nonusers_mask)[0]

    # ==== TEST A: OOF counterfactual predictions ====
    # 5-fold CV: train on non-users in training folds, predict users+non-users in test folds
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    predicted_cf = np.full(len(df), np.nan)

    for fold_i, (train_all, test_all) in enumerate(kf.split(np.arange(len(df)))):
        train_nonusers = np.intersect1d(train_all, nonusers_idx)
        test_users = np.intersect1d(test_all, users_idx)
        test_nonusers = np.intersect1d(test_all, nonusers_idx)

        if len(train_nonusers) < 20:
            continue

        X_train = df.iloc[train_nonusers][ALL_FEATS].values
        y_train = df.iloc[train_nonusers]["slope"].values

        # Predict for test users
        if len(test_users) > 0:
            X_test_u = df.iloc[test_users][ALL_FEATS].values
            predicted_cf[test_users] = train_predict(X_train, y_train, X_test_u)

        # Predict for test non-users (for calibration comparison)
        if len(test_nonusers) > 0:
            X_test_nu = df.iloc[test_nonusers][ALL_FEATS].values
            predicted_cf[test_nonusers] = train_predict(X_train, y_train, X_test_nu)

    # Gather valid predictions
    user_valid = users_mask & ~np.isnan(predicted_cf)
    nonuser_valid = nonusers_mask & ~np.isnan(predicted_cf)
    n_user_valid = int(user_valid.sum())
    n_nonuser_valid = int(nonuser_valid.sum())

    if n_user_valid < 20:
        tlog(f"\n  {drug}: insufficient OOF predictions ({n_user_valid})")
        continue

    pred_cf_users = predicted_cf[user_valid]
    actual_users = df.loc[user_valid, "slope"].values
    pred_cf_nonusers = predicted_cf[nonuser_valid]
    actual_nonusers = df.loc[nonuser_valid, "slope"].values

    tlog(f"\n{'='*60}")
    tlog(f"{drug} | {n_user_valid} users, {n_nonuser_valid} non-users")

    # ==== TEST B: Population drug effect (non-circular) ====
    # predicted_cf trained on non-users only; actual is users' real outcome
    delta = actual_users - pred_cf_users  # negative = drug beneficial
    mean_delta = delta.mean()
    t_stat, t_p = ttest_1samp(delta, 0)
    beneficial = mean_delta < 0
    tlog(f"  B. Population effect: mean(actual - predicted_cf) = {mean_delta:+.4f} "
         f"(t={t_stat:.2f}, p={t_p:.4f}) [{'BENEFICIAL' if beneficial else 'NULL/HARMFUL'}]")

    # ==== TEST C: Calibration slope (non-circular) ====
    # Non-users: actual ~ predicted should be well-calibrated (slope ~1)
    # Users: if drug attenuates decline, slope should be < 1
    slope_nu = np.polyfit(pred_cf_nonusers, actual_nonusers, 1)[0] if n_nonuser_valid > 20 else np.nan
    slope_u = np.polyfit(pred_cf_users, actual_users, 1)[0]
    r_nu = pearsonr(pred_cf_nonusers, actual_nonusers)[0] if n_nonuser_valid > 20 else np.nan
    r_u = pearsonr(pred_cf_users, actual_users)[0]
    slope_atten = slope_u - slope_nu if not np.isnan(slope_nu) else np.nan
    tlog(f"  C. Calibration slope: non-users={slope_nu:.3f} (r={r_nu:.3f}), "
         f"users={slope_u:.3f} (r={r_u:.3f}), attenuation={slope_atten:+.3f}")

    # ==== TEST D: Risk-stratified benefit (non-circular) ====
    # Stratify by predicted_cf (function of baseline X only, NOT TE)
    tertile_cuts = np.percentile(pred_cf_users, [33.3, 66.7])
    strata = {
        "low_risk":  pred_cf_users <= tertile_cuts[0],
        "mid_risk":  (pred_cf_users > tertile_cuts[0]) & (pred_cf_users <= tertile_cuts[1]),
        "high_risk": pred_cf_users > tertile_cuts[1],
    }
    strata_benefits = {}
    for label, smask in strata.items():
        if smask.sum() > 5:
            benefit = (actual_users[smask] - pred_cf_users[smask]).mean()
            strata_benefits[label] = benefit
            tlog(f"  D. {label:10s} (n={smask.sum():3d}): "
                 f"mean(actual - predicted_cf) = {benefit:+.4f}")
        else:
            strata_benefits[label] = np.nan

    # Trend: drug benefits high-risk more if their gap is more negative
    high_risk_more = None
    if not np.isnan(strata_benefits.get("high_risk", np.nan)) and \
       not np.isnan(strata_benefits.get("low_risk", np.nan)):
        high_risk_more = strata_benefits["high_risk"] < strata_benefits["low_risk"]
        tlog(f"  D. High-risk benefits more: {high_risk_more}")

    # ==== TEST D2: Difference-in-differences (more robust) ====
    # Within each predicted-trajectory tertile, compare actual slopes of
    # users vs non-users directly.  Non-users in the same stratum serve as
    # the real counterfactual, controlling for regression-to-the-mean.
    all_valid_mask = ~np.isnan(predicted_cf)
    tertile_cuts_global = np.percentile(predicted_cf[all_valid_mask], [33.3, 66.7])

    def _assign_tertile(arr):
        out = np.full(len(arr), "mid", dtype=object)
        out[arr <= tertile_cuts_global[0]] = "low"
        out[arr >  tertile_cuts_global[1]] = "high"
        return out

    pred_tertiles = _assign_tertile(predicted_cf)
    did_results = {}
    for stratum in ["low", "mid", "high"]:
        s_mask    = pred_tertiles == stratum
        u_in_s    = user_valid    & s_mask
        nu_in_s   = nonuser_valid & s_mask
        n_u, n_nu = int(u_in_s.sum()), int(nu_in_s.sum())
        if n_u < 5 or n_nu < 5:
            did_results[stratum] = np.nan
            tlog(f"  D2. {stratum:5s}: too few (n_users={n_u}, n_nonusers={n_nu})")
            continue
        effect = (df.loc[u_in_s,  "slope"].values.mean() -
                  df.loc[nu_in_s, "slope"].values.mean())
        did_results[stratum] = effect
        tlog(f"  D2. DiD {stratum:5s}: n_u={n_u}, n_nu={n_nu}, "
             f"effect = {effect:+.4f} "
             f"({'BENEFICIAL' if effect < 0 else 'null/harmful'})")

    did_interaction   = np.nan
    did_high_benefits = None
    if not np.isnan(did_results.get("high", np.nan)) and \
       not np.isnan(did_results.get("low",  np.nan)):
        did_interaction   = did_results["high"] - did_results["low"]
        did_high_benefits = did_results["high"] < did_results["low"]
        tlog(f"  D2. DiD interaction (high−low) = {did_interaction:+.4f}  "
             f"[{'HIGH-RISK benefits more ✓' if did_high_benefits else 'LOW-RISK benefits more'}]")

    # ==== TEST E: Independent outcome validation (non-circular) ====
    # predicted_cf from CDRSB model → does it correlate with ADAS13/MMSE slopes?
    user_df = df.loc[user_valid]
    adas_r, adas_p, mmse_r, mmse_p = np.nan, np.nan, np.nan, np.nan

    if has_adas:
        av = user_df["adas13_slope"].values
        am = ~np.isnan(av)
        if am.sum() > 20:
            adas_r, adas_p = pearsonr(pred_cf_users[am], av[am])
            tlog(f"  E. ADAS13 slope vs predicted_cf: r={adas_r:.3f} "
                 f"(p={adas_p:.4f}, n={am.sum()})")

    if has_mmse:
        mv = user_df["mmse_slope"].values
        mm = ~np.isnan(mv)
        if mm.sum() > 20:
            mmse_r, mmse_p = pearsonr(pred_cf_users[mm], mv[mm])
            tlog(f"  E. MMSE slope vs predicted_cf: r={mmse_r:.3f} "
                 f"(p={mmse_p:.4f}, n={mm.sum()})")

    elapsed = time.time() - t0
    tlog(f"  [{elapsed:.1f}s]")

    drug_results.append({
        "drug": drug,
        "n_users": n_user_valid,
        "n_nonusers": n_nonuser_valid,
        # B: Population effect
        "mean_delta": mean_delta,
        "delta_t": t_stat,
        "delta_p": t_p,
        "direction_beneficial": beneficial,
        # C: Calibration
        "calib_slope_nonusers": slope_nu,
        "calib_slope_users": slope_u,
        "calib_attenuation": slope_atten,
        "r_nonusers": r_nu,
        "r_users": r_u,
        # D: Risk-stratified (within-user, vs predicted counterfactual)
        "benefit_low_risk": strata_benefits.get("low_risk", np.nan),
        "benefit_mid_risk": strata_benefits.get("mid_risk", np.nan),
        "benefit_high_risk": strata_benefits.get("high_risk", np.nan),
        "high_risk_benefits_more": high_risk_more,
        # D2: Difference-in-differences (users vs non-users within tertile)
        "did_effect_low":       did_results.get("low",  np.nan),
        "did_effect_mid":       did_results.get("mid",  np.nan),
        "did_effect_high":      did_results.get("high", np.nan),
        "did_interaction":      did_interaction,
        "did_high_benefits_more": did_high_benefits,
        # E: Independent outcomes
        "adas13_r": adas_r,
        "adas13_p": adas_p,
        "mmse_r": mmse_r,
        "mmse_p": mmse_p,
    })


# ============================================================================
# SUMMARY
# ============================================================================
tlog("\n\n========== SUMMARY ==========")
res_df = pd.DataFrame(drug_results)

if len(res_df) > 0:
    n_beneficial = int(res_df["direction_beneficial"].sum())
    n_total = len(res_df)
    binom_p = binomtest(n_beneficial, n_total, 0.5, alternative="greater").pvalue
    tlog(f"\n  B. Direction test: {n_beneficial}/{n_total} drugs beneficial "
         f"(binomial p={binom_p:.4f})")

    mean_atten = res_df["calib_attenuation"].mean()
    tlog(f"  C. Mean calibration attenuation: {mean_atten:+.3f} "
         f"(negative = drug attenuates predicted decline)")

    n_trend = res_df["high_risk_benefits_more"].dropna().sum()
    n_trend_total = res_df["high_risk_benefits_more"].dropna().shape[0]
    tlog(f"  D.  Risk-stratified (vs CF):  {n_trend}/{n_trend_total} drugs show "
         f"high-risk patients benefit more")

    n_did = res_df["did_high_benefits_more"].dropna().sum()
    n_did_total = res_df["did_high_benefits_more"].dropna().shape[0]
    did_binom_p = binomtest(int(n_did), n_did_total, 0.5, alternative="greater").pvalue
    tlog(f"  D2. DiD (users vs non-users): {n_did}/{n_did_total} drugs show "
         f"high-risk patients benefit more (binomial p={did_binom_p:.4f})")

    if has_adas:
        valid_adas = res_df["adas13_r"].notna()
        if valid_adas.sum() > 0:
            tlog(f"  E. ADAS13 cross-validation: mean r={res_df.loc[valid_adas, 'adas13_r'].mean():.3f}")

    tlog("\n  Per-drug detail:")
    for _, row in res_df.iterrows():
        sig = "*" if row["delta_p"] < 0.05 else " "
        d = "+" if row["direction_beneficial"] else "-"
        tlog(f"    {sig}{d} {row['drug']:25s}  delta={row['mean_delta']:+.4f}  "
             f"p={row['delta_p']:.4f}  calib_slope={row['calib_slope_users']:.3f}  "
             f"atten={row['calib_attenuation']:+.3f}")

res_df.to_csv(os.path.join(OUT_DIR, "responder_stratification.csv"), index=False)
tlog("Saved responder_stratification.csv")
tlog("Responder stratification DONE.")
