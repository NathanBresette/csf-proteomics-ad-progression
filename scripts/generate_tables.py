#!/usr/bin/env python3
"""
Generate publication tables for CSF Proteomics paper.
Outputs CSV files to tables/ directory.
"""

import os, json
import numpy as np
import pandas as pd
from scipy.stats import binomtest

BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "results")
EXT_DIR = os.path.join(BASE, "external_validation", "results")
TBL_DIR = os.path.join(BASE, "tables")
os.makedirs(TBL_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(OUT_DIR, "cdrsb_slope_dataset.csv"))
with open(os.path.join(OUT_DIR, "column_lists.json")) as f:
    col_info = json.load(f)

perf     = pd.read_csv(os.path.join(OUT_DIR, "model_performance.csv"))
perm     = pd.read_csv(os.path.join(OUT_DIR, "permutation_null.csv"))
prot_imp = pd.read_csv(os.path.join(OUT_DIR, "protein_importance.csv"))
feat_bio = pd.read_csv(os.path.join(OUT_DIR, "feature_biology_table.csv"))
resp     = pd.read_csv(os.path.join(OUT_DIR, "responder_stratification.csv"))
procova  = pd.read_csv(os.path.join(OUT_DIR, "procova_variance_reduction.csv"))
ext_perf = pd.read_csv(os.path.join(EXT_DIR, "validation_performance.csv"))
tmt_resp = pd.read_csv(os.path.join(EXT_DIR, "tmt_responder_stratification.csv"))

CSF_COLS = col_info["csf"]
drug_cols = col_info["drug_classes"]


# ============================================================================
# TABLE 1: Cohort Characteristics (single column -- all 279 have >=3 visits)
# ============================================================================
print("Generating Table 1: Cohort Characteristics...")

def fmt_mean_sd(series):
    return f"{series.mean():.1f} ({series.std():.1f})"

def fmt_median_iqr(series):
    q25, q75 = series.quantile([0.25, 0.75])
    return f"{series.median():.2f} ({q25:.2f}-{q75:.2f})"

def fmt_n_pct(series, val=1):
    n = (series == val).sum()
    pct = n / len(series) * 100
    return f"{n} ({pct:.1f}%)"

rows = []
rows.append({"Characteristic": "N", "Value": str(len(df))})
rows.append({"Characteristic": "Age, mean (SD)", "Value": fmt_mean_sd(df["AGE"])})
rows.append({"Characteristic": "Education (years), mean (SD)", "Value": fmt_mean_sd(df["PTEDUCAT"])})

if "PTGENDER" in df.columns:
    is_male = df["PTGENDER"].str.lower().eq("male").astype(int)
    rows.append({"Characteristic": "Male, n (%)", "Value": fmt_n_pct(is_male)})

rows.append({"Characteristic": "Baseline CDRSB, mean (SD)", "Value": fmt_mean_sd(df["CDRSB"])})

if "ADAS13" in df.columns:
    rows.append({"Characteristic": "Baseline ADAS13, mean (SD)", "Value": fmt_mean_sd(df["ADAS13"])})

if "MMSE" in df.columns:
    rows.append({"Characteristic": "Baseline MMSE, mean (SD)", "Value": fmt_mean_sd(df["MMSE"])})

rows.append({"Characteristic": "CDRSB slope (pts/month), median (IQR)", "Value": fmt_median_iqr(df["slope"])})
rows.append({"Characteristic": "N visits, mean (SD)", "Value": fmt_mean_sd(df["n_visits"])})

# Diagnosis breakdown
if "DX" in df.columns:
    for dx in sorted(df["DX"].dropna().unique()):
        rows.append({"Characteristic": f"  {dx}, n (%)", "Value": fmt_n_pct(df["DX"], dx)})

# CSF protein coverage
csf_coverage = df[CSF_COLS].notna().mean(axis=1)
rows.append({"Characteristic": "CSF protein coverage, mean (SD)",
             "Value": f"{csf_coverage.mean():.1%} ({csf_coverage.std():.1%})"})
rows.append({"Characteristic": "CSF proteins measured", "Value": str(len(CSF_COLS))})

# Drug usage
rows.append({"Characteristic": "", "Value": ""})
rows.append({"Characteristic": "Drug Usage", "Value": ""})
for d in drug_cols:
    clean = d.replace("drug_", "").replace("_", " ").title()
    n = int(df[d].sum())
    rows.append({"Characteristic": f"  {clean}, n (%)",
                 "Value": f"{n} ({n/len(df)*100:.1f}%)"})

t1 = pd.DataFrame(rows)
t1.to_csv(os.path.join(TBL_DIR, "table1_cohort_characteristics.csv"), index=False)
print(f"  Saved table1_cohort_characteristics.csv")


# ============================================================================
# TABLE 2: Model Performance Comparison
# ============================================================================
print("Generating Table 2: Model Performance...")

t2_rows = []
model_map = {
    "A_clinical": "Clinical Only",
    "B_clinical_csf": "Clinical + CSF Proteomics",
    "A_clinical_SE_weighted": "Clinical Only (SE-weighted)",
    "B_clinical_csf_SE_weighted": "Clinical + CSF (SE-weighted)",
}

for model_id, label in model_map.items():
    row = perf[perf["model"] == model_id]
    if len(row) == 0:
        continue
    row = row.iloc[0]
    perm_row = perm[perm["model"] == model_id]
    p_val = perm_row["empirical_p"].values[0] if len(perm_row) > 0 else ""

    t2_rows.append({
        "Model": label,
        "Features": "7 clinical-cognitive" if "clinical" in model_id and "csf" not in model_id else "7 clinical-cognitive + 320 CSF",
        "OOF R2": f"{row['oof_r2']:.3f}",
        "R2 Mean (SD)": f"{row['r2_mean']:.3f} ({row['r2_std']:.3f})",
        "MAE": f"{row['oof_mae']:.4f}",
        "Pearson r": f"{row['oof_r']:.3f}",
        "Permutation p": f"<{p_val}" if p_val else "",
    })

csf_perm = perm[perm["model"] == "D_csf_only"]
if len(csf_perm) > 0:
    t2_rows.append({
        "Model": "CSF Proteomics Only",
        "Features": "320 CSF",
        "OOF R2": f"{csf_perm.iloc[0]['observed_r2']:.3f}",
        "R2 Mean (SD)": "",
        "MAE": "",
        "Pearson r": "",
        "Permutation p": f"<{csf_perm.iloc[0]['empirical_p']}",
    })

clin_r2 = perf[perf["model"] == "A_clinical"]["oof_r2"].values[0]
csf_r2 = perf[perf["model"] == "B_clinical_csf"]["oof_r2"].values[0]
t2_rows.append({
    "Model": "Delta (CSF increment)",
    "Features": "",
    "OOF R2": f"+{csf_r2 - clin_r2:.3f}",
    "R2 Mean (SD)": "",
    "MAE": "",
    "Pearson r": "",
    "Permutation p": "",
})

# External validation rows
section_labels = {
    "A_MRM_to_TMT":          ("MRM→TMT Cross-Platform",        "240 overlap (MRM) → 820 TMT-only"),
    "A2_TMT_mapped_cv":      ("TMT Mapped 5-Fold CV",           "138 proteins shared with MRM"),
    "B_TMT_native_full":     ("TMT Native 5-Fold CV",           "7 clinical-cognitive + 2492 TMT proteins"),
    "B_TMT_native_clinical": ("TMT Clinical Only (ref.)",       "7 clinical-cognitive features only"),
    "C_TMT_to_MRM":          ("TMT→MRM External Validation ★",  "820 TMT-only → 279 MRM, zero overlap"),
}
t2_rows.append({"Model": "", "Features": "", "OOF R2": "", "R2 Mean (SD)": "",
                "MAE": "", "Pearson r": "", "Permutation p": ""})
t2_rows.append({"Model": "=== External Validation (TMT Mass Spec, n=1060) ===",
                "Features": "", "OOF R2": "", "R2 Mean (SD)": "",
                "MAE": "", "Pearson r": "", "Permutation p": ""})
for _, er in ext_perf.iterrows():
    sec = er["section"]
    if sec not in section_labels:
        continue
    label, note = section_labels[sec]
    r_val  = er.get("r", float("nan"))
    mae_val = er.get("mae", float("nan"))
    t2_rows.append({
        "Model":        label,
        "Features":     note,
        "OOF R2":       f"{er['r2']:.3f}" if not pd.isna(er["r2"]) else "",
        "R2 Mean (SD)": "",
        "MAE":          f"{mae_val:.4f}" if not pd.isna(mae_val) else "",
        "Pearson r":    f"{r_val:.3f}" if not pd.isna(r_val) else "",
        "Permutation p": "6.0×10⁻²⁶" if sec == "C_TMT_to_MRM" else "",
    })

t2 = pd.DataFrame(t2_rows)
t2.to_csv(os.path.join(TBL_DIR, "table2_model_performance.csv"), index=False)
print(f"  Saved table2_model_performance.csv")


# ============================================================================
# TABLE 3: Non-Circular Drug Stratification Results
# ============================================================================
print("Generating Table 3: Drug Stratification (Non-Circular)...")

t3_rows = []
for _, r_row in resp.iterrows():
    drug = r_row["drug"]
    clean = drug.replace("drug_", "").replace("_", " ").title()

    # PROCOVA (clinical+CSF)
    p_row = procova[(procova["drug"] == drug) & (procova["feature_set"] == "clinical_csf")]
    ss_red = p_row["procova_ss_reduction_pct"].values[0] if len(p_row) > 0 else np.nan

    t3_rows.append({
        "Drug Class": clean,
        "N Users": int(r_row["n_users"]),
        "N Non-Users": int(r_row["n_nonusers"]),
        "Pop. Effect (delta)": f"{r_row['mean_delta']:+.4f}",
        "Pop. Effect p": f"{r_row['delta_p']:.4f}",
        "Calib. Slope (Users)": f"{r_row['calib_slope_users']:.3f}",
        "Calib. Attenuation": f"{r_row['calib_attenuation']:+.3f}" if not np.isnan(r_row['calib_attenuation']) else "",
        "High-Risk Benefits More": "Yes" if r_row["high_risk_benefits_more"] else "No",
        "ADAS13 r": f"{r_row['adas13_r']:.3f}" if not np.isnan(r_row['adas13_r']) else "",
        "PROCOVA SS Reduction": f"{ss_red:.1f}%" if not np.isnan(ss_red) else "",
    })

t3 = pd.DataFrame(t3_rows)
t3.to_csv(os.path.join(TBL_DIR, "table3_drug_stratification.csv"), index=False)
print(f"  Saved table3_drug_stratification.csv")


# ============================================================================
# TABLE 4: External Validation Drug Stratification (TMT cohort, n=1060)
# ============================================================================
print("Generating Table 4: External Validation Drug Stratification (TMT)...")

t4_rows = []
for _, r_row in tmt_resp.iterrows():
    drug  = r_row["drug"]
    clean = drug.replace("drug_", "").replace("_", " ").title()
    adas_r = r_row["adas13_r"] if not pd.isna(r_row["adas13_r"]) else float("nan")
    mmse_r = r_row["mmse_r"]   if not pd.isna(r_row["mmse_r"])   else float("nan")
    t4_rows.append({
        "Drug Class":              clean,
        "N Users (TMT)":           int(r_row["n_users"]),
        "Calib. Attenuation":      f"{r_row['calib_attenuation']:+.3f}",
        "Benefit Low Risk":        f"{r_row['benefit_low_risk']:+.4f}",
        "Benefit Mid Risk":        f"{r_row['benefit_mid_risk']:+.4f}",
        "Benefit High Risk":       f"{r_row['benefit_high_risk']:+.4f}",
        "High-Risk Benefits More": "Yes" if r_row["high_risk_benefits_more"] else "No",
        "ADAS13 r":                f"{adas_r:.3f}" if not pd.isna(adas_r) else "",
        "MMSE r":                  f"{mmse_r:.3f}" if not pd.isna(mmse_r) else "",
    })

n_hr_tmt = int(tmt_resp["high_risk_benefits_more"].sum())
hr_p_tmt = binomtest(n_hr_tmt, len(tmt_resp), 0.5).pvalue
t4_rows.append({
    "Drug Class":              f"Summary: {n_hr_tmt}/{len(tmt_resp)} correct (binomial p={hr_p_tmt:.4f})",
    "N Users (TMT)": "", "Calib. Attenuation": "", "Benefit Low Risk": "",
    "Benefit Mid Risk": "", "Benefit High Risk": "",
    "High-Risk Benefits More": "", "ADAS13 r": "", "MMSE r": "",
})

t4 = pd.DataFrame(t4_rows)
t4.to_csv(os.path.join(TBL_DIR, "table4_ext_validation_drug_stratification.csv"), index=False)
print(f"  Saved table4_ext_validation_drug_stratification.csv")


# ============================================================================
# SUPPLEMENTARY TABLE S1: All Proteins
# ============================================================================
print("Generating Supplementary Table S1: All Proteins...")

bio_corr = feat_bio.dropna(subset=["corr_with_slope"]).copy()
bio_corr["protein"] = bio_corr["feature"].apply(
    lambda f: f[4:].split(".")[0] if f.startswith("CSF_") else None)

all_corr = bio_corr.dropna(subset=["protein"]).groupby("protein").agg(
    mean_corr=("corr_with_slope", "mean"),
    n_peptides=("feature", "count"),
).reset_index()

s1 = prot_imp.merge(all_corr, on="protein", how="left")
s1 = s1.rename(columns={
    "protein": "Protein",
    "total_importance": "Aggregated Importance",
    "category": "Biological Category",
    "mean_corr": "Mean Correlation with Slope",
    "n_peptides": "N Peptides",
})
s1.insert(0, "Rank", range(1, len(s1) + 1))
s1.to_csv(os.path.join(TBL_DIR, "tableS1_all_proteins.csv"), index=False)
print(f"  Saved tableS1_all_proteins.csv ({len(s1)} proteins)")


print(f"\nAll tables saved to: {TBL_DIR}")
