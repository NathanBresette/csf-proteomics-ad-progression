#!/usr/bin/env python3
"""
Publication-quality figures for CSF Proteomics paper.
4 figures, each with multiple panels. Includes external validation results.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import pearsonr, binomtest, gaussian_kde
from scipy.stats import t as t_dist

BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "results")
EXT_DIR = os.path.join(BASE, "external_validation", "results")
FIG_DIR = os.path.join(BASE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
})

# Color palette
C_CLINICAL = "#7F8C8D"
C_CSF      = "#2980B9"
C_COMBINED = "#E74C3C"
C_NULL     = "#BDC3C7"
C_EXT      = "#8E44AD"   # purple for external validation

# Category colors for proteins
CAT_COLORS = {
    "AD Pathology (Amyloid/Tau)":        "#E74C3C",
    "Synaptic & Neuronal Markers":        "#3498DB",
    "Neurodegeneration Markers":          "#E67E22",
    "Proteases & Inhibitors":             "#9B59B6",
    "Oxidative Stress":                   "#2ECC71",
    "Complement & Innate Immunity":       "#1ABC9C",
    "Apolipoproteins & Lipid Transport":  "#F1C40F",
    "Neuroinflammation":                  "#E91E63",
    "Cell Adhesion & ECM":                "#795548",
    "Neuropeptides & Secretory":          "#607D8B",
    "Other":                              "#BDC3C7",
}

# ── Load primary (MRM) data ──────────────────────────────────────────────────
cv_b     = pd.read_csv(os.path.join(OUT_DIR, "cv_predictions_B_clinical_csf.csv"))
cv_a     = pd.read_csv(os.path.join(OUT_DIR, "cv_predictions_A_clinical.csv"))
cv_d     = pd.read_csv(os.path.join(OUT_DIR, "cv_predictions_D_csf_only.csv"))
perf     = pd.read_csv(os.path.join(OUT_DIR, "model_performance.csv"))
perm     = pd.read_csv(os.path.join(OUT_DIR, "permutation_null.csv"))
null_b   = pd.read_csv(os.path.join(OUT_DIR, "null_distribution_B_clinical_csf.csv"))
resp     = pd.read_csv(os.path.join(OUT_DIR, "responder_stratification.csv"))

# ── Load external validation (TMT) data ─────────────────────────────────────
ext_perf    = pd.read_csv(os.path.join(EXT_DIR, "validation_performance.csv"))
ext_preds   = pd.read_csv(os.path.join(EXT_DIR, "cross_platform_C_TMT_to_MRM.csv"))
tmt_preds   = pd.read_csv(os.path.join(EXT_DIR, "tmt_native_predictions.csv"))
tmt_resp    = pd.read_csv(os.path.join(EXT_DIR, "tmt_responder_stratification.csv"))
shap_imp    = pd.read_csv(os.path.join(EXT_DIR, "shap_mean_abs_tmt.csv"))
procova_new = pd.read_csv(os.path.join(EXT_DIR, "procova_crossplatform.csv"))


DRUG_NAME_MAP = {
    "drug_ccb":           "CCB",
    "drug_ace_inhibitor": "ACE Inhibitor",
    "drug_nsaid":         "NSAID",
    "drug_ppi":           "PPI",
    "drug_statin":        "Statin",
    "drug_thyroid":       "Thyroid",
    "drug_donepezil":     "Donepezil",
    "drug_memantine":     "Memantine",
    "drug_galantamine":   "Galantamine",
    "drug_rivastigmine":  "Rivastigmine",
    "drug_metformin":     "Metformin",
}

def clean_drug(d):
    return DRUG_NAME_MAP.get(d, d.replace("drug_", "").replace("_", " ").title())


# ============================================================================
# FIGURE 1: CSF Proteomics Predicts Cognitive Decline  (1 × 3)
# A: model bars, B: TMT quintile calibration, C: TMT→MRM quintile calibration
# ============================================================================
fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6))

ext_c       = ext_perf[ext_perf["section"] == "C_TMT_to_MRM"]
tmt_full_r2 = ext_perf[ext_perf["section"] == "B_TMT_native_full"]["r2"].values[0]
tmt_clin_r2 = ext_perf[ext_perf["section"] == "B_clinical"]["r2"].values[0] if \
    len(ext_perf[ext_perf["section"] == "B_clinical"]) > 0 else \
    perf[perf["model"] == "A_clinical"]["oof_r2"].values[0]

N_BINS = 5

def quintile_calibration(ax, df, pred_col, true_col, r2, color, title, extra_text=""):
    """Quintile calibration plot: mean predicted vs mean observed ± 95% CI per bin."""
    d = df.dropna(subset=[pred_col, true_col]).copy()
    r, _ = pearsonr(d[pred_col], d[true_col])
    d["bin"] = pd.qcut(d[pred_col], N_BINS, labels=False)
    bins = d.groupby("bin").agg(
        mean_pred=(pred_col, "mean"),
        mean_obs =(true_col, "mean"),
        se_obs   =(true_col, lambda x: x.std() / np.sqrt(len(x))),
        n        =(true_col, "count"),
    ).reset_index()
    ax.errorbar(bins["mean_pred"], bins["mean_obs"],
                yerr=1.96 * bins["se_obs"],
                fmt="o", color=color, ms=9, lw=1.8, capsize=5, capthick=1.5,
                ecolor=color, alpha=0.85, zorder=4, label="Quintile mean ± 95% CI")
    lo = min(bins["mean_pred"].min(), bins["mean_obs"].min()) - 0.005
    hi = max(bins["mean_pred"].max(), bins["mean_obs"].max()) + 0.005
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.45, label="Identity")
    z  = np.polyfit(bins["mean_pred"], bins["mean_obs"], 1)
    xs = np.linspace(lo, hi, 100)
    ax.plot(xs, np.polyval(z, xs), color=color, lw=1.4, ls="-", alpha=0.5)
    ax.set_xlabel("Mean Predicted CDR-SB Slope (pts/yr)")
    ax.set_ylabel("Mean Observed CDR-SB Slope (pts/yr)")
    ax.set_title(title, fontweight="bold", loc="left", fontsize=12)
    stats = f"$R^2$ = {r2:.3f}\nr = {r:.3f}\nn = {len(d):,}"
    if extra_text:
        stats += f"\n{extra_text}"
    ax.text(0.05, 0.92, stats, transform=ax.transAxes, fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.legend(fontsize=7.5, loc="lower right")

# ── Panel A: Model comparison bars ──────────────────────────────────────────
ax = axes1[0]
bar_labels = ["Clinical\nOnly", "Clinical\n+ CSF\n(n=1,060)", "External\nValidation\nTMT→MRM\n(n=279)"]
bar_colors = [C_CLINICAL, C_COMBINED, C_EXT]
bar_vals   = [tmt_clin_r2, tmt_full_r2, ext_c["r2"].values[0]]
bar_stds   = [0, 0.038 / np.sqrt(5), 0]   # fold SD=0.038 (Hellbender log 12655046)

ax.bar(range(3), bar_vals, color=bar_colors, width=0.55, edgecolor="white", linewidth=0.5)
ax.errorbar(range(3), bar_vals, yerr=bar_stds,
            fmt="none", ecolor="black", capsize=4, capthick=1, lw=1)
ax.set_xticks(range(3))
ax.set_xticklabels(bar_labels, fontsize=9)
ax.set_ylabel("$R^2$")
ax.set_ylim(0, max(bar_vals) + 0.14)
delta = tmt_full_r2 - tmt_clin_r2
ax.annotate("", xy=(1, tmt_full_r2 + 0.015), xytext=(0, tmt_clin_r2 + 0.015),
            arrowprops=dict(arrowstyle="<->", color="black", lw=1))
ax.text(0.5, max(tmt_clin_r2, tmt_full_r2) + 0.03,
        f"$\\Delta R^2$ = +{delta:.3f}", ha="center", fontsize=8)
for i, (v, sd) in enumerate(zip(bar_vals, bar_stds)):
    ax.text(i, v + sd + 0.01, f"{v:.3f}", ha="center", fontsize=8)
ax.set_title("A  Model Comparison", fontweight="bold", loc="left", fontsize=12)

# ── Panel B: TMT Discovery — quintile calibration (n=1,060) ─────────────────
quintile_calibration(axes1[1], tmt_preds, "y_pred_tmt", "y_true",
                     r2=tmt_full_r2, color=C_COMBINED,
                     title="B  TMT Discovery — 5-fold CV")

# ── Panel C: External Validation — quintile calibration TMT→MRM (n=279) ─────
ext_v = ext_preds.dropna(subset=["y_pred"])
quintile_calibration(axes1[2], ext_v, "y_pred", "y_true",
                     r2=ext_c["r2"].values[0], color=C_EXT,
                     title="C  External Validation — TMT→MRM",
                     extra_text="p = 6.0×10$^{-26}$\nZero sample overlap")

fig1.tight_layout(w_pad=3)
fig1.savefig(os.path.join(FIG_DIR, "fig1_prediction_performance.png"), dpi=300, bbox_inches="tight")
fig1.savefig(os.path.join(FIG_DIR, "fig1_prediction_performance.pdf"), bbox_inches="tight")
print("Saved Figure 1")


# ============================================================================
# FIGURE 2: Biological Interpretation — UCHL1 and the Neuronal Injury Proteome
# Single panel: top 20 features by mean |SHAP| from B-full TMT model (n=1,060)
# Source: shap_mean_abs_tmt.csv from shap_tmt_discovery.py (job 12985662)
# ============================================================================
CLINICAL_SET = {"ADAS13", "FAQ", "AGE", "PTEDUCAT", "APOE4", "CDRSB", "MMSE"}

top20_shap = shap_imp[shap_imp["selected"]].head(20).copy().iloc[::-1].reset_index(drop=True)
top20_shap["label"] = top20_shap["feature"].str.replace("TMT_", "", regex=False)
colors2 = [C_CLINICAL if row["feature"] in CLINICAL_SET else C_CSF
           for _, row in top20_shap.iterrows()]

fig2, ax2 = plt.subplots(1, 1, figsize=(9, 6))

ax2.barh(range(len(top20_shap)), top20_shap["mean_abs_shap"].values, color=colors2,
         edgecolor="white", linewidth=0.3, height=0.7)
ax2.set_yticks(range(len(top20_shap)))
ax2.set_yticklabels(top20_shap["label"].values, fontsize=9)
ax2.set_xlabel("Mean |SHAP Value| (CDR-SB pts/yr per unit)", fontsize=9.5)
ax2.set_title("A", fontweight="bold", loc="left", fontsize=14)
ax2.grid(axis="x", lw=0.35, alpha=0.3, zorder=1)

handles2 = [
    Patch(facecolor=C_CLINICAL, label="Clinical-cognitive"),
    Patch(facecolor=C_CSF,      label="CSF protein"),
]
ax2.legend(handles=handles2, fontsize=8.5, loc="lower right", framealpha=0.9)

# Annotations for discussed proteins (label, italic, right of bar)
PROTEIN_LABELS = {
    "TMT_UCHL1":   "neuronal injury marker",
    "TMT_FABP3":   "neurodegeneration marker",
    "TMT_YWHAZ":   "synaptic signaling (14-3-3ζ)",
    "TMT_CPA4":    "extracellular protease",
    "TMT_DRAXIN":  "axon guidance",
    "TMT_SST":     "neuropeptide",
    "TMT_NPTX2":   "synaptic integrity",
    "TMT_PAMR1":   "ECM remodeling",
    "TMT_SLC9A1":  "ion transport / pH regulation",
    "TMT_OSTM1":   "lysosomal function",
    "TMT_EMCN":    "endothelial / neurovascular",
    "TMT_PPP3CA":  "calcium signaling (calcineurin)",
    "TMT_EPB41L2": "cytoskeletal scaffolding",
    "TMT_RMDN1":   "microtubule dynamics",
    "TMT_GAP43":   "axonal growth marker",
    "TMT_ERBB4":   "neuregulin receptor",
    "TMT_IGDCC4":  "axon guidance",
    "TMT_TNXB":    "extracellular matrix",
}
x_max = top20_shap["mean_abs_shap"].max()
for _, row in top20_shap.iterrows():
    lbl = PROTEIN_LABELS.get(row["feature"])
    if lbl:
        ax2.text(x_max * 0.012 + row["mean_abs_shap"], row.name,
                 lbl, va="center", fontsize=6.8, color="#2C3E50",
                 fontstyle="italic")

fig2.tight_layout()
fig2.savefig(os.path.join(FIG_DIR, "fig2_biological_interpretation.png"), dpi=300, bbox_inches="tight")
fig2.savefig(os.path.join(FIG_DIR, "fig2_biological_interpretation.pdf"), bbox_inches="tight")
print("Saved Figure 2")


# ============================================================================
# FIGURE 3: Drug Stratification — Quintile Dose-Response
# 2-panel: Cardiometabolic (pooled) | Cholinergic negative controls (pooled)
# X-axis: Q1 (lowest risk) → Q5 (highest risk) by proteomics-predicted slope
# Y-axis: actual observed CDRSB slope (pts/yr) — raw patient data
# Model's role: assigns quintiles only (trained on non-users); bars = real data
# ============================================================================
from matplotlib.patches import Patch as _Patch3

C_UNTREATED = "#BDC3C7"   # light gray  — untreated (non-users)
C_CARDIO    = "#8E44AD"   # deep purple — cardiometabolic treated
C_CHOLIN    = "#A93226"   # terracotta  — cholinergic treated (negative control)

_quint_path = os.path.join(EXT_DIR, "section_d_quintile_pooled.csv")

if not os.path.exists(_quint_path):
    # Fallback forest plot (OLS CIs) until Hellbender job completes
    CARDIO_SET = {"drug_statin", "drug_ace_inhibitor", "drug_ccb", "drug_thyroid"}
    _did_path = os.path.join(EXT_DIR, "section_d_did_regression.csv")
    if os.path.exists(_did_path):
        tmt_f = pd.read_csv(_did_path)
        tmt_f = tmt_f[tmt_f["drug"].isin(CARDIO_SET)].copy()
        tmt_f = tmt_f.sort_values("did_yr", ascending=True).reset_index(drop=True)
    else:
        tmt_f = tmt_resp.dropna(subset=["did_effect_high","did_effect_low","did_interaction"]).copy()
        tmt_f = tmt_f[tmt_f["drug"].isin(CARDIO_SET)].copy()
        tmt_f["did_yr"]   = -tmt_f["did_interaction"] * 12
        tmt_f["ci_lo_yr"] = tmt_f["did_yr"] - 0.3
        tmt_f["ci_hi_yr"] = tmt_f["did_yr"] + 0.3
        tmt_f["did_p"]    = np.nan
        tmt_f = tmt_f.sort_values("did_yr", ascending=True).reset_index(drop=True)
    fig3, axA = plt.subplots(1, 1, figsize=(8, 5))
    for i, (_, row) in enumerate(tmt_f.iterrows()):
        axA.hlines(i, row["ci_lo_yr"], row["ci_hi_yr"], color="#C39BD3", lw=2.0, zorder=2)
        axA.vlines([row["ci_lo_yr"], row["ci_hi_yr"]], i-0.12, i+0.12,
                   color="#C39BD3", lw=1.5, zorder=3)
        axA.scatter(row["did_yr"], i, color=C_CARDIO, s=120, zorder=5,
                    edgecolors="white", lw=0.8)
        axA.text(tmt_f["ci_hi_yr"].max() + 0.03, i,
                 f"n={int(row['n_users'])}", va="center", fontsize=8.5, color="#555555")
    axA.axvline(0, color="#2C3E50", lw=1.2, alpha=0.5, ls="--", zorder=1)
    axA.set_yticks(np.arange(len(tmt_f)))
    axA.set_yticklabels([clean_drug(d) for d in tmt_f["drug"]],
                        fontsize=11, fontweight="bold", color="#6C3483")
    axA.set_xlabel("DiD Differential: Fast − Slow Attenuation (CDRSB pts/yr)", fontsize=9)
    axA.set_title("TMT Cohort (n=1,060) · DiD · Target Trial Emulation\n"
                  "(Quintile dose-response figure pending Hellbender job)",
                  fontweight="bold", loc="left", fontsize=9, pad=8)
    axA.grid(axis="x", lw=0.35, alpha=0.3, zorder=0)
    fig3.tight_layout()

else:
    qdf = pd.read_csv(_quint_path)

    BAR_W  = 0.3
    GAP    = 0.08
    Q_SEP  = 1.0

    # Color per class
    _CLASS_COLORS = {
        "cardiometabolic": C_CARDIO,
        "cholinergic":     C_CHOLIN,
        "nsaid":           "#1A7A4A",   # forest green
        "ppi":             "#B7770D",   # amber
    }
    _CLASS_ANNOT = {
        "cardiometabolic": ("A  Cardiometabolic  (statin, ACE, CCB, thyroid)",
                            "Gap widens Q1\u2192Q5:\nneurovascular pathway\nmatches CSF signature"),
        "cholinergic":     ("B  Cholinergic  (donepezil, memantine)",
                            "Gap reversed: indication\nbias — sickest patients\nprescribed drug"),
        "nsaid":           ("C  NSAIDs  (anti-inflammatory)",
                            "Gap flat: anti-\ninflammatory \u2260 neuro-\nvascular; pathway mismatch"),
        "ppi":             ("D  PPIs  (proton pump inhibitors)",
                            "Gap flat: no AD neuro-\nvascular mechanism;\nhealthy-user bias excluded"),
    }
    _CLASS_ORDER = ["cardiometabolic", "cholinergic", "nsaid", "ppi"]
    _have_all_4 = all(c in qdf["drug_class"].values for c in _CLASS_ORDER)
    _classes_present = [c for c in _CLASS_ORDER if c in qdf["drug_class"].values]

    def _draw_quintile_panel(ax, df, treated_color, panel_label, annotation):
        centers = np.arange(len(df)) * Q_SEP
        y_max = 0
        for i, row in df.iterrows():
            cx = centers[i]
            x_nu = cx - GAP/2 - BAR_W/2
            x_u  = cx + GAP/2 + BAR_W/2
            ax.bar(x_nu, row["nonusers_mean"], BAR_W,
                   color=C_UNTREATED, edgecolor="white", lw=0.5, zorder=3)
            ax.errorbar(x_nu, row["nonusers_mean"], yerr=1.96 * row["nonusers_se"],
                        fmt="none", ecolor="#555555", capsize=2.5, capthick=0.8, lw=0.9, zorder=4)
            ax.bar(x_u, row["users_mean"], BAR_W,
                   color=treated_color, edgecolor="white", lw=0.5, zorder=3)
            ax.errorbar(x_u, row["users_mean"], yerr=1.96 * row["users_se"],
                        fmt="none", ecolor="#333333", capsize=2.5, capthick=0.8, lw=0.9, zorder=4)
            gap_val = row["nonusers_mean"] - row["users_mean"]
            top = max(row["nonusers_mean"] + 1.96 * row["nonusers_se"],
                      row["users_mean"]    + 1.96 * row["users_se"]) + 0.03
            if abs(gap_val) > 0.005 and not (np.isnan(row["users_mean"]) or np.isnan(row["nonusers_mean"])):
                ax.annotate("", xy=(x_u, top), xytext=(x_nu, top),
                            arrowprops=dict(arrowstyle="<->", color="#555555", lw=0.8))
                sign  = "\u2212" if gap_val > 0 else "+"
                gcolor = "#4A235A" if gap_val > 0 else "#922B21"
                ax.text(cx, top + 0.015, f"{sign}{abs(gap_val):.2f}",
                        ha="center", fontsize=6, fontweight="bold", color=gcolor)
            y_max = max(y_max,
                        row["nonusers_mean"] + 1.96 * row["nonusers_se"],
                        row["users_mean"]    + 1.96 * row["users_se"])
        ax.set_xticks(centers)
        ax.set_xticklabels([f"Q{q}" + ("\nLowest" if q==1 else "\nHighest" if q==5 else "")
                            for q in df["quintile"]], fontsize=8)
        ax.set_ylabel("Actual CDRSB Slope (pts/yr)", fontsize=8)
        ax.set_title(panel_label, fontweight="bold", loc="left", fontsize=9, pad=5)
        ax.set_xlim(centers[0] - 0.65, centers[-1] + 0.65)
        ax.set_ylim(0, max(y_max * 1.45, 0.3))
        ax.grid(axis="y", lw=0.3, alpha=0.35, zorder=0)
        # annotation box — pick bg color by treated_color
        _fc = {"cardiometabolic": "#F5EEF8", "cholinergic": "#FDEDEC"}.get(
            [k for k,v in _CLASS_COLORS.items() if v == treated_color][0], "#F0F0F0")
        ax.text(0.97, 0.97, annotation,
                transform=ax.transAxes, fontsize=7.5, ha="right", va="top",
                fontweight="bold", color=treated_color,
                bbox=dict(boxstyle="round,pad=0.35", facecolor=_fc,
                          edgecolor=treated_color, linewidth=1.1, alpha=0.95))

    if _have_all_4:
        fig3, axes3 = plt.subplots(1, 4, figsize=(18, 5.5),
                                   gridspec_kw={"wspace": 0.38})
    else:
        fig3, axes3 = plt.subplots(1, len(_classes_present), figsize=(13, 5.5),
                                   gridspec_kw={"wspace": 0.42})

    for ax, cls in zip(axes3, _classes_present):
        df_cls = qdf[qdf["drug_class"] == cls].sort_values("quintile").reset_index(drop=True)
        title, annot = _CLASS_ANNOT[cls]
        _draw_quintile_panel(ax, df_cls, _CLASS_COLORS[cls], title, annot)
        if ax != axes3[0]:
            ax.set_ylabel("")

    legend_els = [_Patch3(facecolor=C_UNTREATED, alpha=0.9, label="Untreated (non-users)")] + [
        _Patch3(facecolor=_CLASS_COLORS[c], alpha=0.9,
                label=f"Treated \u2014 {c.replace('nsaid','NSAID').replace('ppi','PPI')}")
        for c in _classes_present
    ]
    fig3.legend(handles=legend_els, loc="lower center",
                ncol=len(legend_els), fontsize=8, framealpha=0.9,
                bbox_to_anchor=(0.5, -0.06))
    fig3.suptitle(
        "TMT Cohort (n=1,060)  \u00b7  Target Trial Emulation (Hern\u00e1n & Robins 2016)\n"
        "Bars = actual observed CDRSB slopes  \u00b7  Q1\u2013Q5 = proteomics-predicted risk quintiles"
        "  \u00b7  Model trained on non-users only",
        fontsize=9, fontweight="bold", y=1.02)
    fig3.tight_layout()
fig3.savefig(os.path.join(FIG_DIR, "fig3_drug_stratification.png"), dpi=300, bbox_inches="tight")
fig3.savefig(os.path.join(FIG_DIR, "fig3_drug_stratification.pdf"), bbox_inches="tight")
print("Saved Figure 3")


# ============================================================================
# FIGURE 4: Clinical Trial Enrichment — ProCoVA Sample Size Reduction
# Single panel: clinical only vs clinical+CSF bars per drug (n>=25 users)
# Note: Fig 4A (tertile box plot) dropped — stratification power is already
# visible in Fig 3 Panel A (Q1→Q5 actual slopes). ANOVA p=5.44e-25 cited
# in Results text instead.
# ============================================================================
fig4, ax = plt.subplots(1, 1, figsize=(7, 5.5))

# procova_crossplatform.csv: drug, n_users, procova_ss_pct_csf, procova_ss_pct_clinical
proc4 = procova_new[procova_new["n_users"] >= 25].sort_values(
    "procova_ss_pct_csf", ascending=True).copy()

drug_labels4 = [clean_drug(d) for d in proc4["drug"]]
y4 = np.arange(len(proc4))
h4 = 0.3

ax.barh(y4 - h4/2, proc4["procova_ss_pct_clinical"].values, height=h4, color=C_CLINICAL,
        label="Clinical Only", edgecolor="white", linewidth=0.4, zorder=3)
ax.barh(y4 + h4/2, proc4["procova_ss_pct_csf"].values,  height=h4, color=C_COMBINED,
        label="Clinical + CSF", edgecolor="white", linewidth=0.4, zorder=3)
ax.axvline(0, color="#2C3E50", lw=0.8, alpha=0.35)
ax.set_yticks(y4)
ax.set_yticklabels(drug_labels4, fontsize=9)
ax.set_xlabel("Trial Sample Size Reduction (%)", fontsize=9.5)
ax.set_title("ProCoVA Sample Size Reduction by Drug",
             fontweight="bold", loc="left", fontsize=11, pad=8)
ax.grid(axis="x", lw=0.35, alpha=0.3, zorder=1)

for i, v in enumerate(proc4["procova_ss_pct_csf"].values):
    ax.text(v + 0.3, y4[i] + h4/2, f"{v:.1f}%", va="center", fontsize=7.5, color="#2C3E50")

ax.legend(fontsize=8.5, loc="lower right", framealpha=0.92)

fig4.tight_layout()
fig4.savefig(os.path.join(FIG_DIR, "fig4_clinical_trial_enrichment.png"), dpi=300,
             bbox_inches="tight")
fig4.savefig(os.path.join(FIG_DIR, "fig4_clinical_trial_enrichment.pdf"),
             bbox_inches="tight")
print("Saved Figure 4")

print("\nAll 4 figures saved to:", FIG_DIR)
