# STROBE Checklist — Cohort Study

**Manuscript:** Cross-Platform CSF Proteomics Enables a Mechanistic Digital Twin of Alzheimer's Disease Progression
**Authors:** Nathan Bresette, Ai-Ling Lin
**Design:** Secondary analysis of two prospective observational cohorts (ADNI TMT and MRM)

---

| Item | Recommendation | Location in Manuscript | Notes |
|------|---------------|------------------------|-------|
| **TITLE AND ABSTRACT** | | | |
| 1a | Indicate the study design with a commonly used term in the title or abstract | Abstract (Methods); Title | "independent...cohort" and cross-platform validation design described |
| 1b | Provide an informative and balanced summary of what was done and what was found | Abstract (all sections) | Structured IMRAD abstract; results and limitations included |
| **INTRODUCTION** | | | |
| 2 | Explain the scientific background and rationale | Introduction, Paragraphs 1–4 | Heterogeneity problem, CSF biology, neurovascular hypothesis, cross-platform gap |
| 3 | State specific objectives, including prespecified hypotheses | Introduction, Paragraph 4 | Objectives stated: trajectory prediction, cross-platform validation, drug stratification, ProCoVA enrichment |
| **METHODS** | | | |
| 4 | Present key elements of study design early | Methods 2.1–2.2; Introduction Paragraph 4 | TMT discovery + MRM external validation design presented at study outset |
| 5 | Describe the setting, locations, and relevant dates | Methods 2.1–2.2 | ADNI multi-site; TMT n=1,060, MRM n=279; ≥3 CDR-SB visits spanning ≥12 months |
| 6 | Give eligibility criteria and sources/methods of participant selection; describe follow-up | Methods 2.1–2.2; Table 1 | Inclusion: ≥1 proteomics measurement, ≥3 CDR-SB visits, ≥12-month span. Mean follow-up: TMT 60.2±39.9 months, MRM 64.3±49.9 months |
| 7 | Clearly define all outcomes, exposures, predictors, potential confounders, and effect modifiers | Methods 2.3–2.5; Table 1 | Outcome: CDR-SB slope (OLS). Predictors: 7 clinical-cognitive variables + CSF proteins. Drug exposure: baseline medication records |
| 8 | For each variable, give sources of data and details of assessment | Methods 2.1–2.5 | CDR-SB from ADNI visits; proteins from TMT (isobaric labeling) and MRM (selected ion monitoring); drug use from baseline medication records |
| 9 | Describe efforts to address potential sources of bias | Methods 2.5–2.7; Results 3.5; Discussion | Non-circular drug model (trained on non-users only); DiD to address prescribing-by-indication; negative control drug classes (cholinergic, NSAID, PPI); permutation null for overfitting |
| 10 | Explain how the study size was arrived at | Methods 2.1–2.2 | Sample sizes determined by ADNI data availability with minimum visit requirements; no prospective power calculation (secondary data analysis) |
| 11 | Explain how quantitative variables were handled | Methods 2.3–2.5 | CDR-SB slope: continuous OLS estimate. Proteins: z-scored within platform. LASSO for feature selection. Quintile stratification for drug analyses |
| 12a | Describe all statistical methods, including those for controlling confounding | Methods 2.5–2.9 | LASSO + GBM; 5-fold CV; OLS DiD with drug-by-risk-tertile interaction; ANCOVA (ProCoVA); permutation null |
| 12b | Describe methods for subgroup and sensitivity analyses | Methods 2.7–2.9; Supplementary | Drug-class subgroups; permutation null (n=1,000); imaging sensitivity analysis (figS5); 18-month sensitivity dropped |
| 12c | Explain how missing data were addressed | Methods 2.5 | Median imputation within each training fold for clinical variables; protein missingness threshold 30% coverage |
| 12d | Describe methods for loss to follow-up | Methods 2.1–2.2 | Minimum visit criteria applied at inclusion; no post-enrollment attrition modeled (secondary ADNI data) |
| 12e | Describe any sensitivity analyses | Methods 2.9; Supplementary figS5 | Permutation null (1,000 permutations); imaging sensitivity (AV45, Hippo_norm importance = 0) |
| **RESULTS** | | | |
| 13 | Report numbers of participants at each stage; give reasons for non-participation | Methods 2.1–2.2; Table 1 | TMT: 1,060 included. MRM: 279 included. Exclusions: <3 CDR-SB visits or <12-month span. Zero sample overlap confirmed by Research ID matching |
| 14 | Give characteristics of study participants; summarize follow-up time | Table 1; Results 3.1 | Age, education, sex, APOE4, cognitive scores, diagnosis, drug use, follow-up duration all reported with p-values |
| 15 | Report numbers of outcome events or summary measures over time | Results 3.2–3.3 | CDR-SB slope: TMT 0.5±0.9 pts/yr, MRM 0.9±1.1 pts/yr (Table 1). Model performance: R²=0.415 (TMT), R²=0.275 (MRM) |
| 16 | Report unadjusted estimates; if applicable, confounder-adjusted estimates | Results 3.5 | Quintile-level mean observed slopes (unadjusted) and OLS DiD coefficients with 95% CIs reported for each drug class |
| 17 | Report other analyses done (subgroups, interactions, sensitivity) | Results 3.2, 3.5, 3.6; Supplementary | Clinical-only baseline (R²=0.250) vs full model; drug-class subgroup analyses; ProCoVA by drug class; imaging sensitivity in Supplementary figS5 |
| **DISCUSSION** | | | |
| 18 | Summarize key results with reference to study objectives | Discussion, Paragraph 1 | Cross-platform R²=0.275 contextualized against published prognostic covariate benchmarks |
| 19 | Discuss limitations, including sources of bias or imprecision | Discussion, Paragraph 4 (Limitations) | Observational drug analyses; co-medication confounding; 138-protein cross-platform bridge; LP requirement; ADNI cohort generalizability |
| 20 | Give a cautious overall interpretation of results | Discussion throughout; Research in Context | Drug stratification described as hypothesis-generating; RCT confirmation required; findings framed as motivating prospective validation |
| 21 | Discuss the generalizability (external validity) of the study results | Discussion, Paragraph 4; Future Directions | ADNI is research-grade; MRI-enriched; predominantly white; LP required; plasma proteomics as future non-invasive alternative noted |
| **OTHER INFORMATION** | | | |
| 22 | Give the source of funding and role of funders | Acknowledgments | NIH/NIA R56AG079586 (A-LL); ADNI U01 AG024904 + DOD W81XWH-12-2-0012; Hellbender HPC DOI:10.32469/10355/97710. Funders had no role in study design, analysis, or decision to publish |

---

*STROBE Statement reference: von Elm E, Altman DG, Egger M, et al. The Strengthening the Reporting of Observational Studies in Epidemiology (STROBE) statement: guidelines for reporting observational studies. Lancet. 2007;370(9596):1453-1457. doi:10.1016/S0140-6736(07)61602-X*
