# CSF Proteomics for Alzheimer's Disease Progression Prediction

Analysis code for:

**Cross-Platform CSF Proteomics Enables a Mechanistic Digital Twin of Alzheimer's Disease Progression**
Nathan Bresette, Ai-Ling Lin
University of Missouri

---

## Overview

This repository contains all analysis scripts used to produce the results, figures, and tables in the manuscript. The pipeline trains gradient-boosting models on CSF proteomics from the ADNI TMT cohort (n=1,060) to predict CDR-SB slope, validates predictions in an independent MRM cohort (n=279), identifies proteomic predictors via SHAP, stratifies cardiometabolic drug response, and quantifies clinical trial enrichment via ProCoVA.

---

## Data Access

Data are not included in this repository. All data are from the Alzheimer's Disease Neuroimaging Initiative (ADNI).

**To access ADNI data:**
1. Apply at [adni.loni.usc.edu](https://adni.loni.usc.edu)
2. Download the TMT CSF proteomics dataset and MRM CSF proteomics dataset
3. Download ADNI clinical data (CDR-SB longitudinal visits, demographics, medication records)

ADNI is funded by the National Institutes of Health (U01 AG024904) and the Department of Defense (W81XWH-12-2-0012).

---

## Repository Structure

```
scripts/
    build_cdrsb_slope_dataset.py      # Build MRM dataset: compute CDR-SB slopes, merge clinical + CSF features + drug flags
    responder_stratification.py        # Drug stratification: non-circular quintile analysis and DiD (Section D, Fig 3)
    procova_variance_reduction.py      # ProCoVA sample size reduction estimation (Section E, Fig 4)
    feature_importance_biology.py      # SHAP analysis of TMT model (Fig 2)
    sensitivity_imaging.py             # Imaging sensitivity analysis (Fig S5)
    permutation_null_1000.py           # Permutation null distribution (n=1,000 permutations)
    generate_figures.py                # Reproduce Figures 1-4
    generate_supplementary_figures.py  # Reproduce supplementary figures
    generate_tables.py                 # Reproduce Table 1

figures/
    fig1_prediction_performance.png    # Model comparison and calibration
    fig2_biological_interpretation.png # SHAP feature importance
    fig3_drug_stratification.png       # Cardiometabolic drug quintile analysis
    fig4_clinical_trial_enrichment.png # ProCoVA sample size reduction
    figS5_imaging_sensitivity.png      # Imaging sensitivity analysis

STROBE_CHECKLIST.md                    # STROBE reporting checklist for observational studies
```

**Note:** TMT discovery cohort training scripts were run on the University of Missouri Hellbender HPC cluster (DOI: 10.32469/10355/97710) and are available upon request.

---

## Reproducibility

### Environment

Python 3.11 is required.

```bash
pip install -r requirements.txt
```

### Running Order

Run scripts in the following order to reproduce the full pipeline:

```bash
# 1. Build MRM dataset
python scripts/build_cdrsb_slope_dataset.py

# 2. Drug stratification (requires TMT model outputs)
python scripts/responder_stratification.py

# 3. ProCoVA enrichment analysis
python scripts/procova_variance_reduction.py

# 4. SHAP feature importance
python scripts/feature_importance_biology.py

# 5. Permutation null
python scripts/permutation_null_1000.py

# 6. Imaging sensitivity
python scripts/sensitivity_imaging.py

# 7. Generate all figures
python scripts/generate_figures.py
python scripts/generate_supplementary_figures.py

# 8. Generate tables
python scripts/generate_tables.py
```

Scripts expect data files at paths defined within each script. Update the `DATA_DIR` variable at the top of each script to match your local ADNI data location.

---

## Key Results

| Model | Cohort | R2 | r | n |
|-------|--------|-----|---|---|
| Clinical-cognitive baseline | TMT (5-fold CV) | 0.250 | -- | 1,060 |
| Clinical + 2,492 CSF proteins | TMT (5-fold CV) | 0.415 | 0.648 | 1,060 |
| Cross-platform (TMT to MRM) | MRM (held out) | 0.275 | 0.575 | 279 |

Top CSF predictor: UCHL1 (mean |SHAP| = 0.0046)

Cardiometabolic drug response: 4/4 directional consistency in fast progressors (DiD)

ProCoVA sample size reduction: 10-43% (CSF + clinical) vs 0.2-23.3% (clinical only)

---

## Citation

If you use this code, please cite the manuscript (citation to be added upon publication).

ADNI citation: Weiner MW et al. *Alzheimers Dement.* 2017;13(5):561-571. doi:10.1016/j.jalz.2016.10.006

---

## License

MIT License. See LICENSE file.

---

## Contact

Nathan Bresette, University of Missouri -- nbhtd@umsystem.edu
