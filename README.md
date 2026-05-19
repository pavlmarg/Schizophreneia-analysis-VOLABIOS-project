# VOLABIOS — First-Episode Psychosis: 10-Year Longitudinal Analysis

This repository contains the full statistical analysis pipeline for the **VOLABIOS** open-access study — a 10-year longitudinal cohort study of patients with first-episode psychosis (FEP). The analysis covers univariate characterization, bivariate outcome comparisons, attrition modeling, and machine learning classification of diagnosis and recovery.

---

## Study Overview

| Item | Detail |
|---|---|
| **Cohort size** | 307 patients |
| **Follow-up period** | 10 years |
| **Primary outcome** | Recovery status at 10 years |
| **Secondary outcome** | Baseline diagnosis (Schizophrenia vs. Other) |
| **Data standard** | OMOP CDM (person, visit, measurement, observation, condition, death) |

---

## Repository Structure

```
.
├── data_files/
│   ├── raw/                        # Raw OMOP CDM tables (person.csv, visit_occurrence.csv, etc.)
│   └── data_processed/
│       └── csv_files/              # Processed master files
│           ├── master_baseline_comprehensive.csv
│           ├── baseline_simplified_data.csv
│           └── followup_10y_data.csv
│
├── scripts/
│   ├── univariate/
│   │   ├── univariate.py           # Full univariate pipeline: ETL, figures, tables
│   │   └── outlier_detection.py    # IQR-based outlier detection & severity segmentation
│   │
│   ├── bivariate/
│   │   ├── bivariate_recov.py      # Baseline predictors vs. 10-year recovery (t-test / chi-square)
│   │   ├── bivariate_10y.py        # Longitudinal symptom change in non-recovered patients (paired t-test)
│   │   ├── bivariate_cross.py      # Cross-variable correlation matrix & scatter plots (Pearson)
│   │   └── attrition_analysis.py   # Attrition bias analysis (Mann-Whitney U / chi-square)
│   │
│   ├── models/
│   │   └── diagnosis/
│   │       ├── lightgbm_diagnosis.py   # LightGBM classifier for diagnosis prediction
│   │       └── xgboost_diagnosis.py    # XGBoost classifier for diagnosis prediction
│   │
│   ├── analysis.py                 # 10-year follow-up data extraction from OMOP tables
│   ├── cleanup.py                  # Adds BPRS subscale columns to baseline file
│   └── completed.py                # Attrition pie chart (Figure 5)
│
├── results/
│   ├── results_biv/
│   │   ├── pictures/               # Bivariate plots (violin, swarm, stacked bar, heatmap)
│   │   └── tables/                 # Statistical summary CSVs
│   └── attrition_analysis/         # Attrition comparison tables & plots
│
└── outputs/
    └── diagnosis/
        └── lightgbm/onset_only/    # Model summaries, feature selection frequencies
```

---

## Pipeline Overview

### 1. Data Extraction & Master File Creation (`scripts/univariate/univariate.py`)

Reads raw OMOP CDM tables and produces `master_baseline_comprehensive.csv` containing:

- **Demographics:** age at onset, gender
- **Symptom scales:** BPRS Total, SAPS Total, SANS Total, BPRS subscales (Positive, Negative, Disorganized)
- **Illness course durations:** DUP, DUI, DAP, DAT (all in months)
- **Social/clinical flags:** cannabis use, family history of psychosis, hospitalization, education, SES, employment, marital status, living situation
- **Diagnosis** (mapped from OMOP condition concept IDs)
- **10-year attrition status** and **recovery status**

Generates 10 standardized univariate figures and two statistical tables (categorical counts, numerical descriptives).

### 2. Follow-Up Data Extraction (`scripts/analysis.py`)

Extracts 10-year measurements and observations from OMOP tables and produces `followup_10y_data.csv` with:
- BPRS, SAPS, SANS at 10 years
- BPRS subscales at 10 years
- Active status and recovery status at 10 years

### 3. Outlier Detection (`scripts/univariate/outlier_detection.py`)

Applies IQR-based outlier detection (1.5× and 3.0× IQR thresholds) across 8 clinical variables. Outputs two sorted patient-level reports:
- `extreme_cases_report.csv` — patients with at least one extreme outlier value
- `regular_cases_report.csv` — patients with only moderate outliers

### 4. Bivariate Analyses

| Script | Comparison | Test |
|---|---|---|
| `bivariate_recov.py` | Baseline variables vs. recovery at 10y | Independent t-test / chi-square |
| `bivariate_10y.py` | Baseline vs. 10y symptoms (Not Recovered group) | Paired t-test |
| `bivariate_cross.py` | Cross-variable correlation matrix | Pearson r |
| `attrition_analysis.py` | Baseline predictors of dropout | Mann-Whitney U / chi-square |

All scripts generate visualizations only for comparisons reaching p < 0.3, and export full statistical summary CSVs regardless of significance.

### 5. Machine Learning — Diagnosis Classification

Both `lightgbm_diagnosis.py` and `xgboost_diagnosis.py` implement an identical, rigorous pipeline to classify **Schizophrenia vs. Other psychosis** from baseline features:

**Pipeline steps (per fold):**
1. Stratified 5-fold CV × 5 random seeds = **25 evaluation folds**
2. Iterative imputation (Random Forest) fitted on train only — no leakage
3. Two-stage feature selection: univariate filter (p < 0.3) → forward sequential selection (macro-F1)
4. SMOTE oversampling on training data
5. Optuna hyperparameter search (25 trials, TPE sampler, 3-fold inner CV)
6. Optimal classification threshold tuned on inner CV
7. Evaluation on held-out outer fold

**Outputs:**
- Mean AUC, F1, Precision, Recall ± SD across 25 folds
- Feature selection frequency across folds
- Confusion matrix, ROC curve, and feature importance plots (median fold)
- SHAP summary plot (full-data model retrained on median fold's parameters)

#### Current Model Performance (LightGBM — Onset Features Only)

| Metric | Mean | Std |
|---|---|---|
| AUC | 0.769 | 0.062 |
| F1 | 0.703 | 0.058 |
| Precision | 0.704 | 0.057 |
| Recall | 0.707 | 0.057 |

**Most consistently selected features (LightGBM):**

| Feature | Selection Rate |
|---|---|
| DAP_months | 88% |
| BPRS_Positive_Onset | 68% |
| DUP_months | 64% |
| DAT_months | 56% |
| hospital_admission_Avoided | 48% |

---

## Key Variables

| Variable | Description |
|---|---|
| `baseline_age` | Age at first episode |
| `BPRS_Total` | Brief Psychiatric Rating Scale — total score |
| `BPRS_Positive_Onset` | BPRS positive subscale (hallucinations, delusions) |
| `BPRS_Negative_Onset` | BPRS negative subscale (emotional/social withdrawal) |
| `BPRS_Disorganized_Onset` | BPRS disorganized subscale (thought disorder) |
| `SAPS_Total` | Scale for Assessment of Positive Symptoms |
| `SANS_Total` | Scale for Assessment of Negative Symptoms |
| `DUP_months` | Duration of Untreated Psychosis |
| `DUI_months` | Duration of Untreated Illness |
| `DAP_months` | Duration of Active Psychosis before baseline |
| `DAT_months` | Duration of Antipsychotic Treatment before baseline |

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scipy
scikit-learn
imbalanced-learn
xgboost
lightgbm
optuna
shap
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn imbalanced-learn xgboost lightgbm optuna shap
```

---

## Running the Pipeline

```bash
# 1. Build the master baseline file and univariate figures
python scripts/univariate/univariate.py

# 2. Add BPRS subscale columns
python scripts/cleanup.py

# 3. Extract 10-year follow-up data
python scripts/analysis.py

# 4. Run bivariate analyses
python scripts/bivariate/bivariate_recov.py
python scripts/bivariate/bivariate_10y.py
python scripts/bivariate/bivariate_cross.py
python scripts/bivariate/attrition_analysis.py

# 5. Run outlier detection
python scripts/univariate/outlier_detection.py

# 6. Train classification models
python scripts/models/diagnosis/lightgbm_diagnosis.py
python scripts/models/diagnosis/xgboost_diagnosis.py
```

---

## License

Open access — VOLABIOS study. See project documentation for data use terms.
