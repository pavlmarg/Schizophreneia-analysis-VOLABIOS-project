# VOLABIOS — First-Episode Psychosis: 10-Year Longitudinal Analysis

> **Note:** This project was developed as part of a university internship placement. It represents a complete, end-to-end data analysis pipeline built on real clinical cohort data — from raw OMOP CDM tables through exploratory analysis to machine learning classification — under the supervision of the VOLABIOS research group.

This repository contains the full statistical analysis pipeline for the **VOLABIOS** open-access study — a 10-year longitudinal cohort study of patients with first-episode psychosis (FEP). The analysis spans four stages: univariate characterization of the cohort at baseline, bivariate exploration of predictors and outcomes, attrition bias assessment, and machine learning classification of diagnosis and long-term recovery. All data are structured according to the OMOP Common Data Model (CDM).

---

## Study Overview

| Item | Detail |
|---|---|
| **Cohort size** | 307 patients |
| **Follow-up period** | 10 years |
| **Primary outcome** | Recovery status at 10 years |
| **Secondary outcome** | Baseline diagnosis (Schizophrenia vs. Other) |
| **Data standard** | OMOP CDM (person, visit, measurement, observation, condition, death) |
| **Clinical scales** | BPRS, SAPS, SANS |
| **Illness course markers** | DUP, DUI, DAP, DAT (all in months) |

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

### Stage 1 — Data Extraction & Master File Creation (`scripts/univariate/univariate.py`)

The pipeline begins with a complete ETL pass over the raw OMOP CDM tables. The script joins `person`, `visit_occurrence`, `measurement`, `observation`, and `condition_occurrence` into a single patient-level master file (`master_baseline_comprehensive.csv`). A custom multi-strategy observation extractor handles the common messiness of real-world OMOP data — mapping both numeric concept IDs and raw source value strings in parallel so that no records are silently dropped due to inconsistent coding.

The master file contains:

- **Demographics:** age at onset, gender
- **Symptom scales:** BPRS Total, SAPS Total, SANS Total, BPRS subscales (Positive, Negative, Disorganized)
- **Illness course durations:** DUP, DUI, DAP, DAT (all in months)
- **Social/clinical flags:** cannabis use, family history of psychosis, hospitalization, education, SES, employment, marital status, living situation
- **Diagnosis** (mapped from OMOP condition concept IDs to readable labels)
- **10-year attrition status** and **recovery status**

The script also generates 10 standardized univariate figures (see the EDA section below) and exports two statistical tables: `table_1_categorical.csv` with counts and percentages for all categorical variables, and `table_2_numerical.csv` with full descriptive statistics (mean, SD, median, min, max, variance) for all continuous variables.

### Stage 2 — Follow-Up Data Extraction (`scripts/analysis.py`)

A dedicated script extracts all 10-year measurements and observations from the OMOP tables and produces `followup_10y_data.csv`. This includes:
- BPRS Total, SAPS Total, SANS Total at 10 years
- BPRS Positive and Negative subscales at 10 years
- Active/inactive status at 10 years
- Recovery status at 10 years

This file is used as the dependent variable source for all downstream bivariate and longitudinal analyses.

### Stage 3 — Outlier Detection (`scripts/univariate/outlier_detection.py`)

IQR-based outlier detection is applied across 8 key clinical variables (`baseline_age`, `BPRS_Total`, `SAPS_Total`, `SANS_Total`, `DUP_months`, `DUI_months`, `DAP_months`, `DAT_months`). Two severity tiers are flagged:

- **Regular outliers:** values beyond 1.5× IQR from the quartile boundaries
- **Extreme outliers:** values beyond 3.0× IQR

Patients are segmented into two output files — `extreme_cases_report.csv` and `regular_cases_report.csv` — both sorted so that patients with available 10-year follow-up data appear first, making clinical review more efficient.

### Stage 4 — Bivariate Analyses

| Script | Comparison | Test |
|---|---|---|
| `bivariate_recov.py` | Baseline variables vs. recovery at 10y | Independent t-test / chi-square |
| `bivariate_10y.py` | Baseline vs. 10y symptoms (Not Recovered group) | Paired t-test |
| `bivariate_cross.py` | Cross-variable correlation matrix | Pearson r |
| `attrition_analysis.py` | Baseline predictors of dropout | Mann-Whitney U / chi-square |

All scripts generate visualizations only for comparisons reaching p < 0.3, and export full statistical summary CSVs regardless of significance. See the EDA section below for a detailed discussion of each analysis.

---

## Exploratory Data Analysis

The EDA was designed to characterize the cohort at baseline, assess whether the sample is representative across demographic and clinical dimensions, identify potential predictors of long-term outcome, and flag data quality concerns before modeling.

### Univariate Analysis

The univariate stage produces a comprehensive descriptive portrait of all 307 patients at first episode. Ten figures are generated, each grouping thematically related variables:

- **Figure 1 — Core Demographics:** Gender distribution (pie chart) and age at onset (histogram with KDE). This establishes the basic composition of the cohort and the shape of the age distribution at first presentation, which in FEP samples tends to peak in early adulthood.
- **Figure 2 — Education & SES:** Count plots for educational level and socioeconomic status. These are important social determinants of outcome — lower SES and shorter educational trajectories are consistently associated with worse long-term functioning in psychosis research.
- **Figure 3 — Employment & Marital Status:** Employment status (Active in work or study vs. Inactive) and marital status at the time of the first episode, capturing the degree to which the illness had already disrupted social role functioning by baseline.
- **Figure 4 — Living Situation & Family History:** Whether the patient lived with parents at onset (a proxy for social independence and support availability), and the presence of a family history of psychosis (a known genetic risk modifier).
- **Figure 5 — Diagnosis & Cannabis Use:** Distribution of baseline diagnoses across the eight OMOP concept categories (ranging from Schizophrenia and Schizoaffective to Brief Reactive Psychosis and Bipolar with psychotic features), and the proportion reporting cannabis use at onset.
- **Figure 6 — Symptom Severity I:** Box plots for BPRS Total and SAPS Total, with individual means marked. These provide a visual check on score distributions and the presence of skew or ceiling effects — SAPS in particular can show floor clustering in samples dominated by negative-symptom presentations.
- **Figure 7 — Symptom Severity II & DUP:** Box plot for SANS Total alongside a histogram of Duration of Untreated Psychosis. DUP is typically strongly right-skewed in FEP cohorts, with a small number of patients going years without treatment pulling the tail.
- **Figure 8 — Illness Delays:** Histograms with KDE for DUI and DAP in months, capturing the full pre-treatment illness timeline. These variables often show bimodal patterns when samples mix rapid-onset and insidious presentations.
- **Figure 9 — Treatment & Hospitalization:** Distribution of Duration of Antipsychotic Treatment before the baseline assessment, and the proportion of patients requiring inpatient hospitalization at onset — an indicator of illness severity and service contact intensity.
- **Figure 10 — 10-Year Outcomes:** A pie chart of cohort attrition across all 307 patients (Completed / Lost to Follow-up / Deceased) and a count plot of recovery status among those who returned, providing an early summary of the study's primary outcome landscape.

Together, these figures serve as a data quality audit as much as a descriptive summary — flagging variables with heavy missingness, unexpected distributions, or categories that may need collapsing before modeling.

### Bivariate Analysis — Baseline Predictors of Recovery (`bivariate_recov.py`)

This script is the core of the exploratory outcome analysis. It tests every baseline variable (11 continuous, 10 categorical) against 10-year recovery status (Recovered / Not Recovered) and produces a comprehensive statistical report.

**For continuous variables**, independent samples t-tests compare group means, with each variable summarized by mean ± SD per group. Variables passing a liberal p < 0.3 threshold are visualized using violin + swarm plot combinations — violin plots to show the full distribution shape per group, swarm plots overlaid to display individual data points and avoid hiding the underlying sample size. The p < 0.3 threshold is intentionally liberal at this exploratory stage, casting a wide net to avoid prematurely discarding potentially informative signals before modeling.

The continuous variables tested span symptom severity (BPRS Total and its three subscales, SAPS Total, SANS Total), illness course timing (DUP, DUI, DAP, DAT), and age at onset. Differences in negative symptom burden (BPRS Negative, SANS Total) and illness duration markers are of particular clinical interest, as both have been hypothesized to predict poorer long-term outcomes in first-episode cohorts.

**For categorical variables**, chi-square tests of independence are used on full contingency tables. Results are visualized as 100%-stacked bar charts showing the proportion of Recovered vs. Not Recovered within each category level — a format that makes between-category recovery rate differences immediately readable. All counts and percentages (including Recovered count, Recovered %, and Not Recovered count) are saved in the summary CSV regardless of whether the result reaches the visualization threshold.

### Bivariate Analysis — Longitudinal Symptom Change (`bivariate_10y.py`)

This script focuses exclusively on patients who were **not recovered** at 10 years, examining how their symptom burden changed over the decade. Paired t-tests compare baseline vs. 10-year scores for five clinical measures: BPRS Total, SAPS Total, SANS Total, BPRS Positive, and BPRS Negative.

Each pair is visualized with a point plot showing the group mean ± SD at both time points (to capture the overall direction and magnitude of change) overlaid with a swarm plot of individual trajectories (to reveal how much heterogeneity underlies the group-level estimate). This combination makes it possible to spot cases where the mean shifts but individual variation is large — a common and clinically meaningful pattern in psychosis research, where some patients improve while others deteriorate even within a "not recovered" group.

This analysis is particularly valuable for understanding whether non-recovery at 10 years represents a stable, chronic state or continued active deterioration, and whether positive and negative symptom dimensions follow different longitudinal trajectories within this subgroup.

### Bivariate Analysis — Cross-Variable Correlation Matrix (`bivariate_cross.py`)

A curated Pearson correlation matrix is computed across a selection of non-redundant continuous and categorical variables spanning both baseline and 10-year timepoints. Categorical variables (gender, cannabis use, hospitalization, diagnosis, recovery status, active status) are label-encoded for inclusion in the matrix. The resulting heatmap uses a diverging RdBu_r palette centered at zero, making positive and negative associations immediately distinguishable.

Beyond the global heatmap, the script generates individual pairwise plots for all variable pairs with |r| between 0.1 and 0.9 — scatter plots with regression lines for continuous–continuous pairs, violin plots for continuous–categorical pairs, and stacked bar charts for categorical–categorical pairs. The lower bound of 0.1 filters out noise; the upper bound of 0.9 excludes near-perfect collinear pairs (e.g., redundant subscales derived from the same assessment) that would add little interpretive value.

This stage is primarily useful for identifying multicollinearity risks ahead of modeling, and for spotting theoretically interesting cross-time associations — for instance between baseline illness duration markers and 10-year symptom scores, or between social functioning variables at baseline and recovery outcomes.

### Bivariate Analysis — Attrition Bias Assessment (`attrition_analysis.py`)

A critical concern in any longitudinal study is whether patients lost to follow-up differ systematically from those who returned. If dropouts are not random — if, for example, more severely ill patients were more likely to disengage — then outcome estimates in the returning sample will be biased toward better prognosis. This script directly tests for attrition bias by comparing all available baseline characteristics between patients who returned for the 10-year assessment and those who did not.

Mann-Whitney U tests are used for continuous variables (preferred over t-tests because illness duration variables like DUP and DUI are typically non-normally distributed with heavy right skew). Chi-square tests are used for categorical variables. Results are sorted by p-value so the strongest potential sources of bias are immediately visible. The output CSV includes group medians and means for continuous variables and category-level counts and percentages for categorical variables — a complete Table S1-style attrition comparison suitable for supplementary reporting in a manuscript.

Visualization is again limited to variables reaching p < 0.3, using the same violin + swarm approach for continuous variables and stacked bar charts for categorical ones.

---

## Machine Learning — Diagnosis Classification

Both `lightgbm_diagnosis.py` and `xgboost_diagnosis.py` implement an identical, rigorous nested cross-validation pipeline to classify **Schizophrenia vs. Other psychosis** from baseline-only features. The pipeline is designed to be methodologically conservative, with multiple safeguards against data leakage and overfitting given the modest sample size.

**Pipeline steps (per fold):**
1. Stratified 5-fold CV × 5 random seeds = **25 evaluation folds**
2. One-hot encoding of categorical features, with validation columns aligned to training columns to handle unseen levels gracefully
3. Iterative imputation (Random Forest estimator) fitted on training data only, applied to validation — no leakage
4. Two-stage feature selection: univariate filter (ANOVA F / chi-square, p < 0.3) → forward sequential selection optimizing macro-F1
5. SMOTE oversampling applied only within training folds
6. Optuna hyperparameter search (25 trials, TPE sampler, 3-fold inner CV on the SMOTE'd training set)
7. Optimal classification threshold tuned on a second inner CV loop (separate from the Optuna loop) to avoid threshold overfitting
8. Evaluation on the held-out outer fold

**Outputs per run:**
- Mean AUC, F1, Precision, Recall ± SD across 25 folds
- Feature selection frequency table across all folds (a robustness indicator for which features matter consistently)
- Confusion matrix, ROC curve, and feature importance plots for the median-AUC fold
- SHAP summary plot generated from the full-data model retrained using the median fold's parameters

### Current Model Performance (LightGBM — Onset Features Only)

| Metric | Mean | Std |
|---|---|---|
| AUC | 0.769 | 0.062 |
| F1 | 0.703 | 0.058 |
| Precision | 0.704 | 0.057 |
| Recall | 0.707 | 0.057 |
| N Folds | 25 | — |

### Most Consistently Selected Features (LightGBM)

| Feature | Selection Count | Selection Rate |
|---|---|---|
| DAP_months | 22 / 25 | 88% |
| BPRS_Positive_Onset | 17 / 25 | 68% |
| DUP_months | 16 / 25 | 64% |
| DAT_months | 14 / 25 | 56% |
| hospital_admission_Avoided | 12 / 25 | 48% |
| DUI_months | 11 / 25 | 44% |
| SANS_Total | 9 / 25 | 36% |
| family_hx_psychosis_Yes | 7 / 25 | 28% |

The dominance of illness duration features (DAP, DUP, DUI, DAT) and positive symptom severity (BPRS Positive) in the selection frequency table aligns with established clinical knowledge: longer untreated illness and more severe positive symptoms at onset are known correlates of schizophrenia spectrum diagnoses relative to other psychotic disorders.

---

## Key Variables

| Variable | Description |
|---|---|
| `baseline_age` | Age at first episode |
| `BPRS_Total` | Brief Psychiatric Rating Scale — total score |
| `BPRS_Positive_Onset` | BPRS positive subscale (hallucinations, delusions) |
| `BPRS_Negative_Onset` | BPRS negative subscale (emotional/social withdrawal) |
| `BPRS_Disorganized_Onset` | BPRS disorganized subscale (thought disorder, bizarre behavior) |
| `SAPS_Total` | Scale for Assessment of Positive Symptoms — total score |
| `SANS_Total` | Scale for Assessment of Negative Symptoms — total score |
| `DUP_months` | Duration of Untreated Psychosis (months) |
| `DUI_months` | Duration of Untreated Illness (months) |
| `DAP_months` | Duration of Active Psychosis before baseline (months) |
| `DAT_months` | Duration of Antipsychotic Treatment before baseline (months) |
| `recovery_status_10y` | 10-year outcome: Recovered / Not Recovered |
| `attrition_status` | Returned for Follow-up / Deceased / Withdrew or Lost |

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

Install all dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn imbalanced-learn xgboost lightgbm optuna shap
```

Python 3.9+ is recommended.

---

## Running the Pipeline

Scripts should be run in the following order, as each stage depends on outputs from the previous one:

```bash
# 1. Build the master baseline file and generate all univariate figures and tables
python scripts/univariate/univariate.py

# 2. Append BPRS subscale columns (Positive, Negative, Disorganized) to the simplified baseline file
python scripts/cleanup.py

# 3. Extract 10-year follow-up measurements and observations
python scripts/analysis.py

# 4. Run bivariate analyses
python scripts/bivariate/bivariate_recov.py      # Baseline vs. recovery outcome
python scripts/bivariate/bivariate_10y.py         # Longitudinal symptom change (Not Recovered)
python scripts/bivariate/bivariate_cross.py       # Cross-variable correlation matrix
python scripts/bivariate/attrition_analysis.py    # Attrition bias assessment

# 5. Run outlier detection on baseline clinical variables
python scripts/univariate/outlier_detection.py

# 6. Train and evaluate classification models
python scripts/models/diagnosis/lightgbm_diagnosis.py
python scripts/models/diagnosis/xgboost_diagnosis.py
```

---

## Internship Context

This project was completed as part of an internship in a clinical research setting. The work involved independently designing the full analysis pipeline from scratch, navigating real-world OMOP CDM data with its inconsistent coding conventions and substantial missingness, and producing results to a standard suitable for open-access publication as part of the VOLABIOS study.

The codebase reflects both the analytical decisions made throughout the internship and the iterative nature of working with longitudinal clinical data — including the debugging required to reconcile OMOP concept ID mappings, handle attrition in a principled way, and ensure strict data leakage prevention throughout the machine learning pipeline. It was a meaningful opportunity to apply statistical and ML methods to a real psychiatric research question with direct clinical relevance.

---

## License

Open access — VOLABIOS study. See project documentation for data use terms.
