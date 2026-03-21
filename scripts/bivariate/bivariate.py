import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, chi2_contingency
import os

# ==========================================
# 1. SETUP & DATA LOADING
# ==========================================

input_file = 'data_processed/csv_files/master_baseline_comprehensive.csv'
output_dir = 'results/results_biv'

df = pd.read_csv(input_file)

# Prepare lists for systematic testing
num_cols = ['baseline_age', 'BPRS_Total', 'SAPS_Total', 'SANS_Total', 'DUP_months', 'DUI_months', 'DAP_months', 'DAT_months']
cat_cols = ['gender', 'diagnosis', 'education', 'socioeconomic_status', 'employment', 'marital_status', 'lives_with_parents', 'cannabis_use']

# ==========================================
# VERSION 1: RECOVERY PREDICTORS
# ==========================================
# Filter for patients who returned for follow-up
recovery_df = df[df['recovery_status'].isin(['Recovered', 'Not Recovered'])].copy()
print(f"Analyzing Version 1: {len(recovery_df)} patients with known outcomes...")

v1_stats = []

# A. Numerical Predictors (T-Tests & Boxplots)
for col in num_cols:
    g_rec = recovery_df[recovery_df['recovery_status'] == 'Recovered'][col].dropna()
    g_not = recovery_df[recovery_df['recovery_status'] == 'Not Recovered'][col].dropna()
    
    if len(g_rec) > 1 and len(g_not) > 1:
        t_stat, p_val = ttest_ind(g_rec, g_not)
        v1_stats.append({'Variable': col, 'Type': 'Numerical', 'p-value': round(p_val, 4)})
        
        plt.figure(figsize=(8, 5))
        sns.boxplot(x='recovery_status', y=col, data=recovery_df, palette='Set1', showmeans=True)
        plt.title(f'Baseline {col} vs. Recovery Status (p={p_val:.4f})')
        plt.savefig(os.path.join(output_dir, f'v1_recovery_vs_{col}.png'))
        plt.close()

# B. Categorical Predictors (Chi-Square & Bar Charts)
for col in cat_cols:
    contingency = pd.crosstab(recovery_df[col], recovery_df['recovery_status'])
    if not contingency.empty and contingency.size > 1:
        chi2, p_val, _, _ = chi2_contingency(contingency)
        v1_stats.append({'Variable': col, 'Type': 'Categorical', 'p-value': round(p_val, 4)})
        
        # Proportional Stacked Bar Chart
        prop_df = contingency.div(contingency.sum(axis=1), axis=0) * 100
        prop_df.plot(kind='bar', stacked=True, color=['#ff9999','#66b3ff'], figsize=(10, 6))
        plt.title(f'{col} Distribution by Outcome (p={p_val:.4f})')
        plt.ylabel('Percentage (%)')
        plt.legend(title='Outcome', bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'v1_recovery_vs_{col}.png'))
        plt.close()

pd.DataFrame(v1_stats).to_csv(os.path.join(output_dir, 'v1_statistical_results.csv'), index=False)

# ==========================================
# VERSION 2: STRUCTURAL FINDINGS
# ==========================================
print("Analyzing Version 2: Internal Variable Relationships...")

# A. Correlation Heatmap (Inter-Duration relationships)
plt.figure(figsize=(10, 8))
corr_matrix = df[num_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Baseline Correlation Matrix (Symptoms & Durations)')
plt.savefig(os.path.join(output_dir, 'v2_correlation_heatmap.png'))
plt.close()

# B. Activity (Work/Study) vs Symptom Severity
for symptom in ['SAPS_Total', 'SANS_Total']:
    plt.figure(figsize=(8, 5))
    sns.violinplot(x='employment', y=symptom, data=df, palette='Set2')
    plt.title(f'{symptom} Severity by Activity Status')
    plt.savefig(os.path.join(output_dir, f'v2_activity_vs_{symptom}.png'))
    plt.close()

# C. Marital Status by Diagnosis Type
plt.figure(figsize=(12, 6))
sns.countplot(x='diagnosis', hue='marital_status', data=df, palette='pastel')
plt.title('Marital Status Distribution across Diagnostic Groups')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'v2_diagnosis_vs_marital.png'))
plt.close()

print(f"Bivariate Analysis Complete. Charts and tables exported to: {output_dir}/")