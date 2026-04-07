import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, chi2_contingency
import os

# ==========================================
# 1. SETUP & DATA LOADING
# ==========================================
output_dir = 'results/attrition_analysis/pictures'
os.makedirs(output_dir, exist_ok=True)

# Load the master baseline file
df = pd.read_csv('data_files/data_processed/csv_files/master_baseline_comprehensive.csv')

# Create a clear binary flag for attrition
df['returned_flag'] = df['attrition_status'].apply(
    lambda x: 'Returned' if x == 'Returned for Follow-up' else 'Did Not Return'
)

# Define the variables to test
num_vars = [
    'baseline_age', 'BPRS_Total', 'BPRS_Positive_Onset', 'BPRS_Negative_Onset',
    'BPRS_Disorganized_Onset', 'DAP_months', 'DAT_months', 'DUI_months',
    'DUP_months', 'SANS_Total', 'SAPS_Total'
]

cat_vars = [
    'gender', 'cannabis_use', 'lives_with_parents', 'family_hx_psychosis',
    'hospital_admission', 'education', 'socioeconomic_status', 'employment',
    'marital_status', 'diagnosis'
]

attrition_results = []

# ==========================================
# 2. NUMERICAL VARIABLES (Mann-Whitney U)
# ==========================================
for var in num_vars:
    if var not in df.columns: continue
    
    group_ret = df[df['returned_flag'] == 'Returned'][var].dropna()
    group_not = df[df['returned_flag'] == 'Did Not Return'][var].dropna()
    
    if len(group_ret) > 1 and len(group_not) > 1:
        # Statistical Test
        stat, p = mannwhitneyu(group_ret, group_not, alternative='two-sided')
        
        attrition_results.append({
            'Variable': var,
            'Type': 'Numerical',
            'Test': 'Mann-Whitney U',
            'Returned_Mean': round(group_ret.mean(), 2),
            'Not_Returned_Mean': round(group_not.mean(), 2),
            'Returned_Median': round(group_ret.median(), 2),
            'Not_Returned_Median': round(group_not.median(), 2),
            'Returned_Counts_Pct': 'N/A',      
            'Not_Returned_Counts_Pct': 'N/A', 
            'p_value': round(p, 4)
        })
        
        # Visualize ONLY if p < 0.3
        if p < 0.3:
            plt.figure(figsize=(8, 6))
            sns.violinplot(x='returned_flag', y=var, data=df, inner=None, palette='pastel', alpha=0.5)
            sns.swarmplot(x='returned_flag', y=var, data=df, color="0.2", size=4, alpha=0.7)
            
            plt.title(f'Baseline {var.replace("_", " ")}\nby Attrition Status (p={p:.4f})', fontweight='bold', fontsize=14)
            plt.xlabel("Attrition Status", fontsize=12)
            plt.ylabel(var.replace("_", " "), fontsize=12)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/attrition_num_{var}.png', dpi=300)
            plt.close()

# ==========================================
# 3. CATEGORICAL VARIABLES (Chi-Square)
# ==========================================
for var in cat_vars:
    if var not in df.columns: continue
    
    contingency = pd.crosstab(df[var], df['returned_flag'])
    if contingency.size > 1 and not contingency.empty:
        # Statistical Test
        chi2, p, dof, ex = chi2_contingency(contingency)
        
        pct_df = contingency.div(contingency.sum(axis=1), axis=0) * 100
        ret_details = []
        not_ret_details = []
        
        for idx in contingency.index:
            r_count = contingency.loc[idx, 'Returned'] if 'Returned' in contingency.columns else 0
            nr_count = contingency.loc[idx, 'Did Not Return'] if 'Did Not Return' in contingency.columns else 0
            r_pct = pct_df.loc[idx, 'Returned'] if 'Returned' in pct_df.columns else 0
            nr_pct = pct_df.loc[idx, 'Did Not Return'] if 'Did Not Return' in pct_df.columns else 0
            
            # Format nicely as "Category: Count (Percentage%)"
            ret_details.append(f"{idx}: {r_count} ({r_pct:.1f}%)")
            not_ret_details.append(f"{idx}: {nr_count} ({nr_pct:.1f}%)")
        
        attrition_results.append({
            'Variable': var,
            'Type': 'Categorical',
            'Test': 'Chi-Square',
            'Returned_Mean': 'N/A',
            'Not_Returned_Mean': 'N/A',
            'Returned_Median': 'N/A',
            'Not_Returned_Median': 'N/A',
            'Returned_Counts_Pct': " | ".join(ret_details),     
            'Not_Returned_Counts_Pct': " | ".join(not_ret_details), 
            'p_value': round(p, 4)
        })
        
        # Visualize ONLY if p < 0.3
        if p < 0.3:
            pct_df = contingency.div(contingency.sum(axis=1), axis=0) * 100
            
            plt.figure(figsize=(9, 6))
            ax = pct_df.plot(kind='bar', stacked=True, color=['#e74c3c', '#2ecc71'], ax=plt.gca())
            
            # Add percentage labels to the bars
            for patch in ax.patches:
                width, height = patch.get_width(), patch.get_height()
                x, y = patch.get_xy()
                if height > 5:
                    ax.text(x + width/2, y + height/2, f'{height:.1f}%',
                            ha='center', va='center', color='white', fontweight='bold')
            
            plt.title(f'Attrition Rate by {var.replace("_", " ").title()}\n(p={p:.4f})', fontweight='bold', fontsize=14)
            plt.ylabel('Percentage within Group (%)', fontsize=12)
            plt.xlabel(var.replace("_", " ").title(), fontsize=12)
            plt.xticks(rotation=0)
            plt.legend(title='Attrition Status', bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/attrition_cat_{var}.png', dpi=300)
            plt.close()

# ==========================================
# 4. SAVE SUMMARY CSV
# ==========================================
attrition_stats_df = pd.DataFrame(attrition_results).sort_values('p_value')
attrition_stats_df.to_csv(f'{output_dir}/../attrition_nonparametric_comparison.csv', index=False)

print("All-in-one attrition analysis complete!")
print(f"Full CSV saved to results/attrition_analysis/attrition_nonparametric_comparison.csv")
print(f"Plots for variables with p < 0.3 saved to {output_dir}")