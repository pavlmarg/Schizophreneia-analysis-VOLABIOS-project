import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, chi2_contingency
import os

# 1. SETUP & DATA LOADING
output_dir = 'results/results_biv/pictures'

# Load datasets
df_base = pd.read_csv('data_files/data_processed/csv_files/master_baseline_comprehensive.csv')
df_follow = pd.read_csv('data_files/data_processed/csv_files/followup_10y_data.csv')

# Merge
df = pd.merge(df_base, df_follow[['person_id', 'recovery_status_10y']], on='person_id', how='left')

df['recovery_status_10y'] = df['recovery_status_10y'].replace({'Not recovered': 'Not Recovered'})
df_ana = df[df['recovery_status_10y'].isin(['Recovered', 'Not Recovered'])].copy()


memos = {
    'baseline_age': "Higher = Older at first episode.",
    'BPRS_Total': "Higher = More severe general psychiatric symptoms.",
    'BPRS_Positive_Onset': "Higher = More severe hallucinations and delusions.",
    'BPRS_Negative_Onset': "Higher = More severe emotional/social withdrawal.",
    'BPRS_Disorganized_Onset': "Higher = More severe thought disorder/bizarre behavior.",
    'DUI_months': "Higher = Longer total duration of untreated illness.",
    'DUP_months': "Higher = Longer duration of untreated psychosis.",
    'DAP_months': "Higher = More time spent in active psychosis before baseline.",
    'DAT_months': "Higher = Longer time on meds before baseline measurement.",
    'SANS_Total': "Higher = More negative symptoms (SANS scale).",
    'SAPS_Total': "Higher = More positive symptoms (SAPS scale)."
}

num_vars = list(memos.keys())
cat_vars = [
    'gender', 'cannabis_use', 'lives_with_parents', 'family_hx_psychosis', 
    'hospital_admission', 'education', 'socioeconomic_status', 
    'employment', 'marital_status', 'diagnosis'
]

statistical_summary = []

# ==========================================
# NUMERICAL ANALYSIS
# ==========================================
for var in num_vars:
    if var not in df_ana.columns: continue
    
    # Split groups
    rec = df_ana[df_ana['recovery_status_10y'] == 'Recovered'][var].dropna()
    not_rec = df_ana[df_ana['recovery_status_10y'] == 'Not Recovered'][var].dropna()
    
    if len(rec) > 1 and len(not_rec) > 1:
        stat, p = ttest_ind(rec, not_rec)
        
        # Add to summary list
        statistical_summary.append({
            'Variable': var,
            'Type': 'Numerical',
            'Recovered_N': len(rec),
            'Recovered_Mean': round(rec.mean(), 2),
            'Recovered_SD': round(rec.std(), 2),
            'NotRecovered_N': len(not_rec),
            'NotRecovered_Mean': round(not_rec.mean(), 2),
            'NotRecovered_SD': round(not_rec.std(), 2),
            'p_value': round(p, 4)
        })
        
        # Visualization if p < 0.3
        if p < 0.3:
            plt.figure(figsize=(9, 7))
            sns.violinplot(x='recovery_status_10y', y=var, data=df_ana, inner=None, palette='pastel', alpha=0.4)
            sns.swarmplot(x='recovery_status_10y', y=var, data=df_ana, color="0.3", size=4)
            
            plt.title(f'Baseline {var} vs. 10y Outcome\n', fontsize=14, fontweight='bold')
            plt.xlabel("Recovery Status", fontsize=12)
            plt.ylabel(f"{var} Score", fontsize=12)
            
            memo_text = f"ANALYSIS MEMO: {memos.get(var, '')}"
            plt.figtext(0.5, 0.02, memo_text, wrap=True, horizontalalignment='center', fontsize=10, 
                        bbox={'facecolor': 'orange', 'alpha': 0.1, 'pad': 5})
            
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            plt.savefig(f'{output_dir}/{var}_bivariate.png')
            plt.close()

# ==========================================
# CATEGORICAL ANALYSIS
# ==========================================
for var in cat_vars:
    if var not in df_ana.columns: continue
    
    ct = pd.crosstab(df_ana[var], df_ana['recovery_status_10y'])
    
    if not ct.empty and ct.size > 1:
        chi2, p, _, _ = chi2_contingency(ct)
        
        cat_details = ""
        for index, row in ct.iterrows():
            total_cat = row.sum()
            rec_p = (row['Recovered'] / total_cat * 100) if total_cat > 0 else 0
            cat_details += f"{index}: [Rec: {row['Recovered']} ({rec_p:.1f}%), Not: {row['Not Recovered']}] | "
            
        statistical_summary.append({
            'Variable': var,
            'Type': 'Categorical',
            'Recovered_N': ct['Recovered'].sum(),
            'NotRecovered_N': ct['Not Recovered'].sum(),
            'Details (Count/%)': cat_details,
            'p_value': round(p, 4)
        })
        
        # Visualization if p < 0.3
        if p < 0.3:
            pct_df = ct.div(ct.sum(axis=1), axis=0) * 100
            ax = pct_df.plot(kind='bar', stacked=True, figsize=(10, 7), color=['#e74c3c','#3498db'])
            
            for patch in ax.patches:
                width, height = patch.get_width(), patch.get_height()
                if height > 5:
                    ax.text(patch.get_x() + width/2, patch.get_y() + height/2, f'{height:.1f}%', 
                            ha='center', va='center', color='white', fontweight='bold')
            
            plt.title(f'{var} vs. Recovery Status\n', fontweight='bold', fontsize=14)
            plt.ylabel('Group Proportion (%)')
            plt.legend(title="Outcome", loc='upper left', bbox_to_anchor=(1, 1))
            plt.tight_layout()
            plt.savefig(f'{output_dir}/{var}_bivariate.png')
            plt.close()

# SAVE COMPREHENSIVE REPORT
final_stats = pd.DataFrame(statistical_summary)
final_stats.to_csv('results/results_biv/tables/recovery_bivariate_report.csv', index=False)

print("Comprehensive Analysis Script execution finished.")
print(f"Report and {len(os.listdir(output_dir))} pictures generated.")