import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel
import os

# 1. SETUP
output_dir = 'results/results_biv/pictures'
os.makedirs(output_dir, exist_ok=True)

df_base = pd.read_csv('data_processed/csv_files/master_baseline_comprehensive.csv')
df_follow = pd.read_csv('data_processed/csv_files/followup_10y_data.csv')

# Merge
df = pd.merge(df_base, df_follow, on='person_id', how='left')

# Normalize case and filter for Not Recovered
df['recovery_status_10y'] = df['recovery_status_10y'].str.capitalize()
df_not_rec = df[df['recovery_status_10y'] == 'Not recovered'].copy()

# Define pairs for longitudinal comparison (Baseline vs 10 Years)
pairs = [
    ('BPRS_Total', 'BPRS_Total_10y'),
    ('SAPS_Total', 'SAPS_Total_10y'),
    ('SANS_Total', 'SANS_Total_10y'),
    ('BPRS_Positive_Onset', 'BPRS_Positive_10y'),
    ('BPRS_Negative_Onset', 'BPRS_Negative_10y')
]

longitudinal_report = []

# 2. ANALYSIS LOOP
for base_var, follow_var in pairs:
    if base_var not in df_not_rec.columns or follow_var not in df_not_rec.columns:
        continue
    
    # Drop rows missing either the baseline or follow-up value
    temp = df_not_rec[[base_var, follow_var]].dropna()
    
    if len(temp) >= 2:
        
        stat, p = ttest_rel(temp[base_var], temp[follow_var])
        
        longitudinal_report.append({
            'Measurement': base_var.replace('_Onset', '').replace('_Total', ' Total'),
            'N': len(temp),
            'Mean_Baseline': round(temp[base_var].mean(), 2),
            'Mean_10y': round(temp[follow_var].mean(), 2),
            'Mean_Difference': round(temp[follow_var].mean() - temp[base_var].mean(), 2),
            'p_value': round(p, 4)
        })
        
        # Visualization: Point plot (Mean shift) + Swarm (All individuals)
        plt.figure(figsize=(8, 6))
        plot_df = temp.melt(var_name='Time', value_name='Score')
        plot_df['Time'] = plot_df['Time'].apply(lambda x: 'Baseline' if x == base_var else '10 Years')
        
        # Plot mean change
        sns.pointplot(data=plot_df, x='Time', y='Score', capsize=.1, errorbar='sd', color='red', markers="D")
        # Overlay individuals
        sns.swarmplot(data=plot_df, x='Time', y='Score', alpha=0.3, color='black', size=4)
        
        title_name = base_var.replace('_Onset', '').replace('_Total', ' Total')
        plt.title(f'10-Year Clinical Change: {title_name}\n(Group: Not Recovered, p={p:.4f})', fontweight='bold')
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/longitudinal_{base_var}_change.png')
        plt.close()

# 3. SAVE THE REPORT
pd.DataFrame(longitudinal_report).to_csv('results/results_biv_10y/longitudinal_stats_not_recovered.csv', index=False)