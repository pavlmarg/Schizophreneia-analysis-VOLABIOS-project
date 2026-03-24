import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import os

# 1. SETUP
output_dir = 'results/results_biv/pictures/cross'


df_base = pd.read_csv('data_processed/csv_files/master_baseline_comprehensive.csv')
df_follow = pd.read_csv('data_processed/csv_files/followup_10y_data.csv')

# Merge
df = pd.merge(df_base, df_follow, on='person_id', how='left')

# 2. SELECT NON-REDUNDANT VARIABLES
selected_num = [
    'baseline_age',
    'DUP_months',
    'DAT_months',
    'SAPS_Total',
    'SANS_Total',
    'SAPS_Total_10y',
    'SANS_Total_10y',
    'DAS_Global_10y'
]

selected_cat = [
    'gender',
    'cannabis_use',
    'hospital_admission',
    'diagnosis',
    'recovery_status_10y',
    'active_10y'
]

# Ensure they exist
selected_num = [c for c in selected_num if c in df.columns]
selected_cat = [c for c in selected_cat if c in df.columns]

# Encoding categories for the matrix
df_encoded = df.copy()
for col in selected_cat:
    df_encoded[col] = df_encoded[col].astype('category').cat.codes
    df_encoded.loc[df_encoded[col] == -1, col] = np.nan

# 3. GENERATE CORRELATION MATRIX
all_selected = selected_num + selected_cat
corr_matrix = df_encoded[all_selected].corr(method='pearson')
corr_matrix.to_csv('results/results_biv/tables/filtered_correlation_matrix.csv')

# Plot Heatmap
plt.figure(figsize=(16, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
            fmt=".2f", linewidths=0.5, square=True)
plt.title('Filtered Clinical Correlation Matrix (Baseline vs 10-Year Outcomes)', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig('results/results_biv/pictures/cross/heatmap_curated.png')
plt.close()

# 4. PLOT IMPORTANT INTERACTIONS (0.3 < |r| < 0.9)
plotted_count = 0
for i in range(len(all_selected)):
    for j in range(i + 1, len(all_selected)):
        col1 = all_selected[i]
        col2 = all_selected[j]
        r = corr_matrix.loc[col1, col2]

        if 0.1 < abs(r) < 0.9:
            temp = df[[col1, col2]].dropna()
            if len(temp) < 10: continue

            plt.figure(figsize=(8, 6))

            if col1 in selected_num and col2 in selected_num:
                sns.regplot(x=col1, y=col2, data=temp, scatter_kws={'alpha':0.4}, line_kws={'color':'red'})
            elif (col1 in selected_cat and col2 in selected_num) or (col1 in selected_num and col2 in selected_cat):
                cat = col1 if col1 in selected_cat else col2
                num = col2 if col1 in selected_cat else col1
                sns.violinplot(x=cat, y=num, data=temp, palette='muted')
                plt.xticks(rotation=45)
            else: # Cat vs Cat
                ct = pd.crosstab(temp[col1], temp[col2])
                ct.div(ct.sum(axis=1), axis=0).plot(kind='bar', stacked=True, ax=plt.gca())
                plt.legend(title=col2, bbox_to_anchor=(1.05, 1))

            plt.title(f'Interaction: {col1} vs {col2}\nCorrelation r={r:.3f}', fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/int_{col1}_vs_{col2}.png")
            plt.close()
            plotted_count += 1

print(f"Curated analysis complete. Heatmap and {plotted_count} plots generated.")