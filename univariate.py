import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore') # Keeps the console output clean

# ==========================================
# 1. SETUP & PATHS
# ==========================================
data_dir = 'data_files'
processed_dir = 'data_processed/results_univ'
os.makedirs(processed_dir, exist_ok=True)

print("Starting the Ultimate Single-File Univariate Pipeline...\n")

# ==========================================
# 2. LOAD RAW DATA
# ==========================================
person = pd.read_csv(os.path.join(data_dir, 'person.csv'))
visit = pd.read_csv(os.path.join(data_dir, 'visit_occurrence.csv'))
meas = pd.read_csv(os.path.join(data_dir, 'measurement.csv'))
obs = pd.read_csv(os.path.join(data_dir, 'observation.csv'))
cond = pd.read_csv(os.path.join(data_dir, 'condition_occurrence.csv'))
death = pd.read_csv(os.path.join(data_dir, 'death.csv')) 

# ==========================================
# 3. MASSIVE DATA EXTRACTION (ALL 307 PATIENTS)
# ==========================================
# A. Demographics
visit['visit_start_date'] = pd.to_datetime(visit['visit_start_date'])
first_visits = visit.sort_values('visit_start_date').groupby('person_id').first().reset_index()

df = pd.merge(person[['person_id', 'gender_concept_id', 'year_of_birth']], 
              first_visits[['person_id', 'visit_start_date']], on='person_id')
df['gender'] = df['gender_concept_id'].map({8507: 'Male', 8532: 'Female'})
df['baseline_age'] = df['visit_start_date'].dt.year - df['year_of_birth']

# B. Numerical Metrics (Symptoms & Durations)
onset_visits = visit[visit['visit_source_value'].str.contains('ONSET', case=False, na=False)][['visit_occurrence_id', 'person_id']]
baseline_meas = pd.merge(meas, onset_visits, on=['visit_occurrence_id', 'person_id'])

concept_map = {
    4155657: 'BPRS_Total', 40219529: 'SAPS_Total', 40219484: 'SANS_Total',
    2000000005: 'DUP_months', 2000000006: 'DUI_months', 
    2000000007: 'DAP_months', 2000000008: 'DAT_months'
}
measurements = baseline_meas[baseline_meas['measurement_concept_id'].isin(concept_map.keys())].copy()
measurements['metric_name'] = measurements['measurement_concept_id'].map(concept_map)
df = pd.merge(df, measurements.pivot_table(index='person_id', columns='metric_name', values='value_as_number').reset_index(), on='person_id', how='left')

# C. Categorical Observations (ULTIMATE BULLETPROOF HELPER)
def extract_obs(pattern, col_name, v_map=None, use_string_col=False):
    temp = obs[obs['observation_source_value'].str.contains(pattern, case=False, na=False)].copy()
    
    if use_string_col:
        # Pull answer directly from value_as_string (Used for Socioeconomic Status)
        if v_map:
            temp[col_name] = temp['value_as_string'].astype(str).str.lower().str.strip().map(v_map)
        else:
            temp[col_name] = temp['value_as_string']
    elif v_map:
        # 1. Try mapping by numerical Concept ID (e.g., 8715)
        numeric_concepts = pd.to_numeric(temp['value_as_concept_id'], errors='coerce')
        num_map = {float(k): v for k, v in v_map.items() if isinstance(k, (int, float))}
        mapped_num = numeric_concepts.map(num_map)
        
        # 2. Try mapping by the raw source string label (e.g., 'hospita-1.0')
        str_source = temp['observation_source_value'].astype(str).str.lower().str.strip()
        str_map = {str(k).lower().strip(): v for k, v in v_map.items()}
        mapped_str = str_source.map(str_map)
        
        # 3. Combine them: String catches it if Concept ID is missing
        temp[col_name] = mapped_num.fillna(mapped_str)
    else:
        temp[col_name] = 'Yes' 
        
    temp = temp.dropna(subset=[col_name])
    return temp[['person_id', col_name]].drop_duplicates(subset=['person_id'])

# Extracting all flags with double-mapping (ID + String) for maximum reliability
flags = [
    extract_obs('cannabinar', 'cannabis_use'),
    extract_obs('livparnt', 'lives_with_parents'),
    extract_obs('fampsic', 'family_hx_psychosis'),
    
    # Hospitalization: Correctly uses numerical Concept IDs
    extract_obs('hospita', 'hospital_admission', {8715.0: 'Admitted', 44792129.0: 'Avoided', 'hospita-1.0': 'Admitted', 'hospita-2.0': 'Avoided'}),
    
    # Education: Correctly handles IDs for Primary/High School
    extract_obs('educlvl', 'education', {1620732.0: 'Primary School', 1620880.0: 'High School', 'educlvl-1.0': 'Primary School', 'educlvl-2.0': 'High School'}),
    
    # SES: Uses the exact lowercase strings found in the 'value_as_string' field
    extract_obs('seclvl', 'socioeconomic_status', {'low': 'Low', 'medium or higher': 'Medium/Higher'}, use_string_col=True),
    
    # Employment
    extract_obs('active0', 'employment', {9181.0: 'Active (Work/Study)', 9177.0: 'Inactive/Other'}),
    
    extract_obs('single', 'marital_status', {45879879.0: 'Single', 45876756.0: 'Married', 'single-1.0': 'Single', 'single-2.0': 'Married'})
]

for flag_df in flags:
    df = pd.merge(df, flag_df, on='person_id', how='left')

# Clean up binary NaNs
df.fillna({
    'cannabis_use': 'No', 'marital_status': 'Unknown', 
    'lives_with_parents': 'No', 'family_hx_psychosis': 'No', 
    'hospital_admission': 'Unknown', 'socioeconomic_status': 'Unknown',
    'education': 'Unknown', 'employment': 'Unknown'
}, inplace=True)

# D. Baseline Diagnosis
cond_map = {
    435783: 'Schizophrenia', 4182683: 'Brief reactive psychosis', 4101149: 'Non-organic psychosis',
    444434: 'Schizophreniform', 4286201: 'Schizoaffective', 48500005: 'Delusional',
    436665: 'Bipolar', 436073: 'Psychotic disorder'
}
diag_cond = cond[['person_id', 'condition_concept_id']].drop_duplicates(subset=['person_id']).copy()
diag_cond['diagnosis'] = diag_cond['condition_concept_id'].map(cond_map)
df = pd.merge(df, diag_cond[['person_id', 'diagnosis']], on='person_id', how='left')

# E. 10-YEAR OUTCOMES & ATTRITION 
recovery_obs = obs[obs['observation_source_value'] == 'recovery10m'][['person_id', 'value_as_string']]
recovery_obs = recovery_obs.rename(columns={'value_as_string': 'recovery_status'})
deceased_ids = death['person_id'].unique()

def determine_attrition(row):
    if pd.notna(row['recovery_status']):
        return 'Returned for Follow-up'
    elif row['person_id'] in deceased_ids:
        return 'Deceased'
    else:
        return 'Withdrew / Lost to Follow-up'

df = pd.merge(df, recovery_obs, on='person_id', how='left')
df['attrition_status'] = df.apply(determine_attrition, axis=1)

# Save Master File
df.to_csv(os.path.join(processed_dir, 'master_baseline_comprehensive.csv'), index=False)

# ==========================================
# 4. EXHAUSTIVE STATISTICAL REPORT
# ==========================================
print("=========================================================")
print("  EXHAUSTIVE UNIVARIATE STATISTICAL REPORT (n=307)       ")
print("=========================================================\n")

categorical_vars = [
    'gender', 'diagnosis', 'education', 'socioeconomic_status', 'employment', 
    'marital_status', 'lives_with_parents', 'family_hx_psychosis', 
    'hospital_admission', 'cannabis_use', 'attrition_status', 'recovery_status'
]

print(">>> CATEGORICAL VARIABLES (Counts & Percentages) <<<\n")
for col in categorical_vars:
    print(f"--- {col.upper()} ---")
    counts = df[col].value_counts(dropna=False)
    pcts = df[col].value_counts(normalize=True, dropna=False) * 100
    stats_df = pd.DataFrame({'Count': counts, 'Percentage (%)': pcts.round(1)})
    print(stats_df)
    print("-" * 40)

print("\n>>> NUMERICAL VARIABLES (Distribution Metrics) <<<\n")
num_cols = ['baseline_age', 'BPRS_Total', 'SAPS_Total', 'SANS_Total', 'DUP_months', 'DUI_months', 'DAP_months', 'DAT_months']
num_stats = df[num_cols].agg(['count', 'mean', 'median', 'std', 'var', 'min', 'max']).T.round(2)
print(num_stats)
print("\n=========================================================\n")

# ==========================================
# 5. GENERATE PLOTS (Max 2 Subplots Per Figure)
# ==========================================
sns.set_theme(style="whitegrid")
mean_props = {"marker":"D", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":8}

def save_fig(fig, filename):
    plt.tight_layout()
    fig.savefig(os.path.join(processed_dir, filename), dpi=300)
    plt.close(fig)

print("Generating 10 comprehensive figures...")

# Fig 1: Age & Gender
fig1, ax1 = plt.subplots(1, 2, figsize=(14, 5))
fig1.suptitle('Figure 1: Core Demographics', fontweight='bold')
ax1[0].pie(df['gender'].value_counts(), labels=df['gender'].value_counts().index, autopct='%1.1f%%', colors=['#66b3ff','#ff9999'])
ax1[0].set_title('A. Gender Distribution')
sns.histplot(df['baseline_age'], bins=15, kde=True, ax=ax1[1], color='purple')
ax1[1].set_title('B. Age at Onset Distribution')
save_fig(fig1, 'uni_fig1_demographics.png')

# Fig 2: Education & Socioeconomic Level
fig2, ax2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle('Figure 2: Education and Socioeconomic Status', fontweight='bold')
sns.countplot(x='education', data=df, ax=ax2[0], palette='Set2')
ax2[0].set_title('A. Educational Level')
sns.countplot(x='socioeconomic_status', data=df, ax=ax2[1], palette='Set2', order=['Low', 'Medium/Higher', 'Unknown'])
ax2[1].set_title('B. Socioeconomic Status (SES)')
save_fig(fig2, 'uni_fig2_education_ses.png')

# Fig 3: Employment & Marital Status
fig3, ax3 = plt.subplots(1, 2, figsize=(14, 5))
fig3.suptitle('Figure 3: Employment and Social Status', fontweight='bold')
sns.countplot(x='employment', data=df, ax=ax3[0], palette='Set3')
ax3[0].set_title('A. Employment Status')
sns.countplot(x='marital_status', data=df, ax=ax3[1], palette='pastel')
ax3[1].set_title('B. Marital Status')
save_fig(fig3, 'uni_fig3_employment_marital.png')

# Fig 4: Living Situation & Clinical History Predictors
fig4, ax4 = plt.subplots(1, 2, figsize=(14, 5))
fig4.suptitle('Figure 4: Living Situation & Family History', fontweight='bold')
sns.countplot(x='lives_with_parents', data=df, ax=ax4[0], palette='pastel')
ax4[0].set_title('A. Living with Parents')
sns.countplot(x='family_hx_psychosis', data=df, ax=ax4[1], palette=['#ff9999', '#99ff99'])
ax4[1].set_title('B. Family History of Psychosis')
save_fig(fig4, 'uni_fig4_living_family.png')

# Fig 5: Diagnosis & Cannabis
fig5, ax5 = plt.subplots(1, 2, figsize=(15, 6))
fig5.suptitle('Figure 5: Baseline Presentation', fontweight='bold')
sns.countplot(y='diagnosis', data=df, ax=ax5[0], palette='muted', order=df['diagnosis'].value_counts().index)
ax5[0].set_title('A. Clinical Diagnosis')
sns.countplot(x='cannabis_use', data=df, ax=ax5[1], palette=['#77dd77', '#ffb347'])
ax5[1].set_title('B. Cannabis Use at Onset')
save_fig(fig5, 'uni_fig5_diag_cannabis.png')

# Fig 6: Symptom Severity I
fig6, ax6 = plt.subplots(1, 2, figsize=(14, 5))
fig6.suptitle('Figure 6: Symptom Severity I', fontweight='bold')
sns.boxplot(y=df['BPRS_Total'], ax=ax6[0], color='coral', showmeans=True, meanprops=mean_props)
ax6[0].set_title('A. BPRS Total (General Severity)')
sns.boxplot(y=df['SAPS_Total'], ax=ax6[1], color='lightgreen', showmeans=True, meanprops=mean_props)
ax6[1].set_title('B. SAPS Total (Positive Symptoms)')
save_fig(fig6, 'uni_fig6_symptoms_1.png')

# Fig 7: Symptom Severity II & Psychosis Delay
fig7, ax7 = plt.subplots(1, 2, figsize=(14, 5))
fig7.suptitle('Figure 7: Symptom Severity II & Psychosis Delay', fontweight='bold')
sns.boxplot(y=df['SANS_Total'], ax=ax7[0], color='lightblue', showmeans=True, meanprops=mean_props)
ax7[0].set_title('A. SANS Total (Negative Symptoms)')
sns.histplot(df['DUP_months'].dropna(), bins=30, kde=True, ax=ax7[1], color='teal')
ax7[1].set_title('B. Duration of Untreated Psychosis (DUP)')
save_fig(fig7, 'uni_fig7_symptoms_dup.png')

# Fig 8: Illness Delays
fig8, ax8 = plt.subplots(1, 2, figsize=(14, 5))
fig8.suptitle('Figure 8: Illness Progression Delays (Months)', fontweight='bold')
sns.histplot(df['DUI_months'].dropna(), bins=30, kde=True, ax=ax8[0], color='indigo')
ax8[0].set_title('A. Duration of Untreated Illness (DUI)')
sns.histplot(df['DAP_months'].dropna(), bins=30, kde=True, ax=ax8[1], color='darkred')
ax8[1].set_title('B. Duration of Active Psychosis (DAP)')
save_fig(fig8, 'uni_fig8_dui_dap.png')

# Fig 9: Treatment Metric & Hospitalization
fig9, ax9 = plt.subplots(1, 2, figsize=(14, 5)) 
fig9.suptitle('Figure 9: Antipsychotic Treatment & Hospitalization', fontweight='bold')
sns.histplot(df['DAT_months'].dropna(), bins=30, kde=True, ax=ax9[0], color='goldenrod')
ax9[0].set_title('A. Duration of Antipsychotic Treatment (DAT)')
sns.countplot(x='hospital_admission', data=df, ax=ax9[1], palette='Set3')
ax9[1].set_title('B. Required Hospital Admission at Onset')
save_fig(fig9, 'uni_fig9_dat_hospita.png')

# Fig 10: OUTCOMES & ATTRITION
fig10, ax10 = plt.subplots(1, 2, figsize=(15, 6))
fig10.suptitle('Figure 10: 10-Year Study Outcomes', fontweight='bold')

attrition_counts = df['attrition_status'].value_counts()
ax10[0].pie(attrition_counts, labels=attrition_counts.index, autopct='%1.1f%%', colors=['#66b3ff', '#ffcc99', '#ff9999'], startangle=140)
ax10[0].set_title('A. Cohort Attrition (All 307 Patients)')

sns.countplot(x='recovery_status', data=df.dropna(subset=['recovery_status']), ax=ax10[1], palette='Set2')
ax10[1].set_title('B. Recovery Status (Returned Patients Only)')
for p in ax10[1].patches:
    ax10[1].annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom')

save_fig(fig10, 'uni_fig10_outcomes.png')

print("Success! 10 figures have been generated in 'data_processed/results_univ'.")

# ==========================================
# 6. EXPORT STATISTICAL TABLES
# ==========================================
output_tables_dir = 'data_processed/results_univ/tables'
os.makedirs(output_tables_dir, exist_ok=True)

cat_list = []
for var in categorical_vars:
    counts = df[var].value_counts(dropna=False)
    pcts = df[var].value_counts(normalize=True, dropna=False) * 100
    
    summary = pd.DataFrame({
        'Variable': var,
        'Category': counts.index,
        'Count (n)': counts.values,
        'Percentage (%)': pcts.values.round(2)
    })
    cat_list.append(summary)

final_categorical_table = pd.concat(cat_list)
final_categorical_table.to_csv(os.path.join(output_tables_dir, 'table_1_categorical.csv'), index=False)

# B. Numerical Table (Mean, Std Dev, Min, Max, etc.)
# The .describe() method provides the N, mean, std, min, 25%, 50%, 75%, and max
numerical_summary = df[num_cols].describe().T.round(2)

# Adding Variance specifically as it is often required for clinical variance checks
numerical_summary['variance'] = df[num_cols].var().round(2)

numerical_summary.to_csv(os.path.join(output_tables_dir, 'table_2_numerical.csv'))

print(f"\nStatistical tables have been exported to: {output_tables_dir}/")