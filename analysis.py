import pandas as pd
import os

# 1. LOAD DATA
meas = pd.read_csv('data_processed/csv_files/measurement.csv')
obs = pd.read_csv('data_processed/csv_files/observation.csv')

# 2. EXTRACT 10-YEAR MEASUREMENTS
# We filter by the source strings we found: 'bprs10y', 'saps10y', etc.
meas_10y = meas[meas['measurement_source_value'].str.contains('10y', case=False, na=False)].copy()

# Map source values to clean column names
meas_map = {
    'bprs10y': 'BPRS_Total_10y',
    'saps10y': 'SAPS_Total_10y',
    'sans10y': 'SANS_Total_10y',
    'positiva10y': 'BPRS_Positive_10y',
    'negativa10y': 'BPRS_Negative_10y'
}
meas_10y['metric'] = meas_10y['measurement_source_value'].map(meas_map)

# Pivot to patient-level (Using index='person_id' safely)
df_meas_10y = meas_10y.pivot_table(
    index='person_id', 
    columns='metric', 
    values='value_as_number'
).reset_index()

# 3. EXTRACT 10-YEAR OBSERVATIONS
# Capture active status and recovery status
obs_10y = obs[obs['observation_source_value'].str.contains('10y|10m', case=False, na=False)].copy()

# A. Active Status at 10 Years
active_10y = obs_10y[obs_10y['observation_source_value'] == 'active10y'].copy()
# 9181.0 = Active, 9177.0 = Inactive
active_10y['active_10y'] = active_10y['value_as_concept_id'].map({9181.0: 'Active', 9177.0: 'Inactive'})

# B. Recovery Status (Often labeled recovery10m in the source)
recovery_10y = obs_10y[obs_10y['observation_source_value'] == 'recovery10m'].copy()
recovery_10y = recovery_10y.rename(columns={'value_as_string': 'recovery_status_10y'})

# 4. MERGE FOLLOW-UP DATA
followup_final = pd.merge(df_meas_10y, active_10y[['person_id', 'active_10y']], on='person_id', how='outer')
followup_final = pd.merge(followup_final, recovery_10y[['person_id', 'recovery_status_10y']], on='person_id', how='outer')

# 5. SAVE TO CSV
os.makedirs('data_processed', exist_ok=True)
followup_final.to_csv('data_processed/followup_10y_data.csv', index=False)

print(f"Extraction successful. Found follow-up records for {followup_final['person_id'].nunique()} patients.")
print("Saved to: data_processed/followup_10y_data.csv")