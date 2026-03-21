import pandas as pd
import os

# ==========================================
# 1. SETUP & LOAD MASTER DATA
# ==========================================
input_path = 'data_processed/csv_files/master_baseline_comprehensive.csv'
output_dir = 'data_processed/outliers'
os.makedirs(output_dir, exist_ok=True)

if not os.path.exists(input_path):
    print(f"Error: Could not find {input_path}. Run univariate.py first.")
else:
    df = pd.read_csv(input_path)
    
    # Clinical variables of interest
    num_cols = [
        'baseline_age', 'BPRS_Total', 'SAPS_Total', 'SANS_Total', 
        'DUP_months', 'DUI_months', 'DAP_months', 'DAT_months'
    ]

    # ==========================================
    # 2. DETECTION LOGIC
    # ==========================================
    all_outlier_rows = []

    for col in num_cols:
        data = df[col].dropna()
        if len(data) == 0: continue
            
        Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
        IQR = Q3 - Q1
        
        # Thresholds
        reg_low, reg_high = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        ext_low, ext_high = Q1 - 3.0 * IQR, Q3 + 3.0 * IQR
        
        # Find Outliers
        outliers = df[(df[col] < reg_low) | (df[col] > reg_high)].copy()
        
        if not outliers.empty:
            outliers['Variable'] = col
            outliers['Value'] = outliers[col]
            outliers['Severity'] = outliers[col].apply(
                lambda x: 'Extreme' if (x < ext_low or x > ext_high) else 'Regular'
            )
            all_outlier_rows.append(outliers[['person_id', 'Variable', 'Value', 'Severity', 'diagnosis', 'recovery_status']])

    # ==========================================
    # 3. SEGMENTATION & FOLLOW-UP SORTING
    # ==========================================
    if all_outlier_rows:
        master_outliers = pd.concat(all_outlier_rows)

        # Identify Patients with at least one "Extreme" value
        extreme_ids = master_outliers[master_outliers['Severity'] == 'Extreme']['person_id'].unique()

        # Split into the two requested cohorts
        extreme_cohort = master_outliers[master_outliers['person_id'].isin(extreme_ids)].copy()
        regular_cohort = master_outliers[~master_outliers['person_id'].isin(extreme_ids)].copy()

        def apply_final_sort(target_df):
            # Create a helper flag: True if we have recovery data, False if 'Unknown'
            # (Matches the fillna('Unknown') from the univariate script)
            target_df['has_followup'] = ~target_df['recovery_status'].isin(['Unknown', None]) & target_df['recovery_status'].notna()
            
            # Sort order:
            # 1. has_followup (True/1 on top, False/0 on bottom)
            # 2. person_id (numerical ascending)
            # 3. Variable (alphabetical)
            sorted_df = target_df.sort_values(
                by=['has_followup', 'person_id', 'Variable'], 
                ascending=[False, True, True]
            )
            return sorted_df.drop(columns=['has_followup'])

        # Apply the sorting logic to both files
        extreme_cases_final = apply_final_sort(extreme_cohort)
        regular_cases_final = apply_final_sort(regular_cohort)

        # ==========================================
        # 4. SAVE OUTPUTS
        # ==========================================
        extreme_cases_final.to_csv(os.path.join(output_dir, 'extreme_cases_report.csv'), index=False)
        regular_cases_final.to_csv(os.path.join(output_dir, 'regular_cases_report.csv'), index=False)

        print(f"Success! Reports generated in {output_dir}/")
        print(f" - Extreme Cases File: {len(extreme_ids)} unique patients")
        print(f" - Regular Cases File: {regular_cases_final['person_id'].nunique()} unique patients")
    else:
        print("No outliers detected.")