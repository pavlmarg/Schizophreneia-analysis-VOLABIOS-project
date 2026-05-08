import pandas as pd
import os

# 1. Φόρτωση αρχείων 
master_df = pd.read_csv('data_files/data_processed/csv_files/baseline_simplified_data.csv') 
meas_df = pd.read_csv('data_files/data_processed/csv_files/measurement.csv') 

# 2. Ορίζουμε τι ψάχνουμε με βάση το SOURCE VALUE
source_map = {
    'positive0': 'BPRS_Positive_Onset',
    'negative0': 'BPRS_Negative_Onset',
    'disorgan0': 'BPRS_Disorganized_Onset'
}

meas_df['source_clean'] = meas_df['measurement_source_value'].astype(str).str.lower().str.strip()

# 3. Φιλτράρισμα και Pivot
target_meas = meas_df[meas_df['source_clean'].isin(source_map.keys())].copy()
target_meas['metric_name'] = target_meas['source_clean'].map(source_map)

new_columns = target_meas.pivot_table(
    index='person_id', 
    columns='metric_name', 
    values='value_as_number'
).reset_index()

# 4. Ενώνουμε τις νέες στήλες (μπαίνουν στο τέλος)
final_df = pd.merge(master_df, new_columns, on='person_id', how='left')

# ---------------------------------------------------------
# 5. ΑΝΑΔΙΑΤΑΞΗ ΣΤΗΛΩΝ (Δυναμική και ασφαλής μέθοδος)
# ---------------------------------------------------------
cols = final_df.columns.tolist()

# Βρίσκουμε ΠΟΙΕΣ από τις 3 νέες στήλες προστέθηκαν ΠΡΑΓΜΑΤΙΚΑ στο τελικό αρχείο
added_cols = [col_name for col_name in source_map.values() if col_name in cols]

# Τις αφαιρούμε από το τέλος της λίστας
for col in added_cols:
    cols.remove(col)

# Ελέγχουμε αν υπάρχει το BPRS_Total στο αρχείο
if 'BPRS_Total' in cols:
    bprs_index = cols.index('BPRS_Total')
    # Βάζουμε όσες στήλες βρήκαμε ακριβώς δίπλα του
    for i, col in enumerate(added_cols):
        cols.insert(bprs_index + 1 + i, col)
else:
    # Αν για κάποιο λόγο λείπει το BPRS_Total, απλά τις ξαναβάζουμε στο τέλος
    cols.extend(added_cols)

# Εφαρμόζουμε τη νέα, ασφαλή σειρά στο DataFrame
final_df = final_df[cols]
# ---------------------------------------------------------
# 6. Αποθήκευση
output_path = 'data_files/data_processed/csv_files/baseline_simplified_data.'
final_df.to_csv(output_path, index=False)

print("Τέλεια! Οι 3 νέες στήλες προστέθηκαν και μετακινήθηκαν ακριβώς δίπλα στο BPRS_Total.")