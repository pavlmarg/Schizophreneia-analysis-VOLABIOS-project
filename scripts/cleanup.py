import pandas as pd
import os

data_dir = 'data_files'

datasets = {}

# 1. Loop through every file in the directory
try:
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(data_dir, filename)
            
            # Load the dataframe
            df = pd.read_csv(file_path)
            original_col_count = df.shape[1]
            
            # 2. Drop columns that are 100% missing (MCAR)
            df = df.dropna(axis=1, how='all')
            after_nan_count = df.shape[1]
            
            # 3. Drop columns where all data is the exact same (Zero Variance)
            # dropna=False ensures we don't accidentally drop binary flags (1.0 and NaN)
            df = df.loc[:, df.nunique(dropna=False) > 1]
            final_col_count = df.shape[1]
            
            # Save the cleaned dataframe to our dictionary
            datasets[filename] = df
            
            # Print a summary report for each file
            print(f"--- {filename} ---")
            print(f"Original columns: {original_col_count}")
            print(f"Dropped {original_col_count - after_nan_count} completely empty columns.")
            print(f"Dropped {after_nan_count - final_col_count} zero-variance columns.")
            print(f"Remaining active columns: {final_col_count}\n")
            
            # Define the new target directory for cleaned files
            processed_dir = 'data_processed'

            # Safely create the folder if it doesn't already exist (prevents FileNotFoundError)
            os.makedirs(processed_dir, exist_ok=True)

            # Loop through our dictionary of cleaned dataframes and save them
            for filename, cleaned_df in datasets.items():
            # Build the full file path (e.g., 'data_processed/person.csv')
                save_path = os.path.join(processed_dir, filename)
    
                # Save the dataframe without the pandas index column
                cleaned_df.to_csv(save_path, index=False)
    
                print(f"Successfully saved cleaned {filename} to {processed_dir}/")

            print("\nAll datasets successfully processed and saved!")

except Exception as e:
    print(f"Error loading files: {e}")
    