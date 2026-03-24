import pandas as pd
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. SETUP & PATHS
# ==========================================
data_dir = 'data_files'
processed_dir = 'data_processed'
os.makedirs(processed_dir, exist_ok=True)

# ==========================================
# 2. LOAD CORE TABLES
# ==========================================
person = pd.read_csv(os.path.join(data_dir, 'person.csv'))
obs = pd.read_csv(os.path.join(data_dir, 'observation.csv'))
death = pd.read_csv(os.path.join(data_dir, 'death.csv'))

# ==========================================
# 3. CALCULATE ATTRITION
# ==========================================
total_patients = len(person)
deceased_count = len(death)

# Completed follow-up (Look for the recovery observation flag)
recovery_obs = obs[obs['observation_source_value'] == 'recovery10m']
completed_count = len(recovery_obs)

# Lost to follow up (The remainder of the original cohort)
lost_count = total_patients - deceased_count - completed_count

# Setup data for plotting
counts = [completed_count, lost_count, deceased_count]
labels = ['Completed 10-Year Follow-up', 'Lost to Follow-up', 'Deceased']
colors = ['#66b3ff', '#ffcc99', '#ff9999']

# Print the statistical breakdown
print("==========================================")
print("     10-YEAR STUDY ATTRITION BREAKDOWN    ")
print("==========================================")
print(f"Original Cohort Size: {total_patients} patients\n")
for label, count in zip(labels, counts):
    pct = (count / total_patients) * 100
    print(f"- {label}: {count} patients ({pct:.1f}%)")
print("==========================================\n")

# ==========================================
# 4. VISUALIZATION (FIGURE 5)
# ==========================================
fig, ax = plt.subplots(figsize=(8, 6))

# Create a clean pie chart with percentages
wedges, texts, autotexts = ax.pie(
    counts, 
    labels=labels, 
    autopct='%1.1f%%', 
    startangle=140, 
    colors=colors,
    explode=(0.05, 0, 0) # Slightly separate the 'Completed' slice for emphasis
)

ax.set_title('Figure 5: 10-Year Study Attrition Breakdown\n(Original Cohort: 307 Patients)', fontsize=14, fontweight='bold')

plt.tight_layout()
save_path = os.path.join(processed_dir, 'figure5_attrition.png')
plt.savefig(save_path, dpi=300)
plt.close(fig)

print(f"Success! Figure 5 saved to: {save_path}")