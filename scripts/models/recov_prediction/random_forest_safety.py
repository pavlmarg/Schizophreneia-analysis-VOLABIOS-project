import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report,roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay, f1_score)

# Load Data
baseline = pd.read_csv('data_files/data_processed/csv_files/master_baseline_comprehensive.csv')
followup = pd.read_csv('data_files/data_processed/csv_files/followup_10y_data.csv')

# Merge and filter for known outcomes
df_full = baseline.merge(
    followup[['person_id', 'recovery_status_10y']],
    on='person_id',
    how='inner'
)
df = df_full[df_full['recovery_status_10y'].isin(['Recovered', 'Not recovered'])].copy()

# Target Mapping: 1 = Recovered, 0 = Not Recovered
df['target'] = df['recovery_status_10y'].map({'Recovered': 1, 'Not recovered': 0})

CONTINUOUS_FEATURES = [
    'baseline_age', 'BPRS_Total', 'BPRS_Positive_Onset',
    'BPRS_Disorganized_Onset', 'DAP_months', 'DAT_months',
    'DUI_months', 'DUP_months', 'SANS_Total', 'SAPS_Total'
]

CATEGORICAL_FEATURES = [
    'gender', 'cannabis_use', 'lives_with_parents', 'family_hx_psychosis',
    'hospital_admission', 'education', 'socioeconomic_status', 'employment',
    'marital_status', 'diagnosis'
]

# Impute Missing Values (Median for numbers, Mode for categories)
for col in CONTINUOUS_FEATURES:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())
for col in CATEGORICAL_FEATURES:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])

# One-Hot Encoding
X = pd.get_dummies(df[CONTINUOUS_FEATURES + CATEGORICAL_FEATURES], columns=CATEGORICAL_FEATURES, drop_first=True)
y = df['target']

# Train/Test Split (80/20 Stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Pipeline: Standardize -> Random Forest 
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(
        class_weight='balanced',
        n_estimators=500,        
        max_depth=4,             
        min_samples_leaf=2,      
        random_state=10,
        n_jobs=-1                
    ))
])

# Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
scoring = ['roc_auc', 'f1']
cv_results = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring)

# Fit and Predict on Test Set
pipeline.fit(X_train, y_train)
y_prob = pipeline.predict_proba(X_test)[:, 1]

# Default 50% Threshold Predictions
default_auc = roc_auc_score(y_test, y_prob)
print(f"Test ROC-AUC Score: {default_auc:.3f}")


# MODIFICATION: CLINICAL SAFETY THRESHOLD
fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)

# Catch at least 80% of "Not Recovered" patients
target_safety_rate = 0.8

# Find all thresholds that meet this safety requirement
valid_indices = np.where(1 - fpr >= target_safety_rate)[0]

# Pick the threshold that meets the safety requirement but still maximizes recovery detection
best_idx = valid_indices[-1]
clinical_threshold = roc_thresholds[best_idx]

print(f"\nTargeting 'Not Recovered' Catch Rate: >= {target_safety_rate:.0%}")
print(f"Clinical Safety Cutoff found at: {clinical_threshold:.3f}")
print(f"\n--- Final Classification Report (Using Safety Cutoff {clinical_threshold:.3f}) ---")

# Apply the new clinical threshold
y_pred_safe = (y_prob >= clinical_threshold).astype(int)
print(classification_report(y_test, y_pred_safe, target_names=['Not Recovered', 'Recovered']))


fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('VOLABIOS: Clinical Safety Random Forest Model\n',fontsize=14, fontweight='bold')


cm = confusion_matrix(y_test, y_pred_safe)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Recovered', 'Recovered'])
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title(f'Confusion Matrix\n(Safety Threshold: {clinical_threshold:.2f})')

axes[1].plot(fpr, tpr, color='purple', lw=2, label=f'AUC = {default_auc:.3f}')
axes[1].plot([0, 1], [0, 1], 'k--', lw=1, label='Random guessing')
axes[1].plot(fpr[best_idx], tpr[best_idx], marker='o', markersize=8, color='red', label='Selected Threshold')
axes[1].set_xlabel('False Positive Rate (1 - Specificity)')
axes[1].set_ylabel('True Positive Rate (Sensitivity)')
axes[1].set_title('ROC Curve')
axes[1].legend()

importances = pipeline.named_steps['clf'].feature_importances_
importance_df = pd.DataFrame({'feature': X.columns, 'importance': importances})
importance_df = importance_df[importance_df['importance'] > 0].sort_values('importance', ascending=False).head(10)

axes[2].barh(importance_df['feature'], importance_df['importance'], color='purple')
axes[2].set_title('Top 10 Drivers of Model Decisions\n(Magnitude of impact)')
axes[2].set_xlabel('Importance Score')
axes[2].invert_yaxis()

plt.tight_layout()
plt.savefig('outputs/master_random_forest_clinical.png', dpi=150, bbox_inches='tight')
print("Visualizations saved to 'outputs/master_random_forest_clinical.png'.")

print("\n SUMMARY")
summary = {
    'Algorithm': 'Random Forest (Class-Balanced & Constrained)',
    'CV ROC-AUC': f"{cv_results['test_roc_auc'].mean():.3f} ± {cv_results['test_roc_auc'].std():.3f}",
    'Test ROC-AUC': f"{default_auc:.3f}",
    'Clinical Threshold': f"{clinical_threshold:.3f}",
    'Optimized Test F1': f"{f1_score(y_test, y_pred_safe):.3f}" 
}
for k, v in summary.items():
    print(f"{k:<30}: {v}")
    
summary_df = pd.DataFrame([summary])

summary_df.to_csv('outputs/summary_clinical_model.csv', index=False)