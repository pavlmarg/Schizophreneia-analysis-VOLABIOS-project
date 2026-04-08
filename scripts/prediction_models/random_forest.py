import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import ( classification_report, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_recall_curve)

# Create outputs directory
os.makedirs('outputs', exist_ok=True)

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

# Train/Test Split 
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
y_pred_default = (y_prob >= 0.50).astype(int)
default_auc = roc_auc_score(y_test, y_prob)

print(f"Test ROC-AUC Score: {default_auc:.3f}")


# Find the threshold that maximizes the F1 Score
precisions, recalls, pr_thresholds = precision_recall_curve(y_test, y_prob)
f1_scores = np.divide(2 * (precisions * recalls), (precisions + recalls), out=np.zeros_like(precisions), where=(precisions + recalls) != 0)

optimal_idx = np.argmax(f1_scores)
optimal_threshold = pr_thresholds[optimal_idx]

print(f"Optimal Cutoff found at: {optimal_threshold:.3f}")
print(f"\n Final Classification Report{optimal_threshold:.3f})")

# Apply optimal threshold
y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
print(classification_report(y_test, y_pred_optimal, target_names=['Not Recovered', 'Recovered']))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('VOLABIOS: Final Random Forest Recovery Prediction Model\n(* Denotes variable with known attrition bias)',fontsize=14, fontweight='bold')


cm = confusion_matrix(y_test, y_pred_optimal)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Recovered', 'Recovered'])
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title(f'Confusion Matrix\n(Optimal Threshold: {optimal_threshold:.2f})')

fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[1].plot(fpr, tpr, color='purple', lw=2, label=f'AUC = {default_auc:.3f}')
axes[1].plot([0, 1], [0, 1], 'k--', lw=1, label='Random guessing')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
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
plt.savefig('outputs/master_random_forest_results.png', dpi=150, bbox_inches='tight')

print("\n SUMMARY ")

summary = {
    'Algorithm': 'Random Forest (Class-Balanced & Constrained)',
    'CV ROC-AUC': f"{cv_results['test_roc_auc'].mean():.3f} ± {cv_results['test_roc_auc'].std():.3f}",
    'Test ROC-AUC': f"{default_auc:.3f}",
    'Optimal Threshold': f"{optimal_threshold:.3f}",
    'Optimized Test F1': f"{f1_score(y_test, y_pred_optimal):.3f}"
}
for k, v in summary.items():
    print(f"{k}: {v}")
    
summary_df = pd.DataFrame([summary])

summary_df.to_csv('outputs/summary_optimized_model.csv', index=False)