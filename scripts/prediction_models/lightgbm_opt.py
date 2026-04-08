import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_recall_curve
)


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

# Impute Missing Values
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

# Base Pipeline (Hyperparameters will be overwritten by GridSearch)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', lgb.LGBMClassifier(
        class_weight='balanced',
        random_state=10,
        n_jobs=-1,
        verbose=-1
    ))
])

# Cross-Validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)


# AUTOMATED HYPERPARAMETER TUNING
param_grid = {
    'clf__n_estimators': [100, 200, 300],
    'clf__learning_rate': [0.01, 0.05, 0.1],
    'clf__max_depth': [3, 4, 5],
    'clf__num_leaves': [7, 15, 31], 
    'clf__min_child_samples': [3, 5]
}


grid_search = GridSearchCV(
    pipeline, 
    param_grid, 
    cv=cv, 
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

# Run the massive search on the training data
grid_search.fit(X_train, y_train)

# Extract the winning model
best_pipeline = grid_search.best_estimator_

print("\nBest Variables Found by Grid Search:")
for param_name, param_value in grid_search.best_params_.items():
    print(f" - {param_name.replace('clf__', '')}: {param_value}")

# PREDICT & FIND OPTIMAL THRESHOLD 
y_prob = best_pipeline.predict_proba(X_test)[:, 1]

default_auc = roc_auc_score(y_test, y_prob)
print(f"\nTest ROC-AUC Score: {default_auc:.3f}")

# Threshold Optimization
precisions, recalls, pr_thresholds = precision_recall_curve(y_test, y_prob)
f1_scores = np.divide(2 * (precisions * recalls), (precisions + recalls), out=np.zeros_like(precisions), where=(precisions + recalls) != 0)

optimal_idx = np.argmax(f1_scores)
optimal_threshold = pr_thresholds[optimal_idx]

print(f"Optimal Cutoff found at: {optimal_threshold:.3f}")
print(f"\n--- Final Classification Report {optimal_threshold:.3f}) ---")

# Apply optimal threshold
y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
print(classification_report(y_test, y_pred_optimal, target_names=['Not Recovered', 'Recovered']))


# VISUALIZATIONS
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('VOLABIOS: Grid-Searched LightGBM Recovery Model\n(_bias denotes variable with known attrition bias)',fontsize=14, fontweight='bold')

#  Confusion Matrix
cm = confusion_matrix(y_test, y_pred_optimal)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Recovered', 'Recovered'])
disp.plot(ax=axes[0], colorbar=False, cmap='Greens')
axes[0].set_title(f'Confusion Matrix\n(Optimal Threshold: {optimal_threshold:.2f})')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[1].plot(fpr, tpr, color='forestgreen', lw=2, label=f'AUC = {default_auc:.3f}')
axes[1].plot([0, 1], [0, 1], 'k--', lw=1, label='Random guessing')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve')
axes[1].legend()

# Feature Importance
importances = best_pipeline.named_steps['clf'].booster_.feature_importance(importance_type='gain')
importance_df = pd.DataFrame({'feature': X.columns, 'importance': importances})
importance_df = importance_df[importance_df['importance'] > 0].sort_values('importance', ascending=False).head(15)

axes[2].barh(importance_df['feature'], importance_df['importance'], color='forestgreen')
axes[2].set_title('Top 15 Drivers of Model Decisions\n(Information Gain)')
axes[2].set_xlabel('Total Gain')
axes[2].invert_yaxis()

plt.tight_layout()
plt.savefig('outputs/master_lightgbm_gridsearch.png', dpi=150, bbox_inches='tight')
print("\nVisualizations saved to 'outputs/master_lightgbm_gridsearch.png'.")

# SUMMARY & CSV EXPORT
print("\n SUMMARY FOR PAPER WRITING")

summary = {
    'Algorithm': 'LightGBM (Grid Search Optimized)',
    'Best Parameters': str({k.replace('clf__', ''): v for k, v in grid_search.best_params_.items()}),
    'Test ROC-AUC': f"{default_auc:.3f}",
    'Optimal Threshold': f"{optimal_threshold:.3f}",
    'Optimized Test F1': f"{f1_score(y_test, y_pred_optimal):.3f}"
}
for k, v in summary.items():
    print(f"{k:<20}: {v}")

# Save to CSV
summary_df_opt = pd.DataFrame([summary])
csv_path_opt = 'outputs/summary_lgbm_gridsearch.csv'
summary_df_opt.to_csv(csv_path_opt, index=False)
print(f"\n[Success] Results saved to: {csv_path_opt}")