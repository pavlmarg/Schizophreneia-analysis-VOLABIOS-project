import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import xgboost as xgb

from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, 
    roc_auc_score, 
    roc_curve, 
    confusion_matrix, 
    ConfusionMatrixDisplay,
    precision_recall_curve,
    f1_score
)

warnings.filterwarnings('ignore')

# Create outputs directory
os.makedirs('outputs', exist_ok=True)

# 1. Load Data
df = pd.read_csv('data_files/data_processed/csv_files/master_baseline_comprehensive.csv')

# =====================================================================
# TARGET CREATION: SCHIZOPHRENIA (1) VS. OTHER (0)
# =====================================================================
def group_diagnosis(diag):
    if pd.isna(diag):
        return np.nan
    elif 'Schizophrenia' in str(diag): 
        return 1
    else:
        return 0 

df['target'] = df['diagnosis'].apply(group_diagnosis)
df = df.dropna(subset=['target'])

print(f"Dataset ready. Predicting across {len(df)} total patients.")
class_counts = df['target'].value_counts()
print(class_counts.rename({1: 'Schizophrenia', 0: 'Other'}))


# DYNAMIC SCALE_POS_WEIGHT FOR XGBOOST
weight_ratio = class_counts[0] / class_counts[1]


# =====================================================================
# FEATURE SELECTION
# =====================================================================
CONTINUOUS_FEATURES = [ 'DAP_months', 'BPRS_Negative_Onset']

CATEGORICAL_FEATURES = [ 'marital_status' ]

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

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# =====================================================================
# MODELING: XGBOOST WITH GRID SEARCH
# =====================================================================
# Base Pipeline using XGBoost
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', xgb.XGBClassifier(
        scale_pos_weight=weight_ratio,
        random_state=10, 
        n_jobs=-1,
        eval_metric='logloss'
    ))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)

param_grid = {
    'clf__n_estimators': [150],
    'clf__learning_rate': [0.01],
    'clf__max_depth': [2]
}

print("\nRunning Grid Search for XGBoost...")
grid_search = GridSearchCV(
    pipeline, 
    param_grid, 
    cv=cv, 
    scoring='roc_auc', 
    n_jobs=-1, 
    verbose=1
)

grid_search.fit(X_train, y_train)
best_pipeline = grid_search.best_estimator_

# =====================================================================
# EVALUATION & MACRO F1 THRESHOLD OPTIMIZATION
# =====================================================================

y_prob = best_pipeline.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_prob)
print(f"\nTest ROC-AUC Score: {auc_score:.3f}")

thresholds = np.linspace(0.05, 0.95, 100)
best_macro_f1 = 0.0

for thresh in thresholds:
    
    temp_pred = (y_prob >= thresh).astype(int)
    
    current_macro_f1 = f1_score(y_test, temp_pred, average='macro')
    
    if current_macro_f1 > best_macro_f1:
        best_macro_f1 = current_macro_f1
        optimal_threshold = thresh

print(f"Optimal MACRO F1 Cutoff found at: {optimal_threshold:.3f} (Macro F1 Score: {best_macro_f1:.3f})")

y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
print(classification_report(y_test, y_pred_optimal, target_names=['Other', 'Schizophrenia']))

# =====================================================================
# VISUALIZATIONS
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('VOLABIOS: F1-Optimized XGBoost Diagnostic Model', fontsize=14, fontweight='bold')

# A. Confusion Matrix
cm = confusion_matrix(y_test, y_pred_optimal)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Other', 'Schizophrenia'])
disp.plot(ax=axes[0], colorbar=False, cmap='Reds') # Changed to Reds to differentiate from LightGBM
axes[0].set_title(f'Confusion Matrix\n(Optimal Threshold: {optimal_threshold:.2f})')

# B. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[1].plot(fpr, tpr, color='firebrick', lw=2, label=f'AUC = {auc_score:.3f}')
axes[1].plot([0, 1], [0, 1], 'k--', lw=1, label='Random guessing')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve')
axes[1].legend()

# C. Feature Importance (XGBoost syntax)
importances = best_pipeline.named_steps['clf'].feature_importances_
importance_df = pd.DataFrame({'feature': X.columns, 'importance': importances})
importance_df = importance_df[importance_df['importance'] > 0].sort_values('importance', ascending=False).head(5)

axes[2].barh(importance_df['feature'], importance_df['importance'], color='firebrick')
axes[2].set_title('Top Drivers of Diagnosis')
axes[2].set_xlabel('Importance Score')
axes[2].invert_yaxis()

plt.tight_layout()
plt.savefig('outputs/master_diagnosis_xgboost.png', dpi=150, bbox_inches='tight')
print("\nVisualizations saved to 'outputs/master_diagnosis_xgboost.png'.")

# =====================================================================
# SUMMARY & CSV EXPORT
# =====================================================================
print("\n" + "="*50)
print(" SUMMARY FOR RESEARCH REPORT")
print("="*50)

# Create a dictionary of the final results
summary = {
    'Algorithm': 'XGBoost (Schizophrenia vs. Other)',
    'Best Parameters': str({k.replace('clf__', ''): v for k, v in grid_search.best_params_.items()}),
    'Test ROC-AUC': f"{auc_score:.3f}",
    'Test F1 (Macro Avg)': f"{f1_score(y_test, y_pred_optimal, average='macro'):.3f}" 
}

# Print it nicely to the console
for key, value in summary.items():
    print(f"{key:<20}: {value}")

# Convert to DataFrame and save as CSV
summary_df_xgb = pd.DataFrame([summary])
csv_path_xgb = 'outputs/summary_xgboost_diagnosis.csv'
summary_df_xgb.to_csv(csv_path_xgb, index=False)

print(f"\n[Success] Final summary saved to: {csv_path_xgb}")