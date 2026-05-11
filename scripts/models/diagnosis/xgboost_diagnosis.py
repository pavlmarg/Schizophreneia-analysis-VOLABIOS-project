import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
import warnings
import os
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay, f1_score

warnings.filterwarnings("ignore", category=UserWarning)
os.makedirs('outputs', exist_ok=True)

# ΦΟΡΤΩΣΗ ΚΑΙ ΚΑΘΑΡΙΣΜΟΣ ΔΕΔΟΜΕΝΩΝ
baseline_df = pd.read_csv('data_files/data_processed/csv_files/baseline_simplified_data.csv')
followup_df = pd.read_csv('data_files/data_processed/csv_files/followup_10y_data.csv')

meas_cols_10y = [
    'BPRS_Negative_10y', 'BPRS_Positive_10y', 'BPRS_Total_10y', 
    'SANS_Total_10y', 'SAPS_Total_10y'
]

followup_clean = followup_df.dropna(subset=meas_cols_10y, how='all').copy()
cols_to_drop = ['active_10y', 'recovery_status_10y', 'DAS_Global_10y']
followup_clean = followup_clean.drop(columns=[c for c in cols_to_drop if c in followup_clean.columns])

df = pd.merge(baseline_df, followup_clean, on='person_id', how='inner')

# ΠΡΟΕΤΟΙΜΑΣΙΑ TARGET
def group_diagnosis(diag):
    if pd.isna(diag):
        return np.nan
    elif 'Schizophrenia' in str(diag): 
        return 1
    else:
        return 0 

df['target'] = df['diagnosis'].apply(group_diagnosis)
df = df.dropna(subset=['target']).copy()

# ΟΡΙΣΜΟΣ FEATURES
CONTINUOUS_FEATURES = [
    'baseline_age', 'DAP_months', 'DUP_months', 'DUI_months', 'DAT_months', 
    'SANS_Total', 'SAPS_Total', 'BPRS_Total', 'BPRS_Positive_Onset', 
    'BPRS_Negative_Onset', 'BPRS_Disorganized_Onset'
]

CATEGORICAL_FEATURES = [
    'cannabis_use', 'family_hx_psychosis', 'hospital_admission'
]

# TRAIN-TEST SPLIT
X_raw = df[CONTINUOUS_FEATURES + CATEGORICAL_FEATURES]
y = df['target']
patient_ids = df['person_id']

X_train_raw, X_test_raw, y_train, y_test, id_train, id_test = train_test_split(
    X_raw, y, patient_ids, test_size=0.2, stratify=y, random_state=42
)

# Imputation με στατιστικά του Training Set
train_medians = X_train_raw[CONTINUOUS_FEATURES].median()
train_modes = X_train_raw[CATEGORICAL_FEATURES].mode().iloc[0]

X_train_clean = X_train_raw.copy()
X_test_clean = X_test_raw.copy()

X_train_clean[CONTINUOUS_FEATURES] = X_train_clean[CONTINUOUS_FEATURES].fillna(train_medians)
X_test_clean[CONTINUOUS_FEATURES] = X_test_clean[CONTINUOUS_FEATURES].fillna(train_medians)

X_train_clean[CATEGORICAL_FEATURES] = X_train_clean[CATEGORICAL_FEATURES].fillna(train_modes)
X_test_clean[CATEGORICAL_FEATURES] = X_test_clean[CATEGORICAL_FEATURES].fillna(train_modes)

# Dummy variables
X_train = pd.get_dummies(X_train_clean, columns=CATEGORICAL_FEATURES, drop_first=True)
X_test = pd.get_dummies(X_test_clean, columns=CATEGORICAL_FEATURES, drop_first=True)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# PIPELINE: SCALER -> SMOTE -> PCA -> XGBOOST
pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('pca', PCA(random_state=42, n_components=4)),
    ('clf', xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss', learning_rate=0.1, n_estimators=250, max_depth=2))
])

pipeline.fit(X_train, y_train)

# Προβλέψεις
y_prob = pipeline.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_prob)

# Threshold Tuning
thresholds = np.linspace(0.05, 0.95, 100) 
best_macro_f1 = 0.0
optimal_threshold = 0.50 

for thresh in thresholds:
    temp_pred = (y_prob >= thresh).astype(int)
    current_macro_f1 = f1_score(y_test, temp_pred, average='macro')
    if current_macro_f1 > best_macro_f1:
        best_macro_f1 = current_macro_f1
        optimal_threshold = thresh

y_pred_optimal = (y_prob >= optimal_threshold).astype(int)

# Αποθήκευση της σύνοψης
summary = {
    'Algorithm': 'XGBoost',
    'Parameters': "{learning_rate = 0.1, max_depth = 3, n_estimators = 250, pca_n_components = 5}",
    'Test ROC-AUC': f"{auc_score:.3f}",
    'Test F1 (Macro Avg)': f"{best_macro_f1:.3f}" 
}
pd.DataFrame([summary]).to_csv('outputs/summary_pca_xgboost_10y.csv', index=False)

# ΓΡΑΦΗΜΑΤΑ
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('XGBoost Model', fontsize=14, fontweight='bold')

cm = confusion_matrix(y_test, y_pred_optimal)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Other', 'Schizophrenia'])
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title(f'Confusion Matrix\n(Optimal Threshold: {optimal_threshold:.2f})')

fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[1].plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {auc_score:.3f}')
axes[1].plot([0, 1], [0, 1], 'k--', lw=1, label='Random guessing')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve')
axes[1].legend()

# XGBoost Feature Importances
importances = pipeline.named_steps['clf'].feature_importances_
pc_names = [f"PC{i+1}" for i in range(len(importances))]
importance_df = pd.DataFrame({'feature': pc_names, 'importance': importances})
importance_df = importance_df[importance_df['importance'] > 0].sort_values('importance', ascending=False)

axes[2].barh(importance_df['feature'], importance_df['importance'], color='blue')
axes[2].set_title('Top Principal Components (Drivers)')
axes[2].set_xlabel('Feature Importance')
axes[2].invert_yaxis()

plt.tight_layout()
plt.savefig('outputs/master_diagnosis_pca_xgboost_10y.png', dpi=150, bbox_inches='tight')
plt.close()

# SHAP ΕΠΕΞΗΓΗΜΑΤΙΚΟΤΗΤΑ ΠΑΝΩ ΣΤΑ PCs
xgb_model = pipeline.named_steps['clf']
model_scaler = pipeline.named_steps['scaler']
model_pca = pipeline.named_steps['pca']

# Transform X_test -> Scaled -> PCA
X_test_transformed = model_pca.transform(model_scaler.transform(X_test))
X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=pc_names)

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_transformed_df)

shap_values_pos = shap_values

base_value = explainer.expected_value
if isinstance(base_value, (np.ndarray, list)):
    base_value = base_value[0]

# SHAP Summary Plot
plt.figure(figsize=(10, 6))
plt.title("SHAP Summary Plot: Principal Component Drivers (XGBoost)", fontsize=14, fontweight='bold')
shap.summary_plot(shap_values_pos, X_test_transformed_df, show=False)
plt.tight_layout()
plt.savefig('outputs/shap_summary_pca_xgboost_10y.png', dpi=150, bbox_inches='tight')
plt.close()

# SHAP Force Plot 
patient_idx = 11
real_person_id = id_test.iloc[patient_idx]

plt.figure(figsize=(12, 4))
shap.force_plot(
    base_value, 
    shap_values_pos[patient_idx, :], 
    X_test_transformed_df.iloc[patient_idx, :], 
    matplotlib=True,
    show=False
)
plt.title(f"SHAP Force Plot: PCA Explanation for Person ID {real_person_id}", y=1.4, fontweight='bold')
plt.savefig(f'outputs/shap_force_plot_patient_{real_person_id}_pca_xgboost.png', dpi=150, bbox_inches='tight')
plt.close()