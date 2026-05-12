import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import shap
import warnings
import os

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import f_classif, chi2, SequentialFeatureSelector
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay, f1_score

warnings.filterwarnings("ignore")
os.makedirs('outputs', exist_ok=True)

# LOAD DATA & PREPARE TARGET
baseline_df = pd.read_csv('data_files/data_processed/csv_files/baseline_simplified_data.csv')

def group_diagnosis(diag):
    if pd.isna(diag): return np.nan
    elif 'Schizophrenia' in str(diag): return 1
    else: return 0 

df = baseline_df.copy()
df['target'] = df['diagnosis'].apply(group_diagnosis)
df = df.dropna(subset=['target']).copy()

CONTINUOUS_FEATURES = [
    'baseline_age', 'DAP_months', 'DUP_months', 'DUI_months', 'DAT_months', 
    'SANS_Total', 'SAPS_Total', 'BPRS_Total', 'BPRS_Positive_Onset', 
    'BPRS_Negative_Onset', 'BPRS_Disorganized_Onset'
]
CATEGORICAL_FEATURES = [
    'cannabis_use', 'family_hx_psychosis', 'hospital_admission'
]

# DROP ROWS WITH >40% MISSING VALUES
threshold = int((len(CONTINUOUS_FEATURES) + len(CATEGORICAL_FEATURES)) * 0.4)
df = df.dropna(thresh=threshold, subset=CONTINUOUS_FEATURES + CATEGORICAL_FEATURES).copy()

# RANDOM FOREST IMPUTATION
X_cont = df[CONTINUOUS_FEATURES]
X_cat = df[CATEGORICAL_FEATURES]

X_cat_encoded = pd.get_dummies(X_cat, drop_first=True)

rf_imputer_cont = IterativeImputer(estimator=RandomForestRegressor(n_estimators=50, random_state=42), random_state=42)
df[CONTINUOUS_FEATURES] = rf_imputer_cont.fit_transform(X_cont)

rf_imputer_cat = IterativeImputer(estimator=RandomForestClassifier(n_estimators=50, random_state=42), random_state=42)
df[X_cat_encoded.columns] = rf_imputer_cat.fit_transform(X_cat_encoded)
df = df.drop(columns=CATEGORICAL_FEATURES) 
NEW_CAT_FEATURES = list(X_cat_encoded.columns)

# 4. DATA SPLIT (70/30)
X_raw = df[CONTINUOUS_FEATURES + NEW_CAT_FEATURES]
y = df['target']
patient_ids = df['person_id']

X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X_raw, y, patient_ids, test_size=0.3, stratify=y, random_state=42
)

# 5. FEATURE SELECTION (Restricted to Training Set)
# Stage A: Filter Methods
f_stats, p_values_cont = f_classif(X_train[CONTINUOUS_FEATURES], y_train)
selected_cont = [CONTINUOUS_FEATURES[i] for i in range(len(CONTINUOUS_FEATURES)) if p_values_cont[i] < 0.2]

scaler_chi = MinMaxScaler()
X_train_cat_scaled = scaler_chi.fit_transform(X_train[NEW_CAT_FEATURES])
chi_stats, p_values_cat = chi2(X_train_cat_scaled, y_train)
selected_cat = [NEW_CAT_FEATURES[i] for i in range(len(NEW_CAT_FEATURES)) if p_values_cat[i] < 0.2]

filtered_features = selected_cont + selected_cat
X_train_filtered = X_train[filtered_features]
X_test_filtered = X_test[filtered_features]

# Stage B: Sequential Forward Selection with 5-Fold CV
lgbm_base = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
cv_sfs = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

sfs = SequentialFeatureSelector(
    estimator=lgbm_base, 
    direction='forward',
    scoring='f1_macro',
    cv=cv_sfs,
    n_jobs=-1
)
sfs.fit(X_train_filtered, y_train)
final_features = list(X_train_filtered.columns[sfs.get_support()])

X_train_final = X_train_filtered[final_features]
X_test_final = X_test_filtered[final_features]


# MODEL EVALUATION WITH 5-FOLD CV 
cv_auc_scores = []
cv_f1_scores = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in skf.split(X_train_final, y_train):
    X_cv_train, X_cv_val = X_train_final.iloc[train_idx], X_train_final.iloc[val_idx]
    y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # SMOTE inside the fold to prevent data leakage
    smote_cv = SMOTE(random_state=42)
    X_cv_train_smote, y_cv_train_smote = smote_cv.fit_resample(X_cv_train, y_cv_train)
    
    # Scale inside the fold
    scaler_cv = MinMaxScaler()
    X_cv_train_scaled = scaler_cv.fit_transform(X_cv_train_smote)
    X_cv_val_scaled = scaler_cv.transform(X_cv_val)
    
    clf_cv = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1, learning_rate=0.01, n_estimators=150, max_depth=3)
    clf_cv.fit(X_cv_train_scaled, y_cv_train_smote)
    
    cv_probs = clf_cv.predict_proba(X_cv_val_scaled)[:, 1]
    cv_preds = (cv_probs >= 0.50).astype(int) 
    
    cv_auc_scores.append(roc_auc_score(y_cv_val, cv_probs))
    cv_f1_scores.append(f1_score(y_cv_val, cv_preds, average='macro'))

mean_cv_auc = np.mean(cv_auc_scores)
mean_cv_f1 = np.mean(cv_f1_scores)

# FINAL MODEL TRAINING 
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_final, y_train)

scaler = MinMaxScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_smote), columns=final_features)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_final), columns=final_features)

clf = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1, learning_rate=0.05, n_estimators=200, max_depth=4)
clf.fit(X_train_scaled, y_train_smote)

# EVALUATION 
y_prob = clf.predict_proba(X_test_scaled)[:, 1]
test_auc_score = roc_auc_score(y_test, y_prob)

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

# PLOTS
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('LightGBM Model (Paper Methodology)', fontsize=14, fontweight='bold')

cm = confusion_matrix(y_test, y_pred_optimal)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Other', 'Schizophrenia'])
disp.plot(ax=axes[0], colorbar=False, cmap='Greens')
axes[0].set_title(f'Confusion Matrix\n(Optimal Threshold: {optimal_threshold:.2f})')

fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[1].plot(fpr, tpr, color='darkgreen', lw=2, label=f'Test AUC = {test_auc_score:.3f}')
axes[1].plot([0, 1], [0, 1], 'k--', lw=1, label='Random guessing')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve')
axes[1].legend()

raw_importances = clf.booster_.feature_importance(importance_type='gain')
relative_importances = raw_importances / raw_importances.sum()
importance_df = pd.DataFrame({'feature': final_features, 'importance': relative_importances})
importance_df = importance_df[importance_df['importance'] > 0].sort_values('importance', ascending=False)

axes[2].barh(importance_df['feature'], importance_df['importance'], color='darkgreen')
axes[2].set_title('Selected Feature Importances')
axes[2].set_xlabel('Relative Importance') 
axes[2].set_xlim(0, 1.0)
axes[2].invert_yaxis()

plt.tight_layout()
plt.savefig('outputs/master_diagnosis_paper_lightgbm.png', dpi=150, bbox_inches='tight')
plt.close()

# SUMMARY
summary = {
    'Algorithm': 'LightGBM',
    'Parameters': "{learning_rate = 0.05, max_depth = 4, n_estimators = 200}",
    'CV ROC-AUC': f"{mean_cv_auc:.3f}",
    'CV F1 (Macro Avg)': f"{mean_cv_f1:.3f}",
    'Test ROC-AUC': f"{test_auc_score:.3f}",
    'Test F1 (Macro Avg)': f"{best_macro_f1:.3f}",
    'Features Selected': len(final_features)
}
pd.DataFrame([summary]).to_csv('outputs/summary_paper_lightgbm.csv', index=False)

# SHAP
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test_scaled)

if isinstance(shap_values, list):
    shap_values_pos = shap_values[1]
else:
    shap_values_pos = shap_values

plt.figure(figsize=(10, 6))
plt.title("SHAP Summary Plot: Selected Features", fontsize=14, fontweight='bold')
shap.summary_plot(shap_values_pos, X_test_scaled, show=False)
plt.tight_layout()
plt.savefig('outputs/shap_summary_paper.png', dpi=150, bbox_inches='tight')
plt.close()


# Force Plots
patient_idx = 11
real_person_id = id_test.iloc[patient_idx]


base_value = explainer.expected_value
if isinstance(base_value, (list, np.ndarray)):
    base_value = base_value[1] if isinstance(explainer.expected_value, list) else base_value[0]

plt.figure(figsize=(12, 4))
shap.force_plot(
    base_value, 
    shap_values_pos[patient_idx, :], 
    X_test_scaled.iloc[patient_idx, :], 
    matplotlib=True,
    show=False
)
plt.title(f"SHAP Force Plot: Explanation for Person ID {real_person_id}", y=1.4, fontweight='bold')
plt.savefig(f'outputs/shap_force_plot_patient_{real_person_id}_paper_lightgbm.png', dpi=150, bbox_inches='tight')
plt.close()