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
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score

warnings.filterwarnings("ignore")
os.makedirs('outputs', exist_ok=True)

# LOAD DATA
baseline_df = pd.read_csv('data_files/data_processed/csv_files/baseline_simplified_data.csv')

# TARGET PREPARATION
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

# 70/30 TRAIN TEST SPLIT
df_train, df_test = train_test_split(df, test_size=0.3, stratify=df['target'], random_state=42)

# DROP FEATURES WITH >40% EMPTY CELLS
threshold = int((len(CONTINUOUS_FEATURES) + len(CATEGORICAL_FEATURES)) * 0.4)
df_train = df_train.dropna(thresh=threshold, subset=CONTINUOUS_FEATURES + CATEGORICAL_FEATURES).copy()
df_test = df_test.dropna(thresh=threshold, subset=CONTINUOUS_FEATURES + CATEGORICAL_FEATURES).copy()

y_train = df_train['target']
id_train = df_train['person_id']
X_train_cont = df_train[CONTINUOUS_FEATURES]
X_train_cat = df_train[CATEGORICAL_FEATURES]

y_test = df_test['target']
id_test = df_test['person_id']
X_test_cont = df_test[CONTINUOUS_FEATURES]
X_test_cat = df_test[CATEGORICAL_FEATURES]

X_train_cat_encoded = pd.get_dummies(X_train_cat, drop_first=True)
X_test_cat_encoded = pd.get_dummies(X_test_cat, drop_first=True)
X_test_cat_encoded = X_test_cat_encoded.reindex(columns=X_train_cat_encoded.columns, fill_value=0)

NEW_CAT_FEATURES = list(X_train_cat_encoded.columns)

# ITERATIVE IMPUTER
rf_imputer_cont = IterativeImputer(estimator=RandomForestRegressor(n_estimators=50, random_state=42), random_state=42)
X_train_cont_imp = pd.DataFrame(rf_imputer_cont.fit_transform(X_train_cont), columns=CONTINUOUS_FEATURES, index=X_train_cont.index)
X_test_cont_imp = pd.DataFrame(rf_imputer_cont.transform(X_test_cont), columns=CONTINUOUS_FEATURES, index=X_test_cont.index)

rf_imputer_cat = IterativeImputer(estimator=RandomForestClassifier(n_estimators=50, random_state=42), random_state=42)
X_train_cat_imp = pd.DataFrame(rf_imputer_cat.fit_transform(X_train_cat_encoded), columns=NEW_CAT_FEATURES, index=X_train_cat_encoded.index)
X_test_cat_imp = pd.DataFrame(rf_imputer_cat.transform(X_test_cat_encoded), columns=NEW_CAT_FEATURES, index=X_test_cat_encoded.index)

X_train = pd.concat([X_train_cont_imp, X_train_cat_imp], axis=1)
X_test = pd.concat([X_test_cont_imp, X_test_cat_imp], axis=1)


# FEATURE SELECTION
# Filter important features (p>0.3)
f_stats, p_values_cont = f_classif(X_train[CONTINUOUS_FEATURES], y_train)
selected_cont = [CONTINUOUS_FEATURES[i] for i in range(len(CONTINUOUS_FEATURES)) if p_values_cont[i] < 0.3]

scaler_chi = MinMaxScaler()
X_train_cat_scaled = scaler_chi.fit_transform(X_train[NEW_CAT_FEATURES])
chi_stats, p_values_cat = chi2(X_train_cat_scaled, y_train)
selected_cat = [NEW_CAT_FEATURES[i] for i in range(len(NEW_CAT_FEATURES)) if p_values_cat[i] < 0.3]

filtered_features = selected_cont + selected_cat
X_train_filtered = X_train[filtered_features]
X_test_filtered = X_test[filtered_features]


# Sequential Feature Forward Selector with 5-fold cross-validation
sfs = SequentialFeatureSelector(
    estimator=lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1), 
    direction='forward',
    scoring='f1_macro',
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1
)
sfs.fit(X_train_filtered, y_train)
final_features = list(X_train_filtered.columns[sfs.get_support()])

X_train_final = X_train_filtered[final_features]
X_test_final = X_test_filtered[final_features]


# 5-FOLD CROSS-VALIDATION (EVALUATION)
cv_auc_scores = []
cv_f1_scores = []
cv_precision_scores = []
cv_recall_scores = []
cv_thresholds = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in skf.split(X_train_final, y_train):
    X_cv_train, X_cv_val = X_train_final.iloc[train_idx], X_train_final.iloc[val_idx]
    y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # SMOTE IN THE CV
    smote_cv = SMOTE(random_state=42)
    X_cv_train_smote, y_cv_train_smote = smote_cv.fit_resample(X_cv_train, y_cv_train)
    
    scaler_cv = MinMaxScaler()
    X_cv_train_scaled = scaler_cv.fit_transform(X_cv_train_smote)
    X_cv_val_scaled = scaler_cv.transform(X_cv_val)
    
    clf_cv = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1, learning_rate=0.03, n_estimators=150, max_depth=2)
    clf_cv.fit(X_cv_train_scaled, y_cv_train_smote)
    
    cv_probs = clf_cv.predict_proba(X_cv_val_scaled)[:, 1]
    
    best_fold_f1 = 0.0
    best_fold_thresh = 0.50
    for thresh in np.linspace(0.05, 0.95, 100):
        temp_pred = (cv_probs >= thresh).astype(int)
        f1 = f1_score(y_cv_val, temp_pred, average='macro')
        if f1 > best_fold_f1:
            best_fold_f1 = f1
            best_fold_thresh = thresh
            
    cv_thresholds.append(best_fold_thresh)
    cv_preds = (cv_probs >= best_fold_thresh).astype(int) 
    
    cv_auc_scores.append(roc_auc_score(y_cv_val, cv_probs))
    cv_f1_scores.append(f1_score(y_cv_val, cv_preds, average='macro'))
    cv_precision_scores.append(precision_score(y_cv_val, cv_preds, average='macro'))
    cv_recall_scores.append(recall_score(y_cv_val, cv_preds, average='macro'))

mean_cv_auc = np.mean(cv_auc_scores)
mean_cv_f1 = np.mean(cv_f1_scores)
mean_cv_precision = np.mean(cv_precision_scores)
mean_cv_recall = np.mean(cv_recall_scores)
optimal_threshold = np.mean(cv_thresholds)

# SMOTE OUT OF THE CV
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_final, y_train)

scaler = MinMaxScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_smote), columns=final_features)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_final), columns=final_features)

# MODEL TRAINING AND TEST SET EVALUATION
clf = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1, learning_rate=0.03, n_estimators=150, max_depth=2)
clf.fit(X_train_scaled, y_train_smote)

y_prob = clf.predict_proba(X_test_scaled)[:, 1]
test_auc_score = roc_auc_score(y_test, y_prob)

y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
best_macro_f1 = f1_score(y_test, y_pred_optimal, average='macro')
test_precision = precision_score(y_test, y_pred_optimal, average='macro')
test_recall = recall_score(y_test, y_pred_optimal, average='macro')

# ΓΡΑΦΗΜΑΤΑ
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('LightGBM Model', fontsize=14, fontweight='bold')

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
plt.savefig('outputs/diagnosis_lightgbm.png', dpi=150, bbox_inches='tight')
plt.close()

summary = {
    'Algorithm': 'LightGBM',
    'Parameters': "{learning_rate = 0.03, max_depth = 2, n_estimators = 150}",
    'CV ROC-AUC': f"{mean_cv_auc:.3f}",
    'CV F1': f"{mean_cv_f1:.3f}",
    'CV Precision': f"{mean_cv_precision:.3f}",
    'CV Recall': f"{mean_cv_recall:.3f}",
    'Test ROC-AUC': f"{test_auc_score:.3f}",
    'Test F1': f"{best_macro_f1:.3f}",
    'Test Precision': f"{test_precision:.3f}",
    'Test Recall': f"{test_recall:.3f}",
    'Features Selected': len(final_features)
}
pd.DataFrame([summary]).to_csv('outputs/summary_lightgbm.csv', index=False)

# SHAP explanation
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
plt.savefig('outputs/shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()

# Force Plot
patient_idx = 11
real_person_id = id_test.iloc[patient_idx]

base_value = explainer.expected_value
if isinstance(base_value, (list, np.ndarray)):
    base_value = base_value[1] if isinstance(explainer.expected_value, list) else base_value[0]

force_plot_html = shap.force_plot(
    base_value, 
    shap_values_pos[patient_idx, :], 
    X_test_scaled.iloc[patient_idx, :], 
    link="logit" 
)

filepath = f'outputs/shap_force_plot_patient_{real_person_id}.html'
shap.save_html(filepath, force_plot_html)
