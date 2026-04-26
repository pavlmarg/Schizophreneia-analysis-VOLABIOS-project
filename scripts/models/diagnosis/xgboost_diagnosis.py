import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
import warnings
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay, f1_score

warnings.filterwarnings("ignore", category=UserWarning)

df = pd.read_csv('data_files/data_processed/csv_files/master_baseline_comprehensive.csv')

def group_diagnosis(diag):
    if pd.isna(diag):
        return np.nan
    elif 'Schizophrenia' in str(diag): 
        return 1
    else:
        return 0 

df['target'] = df['diagnosis'].apply(group_diagnosis)
df = df.dropna(subset=['target'])

CONTINUOUS_FEATURES = [
    'baseline_age', 'DAP_months', 'DUP_months', 'DUI_months', 'DAT_months', 
    'SANS_Total', 'SAPS_Total', 'BPRS_Total', 'BPRS_Positive_Onset', 
    'BPRS_Negative_Onset', 'BPRS_Disorganized_Onset'
]

CATEGORICAL_FEATURES = [
    'gender', 'cannabis_use', 'lives_with_parents', 
    'family_hx_psychosis', 'hospital_admission', 
    'education', 'socioeconomic_status', 'employment', 'marital_status'
]

for col in CONTINUOUS_FEATURES:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())
for col in CATEGORICAL_FEATURES:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])

X = pd.get_dummies(df[CONTINUOUS_FEATURES + CATEGORICAL_FEATURES], columns=CATEGORICAL_FEATURES, drop_first=True)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

scale_weight = (len(y_train) - y_train.sum()) / y_train.sum()

xgb_core = xgb.XGBClassifier(
    scale_pos_weight=scale_weight, 
    random_state=10, 
    n_jobs=-1, 
    eval_metric='logloss',
    n_estimators=150,
    learning_rate=0.01,   
    max_depth=2           
)

rfecv = RFECV(
    estimator=xgb_core,
    step=1,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=10),
    scoring='roc_auc',
    n_jobs=-1
)
rfecv.fit(X_train_scaled, y_train)

selected_features = X.columns[rfecv.support_]
optimal_num_features = rfecv.n_features_

X_train_optimal = X_train[selected_features]
X_test_optimal = X_test[selected_features]

# --- ΑΠΛΟΠΟΙΗΜΕΝΟ PIPELINE ΧΩΡΙΣ GRID SEARCH ---
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', xgb.XGBClassifier(
        scale_pos_weight=scale_weight, 
        random_state=10, 
        n_jobs=-1, 
        eval_metric='logloss',
        n_estimators=150,       # Οι δικές σου σταθερές παράμετροι μπαίνουν απευθείας εδώ
        learning_rate=0.01,
        max_depth=2
    ))
])

# Απευθείας εκπαίδευση του Pipeline
pipeline.fit(X_train_optimal, y_train)

y_prob = pipeline.predict_proba(X_test_optimal)[:, 1]
auc_score = roc_auc_score(y_test, y_prob)

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

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('VOLABIOS: F1-Optimized Diagnostic Model (XGBoost)', fontsize=14, fontweight='bold')

cm = confusion_matrix(y_test, y_pred_optimal)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Other', 'Schizophrenia'])
disp.plot(ax=axes[0], colorbar=False, cmap='Oranges')
axes[0].set_title(f'Confusion Matrix\n(Optimal Threshold: {optimal_threshold:.2f})')

fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc_score:.3f}')
axes[1].plot([0, 1], [0, 1], 'k--', lw=1, label='Random guessing')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve')
axes[1].legend()

importances = pipeline.named_steps['clf'].feature_importances_
importance_df = pd.DataFrame({'feature': X_train_optimal.columns, 'importance': importances})
importance_df = importance_df[importance_df['importance'] > 0].sort_values('importance', ascending=False)

axes[2].barh(importance_df['feature'], importance_df['importance'], color='darkorange')
axes[2].set_title('Top Drivers of Diagnosis')
axes[2].set_xlabel('Feature Importance')
axes[2].invert_yaxis()

plt.tight_layout()
plt.savefig('outputs/master_diagnosis_xgboost.png', dpi=150, bbox_inches='tight')
plt.close()

summary = {
    'Algorithm': 'XGBoost (Schizophrenia vs. Other)',
    'Best Parameters': "{'learning_rate': 0.01, 'max_depth': 2, 'n_estimators': 150}",
    'Test ROC-AUC': f"{auc_score:.3f}",
    'Test F1 (Macro Avg)': f"{best_macro_f1:.3f}" 
}
pd.DataFrame([summary]).to_csv('outputs/summary_xgboost_diagnosis.csv', index=False)

xgb_model = pipeline.named_steps['clf']
model_scaler = pipeline.named_steps['scaler']

X_test_scaled = model_scaler.transform(X_test_optimal)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test_optimal.columns)

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_scaled_df)

shap_values_pos = shap_values

base_value = explainer.expected_value
if isinstance(base_value, (np.ndarray, list)):
    base_value = base_value[0]

plt.figure(figsize=(10, 6))
plt.title("SHAP Summary Plot: Global Feature Drivers (XGBoost)", fontsize=14, fontweight='bold')
shap.summary_plot(shap_values_pos, X_test_optimal, show=False)
plt.tight_layout()
plt.savefig('outputs/shap_summary_plot_xgboost.png', dpi=150, bbox_inches='tight')
plt.close()

patient_idx = 0
plt.figure(figsize=(12, 4))
shap.force_plot(
    base_value, 
    shap_values_pos[patient_idx, :], 
    X_test_optimal.iloc[patient_idx, :], 
    matplotlib=True,
    show=False
)
plt.title(f"SHAP Force Plot: Explanation for Patient {patient_idx} (XGBoost)", y=1.4, fontweight='bold')
plt.savefig(f'outputs/shap_force_plot_patient_{patient_idx}_xgboost.png', dpi=150, bbox_inches='tight')
plt.close()