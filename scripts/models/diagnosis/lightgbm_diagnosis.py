import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import optuna
import shap
import warnings
import os
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import f_classif, chi2, SequentialFeatureSelector
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score

warnings.filterwarnings("ignore")
os.makedirs('outputs', exist_ok=True)

# --- LOAD & LABEL ---
baseline_df = pd.read_csv('data_files/data_processed/csv_files/baseline_simplified_data.csv')

def group_diagnosis(diag):
    if pd.isna(diag):
        return np.nan
    elif 'Schizophrenia' in str(diag):
        return 1
    else:
        return 0

df = baseline_df.copy()
df['target'] = df['diagnosis'].apply(group_diagnosis)
df = df.dropna(subset=['target']).copy()

# --- FEATURE DEFINITIONS ---
CONTINUOUS_FEATURES = [
    'baseline_age', 'DAP_months', 'DUP_months', 'DUI_months', 'DAT_months',
    'SANS_Total', 'SAPS_Total', 'BPRS_Total', 'BPRS_Positive_Onset',
    'BPRS_Negative_Onset', 'BPRS_Disorganized_Onset'
]
CATEGORICAL_FEATURES = [
    'cannabis_use', 'family_hx_psychosis', 'hospital_admission'
]
ALL_FEATURES = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES

# Five seeds and 5-fold
SEEDS = [0, 7, 21, 42, 84]

# Drop rows missing more than 60% of features
df = df.dropna(thresh=int(len(ALL_FEATURES) * 0.4), subset=ALL_FEATURES).copy()

X_full = df[ALL_FEATURES].copy()
y_full = df['target'].reset_index(drop=True)
ids_full = df['person_id'].reset_index(drop=True)
X_full = X_full.reset_index(drop=True)

# --- METRIC STORAGE ---
all_auc, all_f1, all_prec, all_rec = [], [], [], []
feature_selection_counts = Counter()
total_folds = 0

# Per-fold snapshots for median fold selection
fold_snapshots = []

for seed in SEEDS:
    outer_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    for fold_idx, (train_idx, val_idx) in enumerate(outer_skf.split(X_full, y_full)):

        X_train_raw = X_full.iloc[train_idx].copy()
        X_val_raw = X_full.iloc[val_idx].copy()
        y_train = y_full.iloc[train_idx].reset_index(drop=True)
        y_val = y_full.iloc[val_idx].reset_index(drop=True)
        ids_val = ids_full.iloc[val_idx].reset_index(drop=True)

        # One-hot encode categoricals; align val columns to train to handle unseen levels
        X_train_cat = pd.get_dummies(X_train_raw[CATEGORICAL_FEATURES], drop_first=True)
        X_val_cat = pd.get_dummies(X_val_raw[CATEGORICAL_FEATURES], drop_first=True)
        X_val_cat = X_val_cat.reindex(columns=X_train_cat.columns, fill_value=0)
        new_cat_features = list(X_train_cat.columns)

        X_train_cont = X_train_raw[CONTINUOUS_FEATURES].reset_index(drop=True)
        X_val_cont = X_val_raw[CONTINUOUS_FEATURES].reset_index(drop=True)
        X_train_cat = X_train_cat.reset_index(drop=True)
        X_val_cat = X_val_cat.reset_index(drop=True)

        # Iterative imputation fitted on train only, applied to val to prevent leakage
        imp_cont = IterativeImputer(estimator=RandomForestRegressor(n_estimators=50, random_state=seed),random_state=seed)
        X_train_cont_imp = pd.DataFrame(imp_cont.fit_transform(X_train_cont),columns=CONTINUOUS_FEATURES)
        X_val_cont_imp = pd.DataFrame(imp_cont.transform(X_val_cont),columns=CONTINUOUS_FEATURES)

        imp_cat = IterativeImputer(estimator=RandomForestClassifier(n_estimators=50, random_state=seed),random_state=seed)
        X_train_cat_imp = pd.DataFrame(
        imp_cat.fit_transform(X_train_cat),columns=new_cat_features)
        X_val_cat_imp = pd.DataFrame(imp_cat.transform(X_val_cat),columns=new_cat_features)

        X_train = pd.concat([X_train_cont_imp, X_train_cat_imp], axis=1)
        X_val = pd.concat([X_val_cont_imp, X_val_cat_imp], axis=1)

        # --- FEATURE SELECTION A: univariate filter (p < 0.3) ---
        _, p_cont = f_classif(X_train[CONTINUOUS_FEATURES], y_train)
        selected_cont = [CONTINUOUS_FEATURES[i] for i in range(len(CONTINUOUS_FEATURES)) if p_cont[i] < 0.3]

        scaler_chi = MinMaxScaler()
        X_train_cat_scaled = scaler_chi.fit_transform(X_train[new_cat_features])
        _, p_cat = chi2(X_train_cat_scaled, y_train)
        selected_cat = [new_cat_features[i] for i in range(len(new_cat_features))if p_cat[i] < 0.3]

        filtered_features = selected_cont + selected_cat

        X_train_filtered = X_train[filtered_features]
        X_val_filtered = X_val[filtered_features]

        # --- FEATURE SELECTION B: forward sequential selection (LightGBM, f1_macro) ---
        sfs = SequentialFeatureSelector(
            estimator=lgb.LGBMClassifier(random_state=seed, n_jobs=-1, verbose=-1),
            direction='forward',
            scoring='f1_macro',
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
            n_jobs=-1
        )
        sfs.fit(X_train_filtered, y_train)
        final_features = list(X_train_filtered.columns[sfs.get_support()])

        # Record which features were selected in this fold
        for f in final_features:
            feature_selection_counts[f] += 1
        total_folds += 1

        X_train_final = X_train_filtered[final_features]
        X_val_final = X_val_filtered[final_features]

        # --- OPTUNA HYPERPARAMETER SEARCH ---
        smote_opt = SMOTE(random_state=seed)
        X_train_opt_smote, y_train_opt_smote = smote_opt.fit_resample(X_train_final, y_train)
        scaler_opt = MinMaxScaler()
        X_train_opt_scaled = scaler_opt.fit_transform(X_train_opt_smote)

        def objective(trial):
            params = {
                'learning_rate':    trial.suggest_categorical('learning_rate', [0.01, 0.03, 0.05, 0.07, 0.1]),
                'n_estimators':     trial.suggest_int('n_estimators', 100, 250, step=50),
                'max_depth':        trial.suggest_int('max_depth', 2, 4, step=1),
                'random_state': seed, 'n_jobs': -1, 'verbose': -1,
            }
            # 3-fold CV on the SMOTE'd train set; returns mean macro-F1
            inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
            fold_f1s = []
            for inner_train_idx, inner_val_idx in inner_cv.split(X_train_opt_scaled, y_train_opt_smote):
                clf_t = lgb.LGBMClassifier(**params)
                clf_t.fit(X_train_opt_scaled[inner_train_idx], y_train_opt_smote.iloc[inner_train_idx])
                probs_t = clf_t.predict_proba(X_train_opt_scaled[inner_val_idx])[:, 1]
                preds_t = (probs_t >= 0.5).astype(int)
                fold_f1s.append(f1_score(y_train_opt_smote.iloc[inner_val_idx], preds_t, average='macro'))
            return np.mean(fold_f1s)

        # TPE sampler; maximise macro-F1 over 25 trials
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=25, show_progress_bar=False)
        best_params = study.best_params
        best_params.update({'random_state': seed, 'n_jobs': -1, 'verbose': -1})

        # --- INNER CV: find optimal classification threshold ---
        # Uses the tuned params; threshold is averaged across inner folds
        inner_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        inner_thresholds = []

        for i_tr_idx, i_val_idx in inner_skf.split(X_train_final, y_train):
            X_inner_train, X_inner_val = X_train_final.iloc[i_tr_idx], X_train_final.iloc[i_val_idx]
            y_inner_train, y_inner_val = y_train.iloc[i_tr_idx], y_train.iloc[i_val_idx]

            smote_i = SMOTE(random_state=seed)
            X_inner_train_smote, y_inner_train_smote = smote_i.fit_resample(X_inner_train, y_inner_train)

            scaler_i = MinMaxScaler()
            X_inner_train_scaled = scaler_i.fit_transform(X_inner_train_smote)
            X_inner_val_scaled = scaler_i.transform(X_inner_val)

            clf_i = lgb.LGBMClassifier(**best_params)
            clf_i.fit(X_inner_train_scaled, y_inner_train_smote)
            probs_i = clf_i.predict_proba(X_inner_val_scaled)[:, 1]

            best_f1_i, best_thresh_i = 0.0, 0.50
            for thresh in np.linspace(0.05, 0.95, 100):
                preds_i = (probs_i >= thresh).astype(int)
                f1_i = f1_score(y_inner_val, preds_i, average='macro')
                if f1_i > best_f1_i:
                    best_f1_i = f1_i
                    best_thresh_i = thresh
            inner_thresholds.append(best_thresh_i)

        optimal_threshold = np.mean(inner_thresholds)

        # --- FINAL TRAIN & EVALUATE ON OUTER FOLD ---
        smote = SMOTE(random_state=seed)
        X_train_smote, y_train_smote = smote.fit_resample(X_train_final, y_train)

        scaler = MinMaxScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_smote), columns=final_features)
        X_val_scaled = pd.DataFrame(scaler.transform(X_val_final), columns=final_features)

        clf = lgb.LGBMClassifier(**best_params)
        clf.fit(X_train_scaled, y_train_smote)

        y_prob = clf.predict_proba(X_val_scaled)[:, 1]
        y_pred = (y_prob >= optimal_threshold).astype(int)

        fold_auc   = roc_auc_score(y_val, y_prob)
        fold_f1    = f1_score(y_val, y_pred, average='macro')
        fold_prec  = precision_score(y_val, y_pred, average='macro')
        fold_rec   = recall_score(y_val, y_pred, average='macro')

        all_auc.append(fold_auc)
        all_f1.append(fold_f1)
        all_prec.append(fold_prec)
        all_rec.append(fold_rec)

        # Store everything needed to reconstruct plots or retrain for this fold
        fold_snapshots.append({
            'fold_global_idx': total_folds - 1,
            'seed': seed,
            'fold_idx': fold_idx,
            'auc': fold_auc,
            'f1': fold_f1,
            'best_params': best_params,
            'optimal_threshold': optimal_threshold,
            'final_features': final_features,
            'y_val': y_val,
            'y_prob': y_prob,
            'ids_val': ids_val,
            'clf': clf,
            'X_val_sc': X_val_scaled,
        })

mean_auc  = np.mean(all_auc);  std_auc  = np.std(all_auc)
mean_f1   = np.mean(all_f1);   std_f1   = np.std(all_f1)
mean_prec = np.mean(all_prec); std_prec = np.std(all_prec)
mean_rec  = np.mean(all_rec);  std_rec  = np.std(all_rec)

# SELECT MEDIAN FOLD (closest AUC to mean) 
auc_dists = [abs(s['auc'] - mean_auc) for s in fold_snapshots]
median_fold_idx = int(np.argmin(auc_dists))
median_snap = fold_snapshots[median_fold_idx]

for k, v in median_snap['best_params'].items():
    print(f"  {k}: {v}")

# Features selected CSV generation
robust_df = pd.DataFrame([
    {'feature': f, 'selection_count': c, 'selection_rate': c / total_folds}
    for f, c in sorted(feature_selection_counts.items(), key=lambda x: -x[1])
])
robust_df.to_csv('outputs/feature_selection_frequency.csv', index=False)

#  RETRAIN ON FULL DATA using median fold's best params FOR SHAP PLOT

median_best_params    = median_snap['best_params']
median_final_features = median_snap['final_features']
median_threshold      = median_snap['optimal_threshold']

# One-hot encode on full data
X_full_cat = pd.get_dummies(X_full[CATEGORICAL_FEATURES], drop_first=True)
new_cat_features_full = list(X_full_cat.columns)

X_full_cont = X_full[CONTINUOUS_FEATURES].copy()
X_full_cat  = X_full_cat.reset_index(drop=True)
X_full_cont = X_full_cont.reset_index(drop=True)

# Iterative imputation on full data
imp_cont_full = IterativeImputer(estimator=RandomForestRegressor(n_estimators=50, random_state=42),random_state=42)
X_full_cont_imp = pd.DataFrame(imp_cont_full.fit_transform(X_full_cont),columns=CONTINUOUS_FEATURES)

imp_cat_full = IterativeImputer(estimator=RandomForestClassifier(n_estimators=50, random_state=42),random_state=42)
X_full_cat_imp = pd.DataFrame(imp_cat_full.fit_transform(X_full_cat),columns=new_cat_features_full)

X_full_imp = pd.concat([X_full_cont_imp, X_full_cat_imp], axis=1)

# Keep only the features selected by the median fold
X_full_final = X_full_imp.reindex(columns=median_final_features, fill_value=0)

# SMOTE + scale on full data
smote_full = SMOTE(random_state=42)
X_full_s, y_full_s = smote_full.fit_resample(X_full_final, y_full)

scaler_full = MinMaxScaler()
X_full_sc = pd.DataFrame(
    scaler_full.fit_transform(X_full_s),
    columns=median_final_features)

# Fit final model
clf_full = lgb.LGBMClassifier(**median_best_params)
clf_full.fit(X_full_sc, y_full_s)
print("Full-data retrain complete.")

# Original scaled data used for SHAP background
X_original_scaled = pd.DataFrame(scaler_full.transform(X_full_final),columns=median_final_features)


# --- PLOTS --- 
median_y_val   = median_snap['y_val']
median_y_prob  = median_snap['y_prob']
median_thresh  = median_snap['optimal_threshold']
median_y_pred  = (median_y_prob >= median_thresh).astype(int)
median_clf     = median_snap['clf']
median_X_val   = median_snap['X_val_sc']

fold_label = f"seed={median_snap['seed']}, fold {median_snap['fold_idx']+1}"

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f'LightGBM Model — Median Fold ({fold_label})', fontsize=14, fontweight='bold')

cm = confusion_matrix(median_y_val, median_y_pred)
display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Other', 'Schizophrenia'])
display.plot(ax=axes[0], colorbar=False, cmap='Greens')
axes[0].set_title(f'Confusion Matrix\n(Threshold: {median_thresh:.2f})')

fpr, tpr, _ = roc_curve(median_y_val, median_y_prob)
axes[1].plot(fpr, tpr, color='darkgreen', lw=2, label=f'AUC = {median_snap["auc"]:.3f} (median fold)')
axes[1].plot([0, 1], [0, 1], 'k--', lw=1, label='Random guessing')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title(f'ROC Curve\nMean AUC = {mean_auc:.3f} ± {std_auc:.3f}')
axes[1].legend()

# Feature importances from the full-data model
raw_imp = clf_full.booster_.feature_importance(importance_type='gain')
rel_imp = raw_imp / raw_imp.sum() if raw_imp.sum() > 0 else raw_imp
imp_df = pd.DataFrame({'feature': median_final_features, 'importance': rel_imp})
imp_df = imp_df[imp_df['importance'] > 0].sort_values('importance', ascending=False)
axes[2].barh(imp_df['feature'], imp_df['importance'], color='darkgreen')
axes[2].set_title('Feature Importances (full-data model)')
axes[2].set_xlabel('Relative Importance')
axes[2].set_xlim(0, 1.0)
axes[2].invert_yaxis()

plt.tight_layout()
plt.savefig('outputs/diagnosis_lightgbm.png', dpi=150, bbox_inches='tight')
plt.close()

# SHAP — full-data model on original scaled data
explainer_full = shap.TreeExplainer(clf_full, data=X_original_scaled)
shap_values_full = explainer_full.shap_values(X_original_scaled)

shap_values_pos = (shap_values_full[1]if isinstance(shap_values_full, list) else shap_values_full)

plt.figure(figsize=(10, 6))
plt.title("SHAP Summary Plot: Selected Features (LightGBM — full-data model)", fontsize=14, fontweight='bold')
shap.summary_plot(shap_values_pos, X_original_scaled, show=False)
plt.tight_layout()
plt.savefig('outputs/shap_summary_lightgbm.png', dpi=150, bbox_inches='tight')
plt.close()

# --- SUMMARY CSV ---
summary = {
    'Algorithm':      'LightGBM',
    'Parameters':     'Optuna-tuned per fold',
    'Mean AUC':       f"{mean_auc:.3f}",
    'Std AUC':        f"{std_auc:.3f}",
    'Mean F1':        f"{mean_f1:.3f}",
    'Std F1':         f"{std_f1:.3f}",
    'Mean Precision': f"{mean_prec:.3f}",
    'Std Precision':  f"{std_prec:.3f}",
    'Mean Recall':    f"{mean_rec:.3f}",
    'Std Recall':     f"{std_rec:.3f}",
    'N Folds':        len(all_auc),
}
pd.DataFrame([summary]).to_csv('outputs/summary_lightgbm.csv', index=False)
